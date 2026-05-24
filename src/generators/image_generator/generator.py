"""
Image Generator Module

This module provides functionality to generate images from scene graphs
using Text-to-Image diffusion models (e.g., Stable Diffusion XL).
"""

import json
import re
import logging
import multiprocessing
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict
from abc import ABC, abstractmethod

import torch
from tqdm import tqdm

from .data_types import (
    SceneGraphItem,
    ImageGenerationTask,
    ImageGenerationResult,
    SyntheticImageDataset,
    GeneratorConfig
)

logger = logging.getLogger(__name__)


class BaseImageGenerator(ABC):
    """Abstract base class for image generators."""
    
    @abstractmethod
    def generate_image(self, prompt: str, seed: int) -> Any:
        """Generate a single image from a prompt."""
        pass
    
    @abstractmethod
    def save_image(self, image: Any, filepath: Path) -> bool:
        """Save an image to a file."""
        pass


class DiffusionImageGenerator(BaseImageGenerator):
    """
    Image generator using diffusion models (e.g., Stable Diffusion XL).
    
    This class handles:
    - Loading and managing diffusion models
    - Generating images from text prompts
    - Saving images to disk
    """
    
    def __init__(self, config: GeneratorConfig):
        """
        Initialize the image generator.
        
        Args:
            config: Generator configuration
        """
        self.config = config
        self.pipe = None
        self._initialized = False
        
        logger.info(f"Created DiffusionImageGenerator with config: {config.model_name}")
    
    def _get_torch_dtype(self) -> torch.dtype:
        """Get torch dtype from config string."""
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16
        }
        return dtype_map.get(self.config.torch_dtype, torch.float16)
    
    def initialize(self) -> None:
        """Initialize the diffusion pipeline."""
        if self._initialized:
            return
        
        try:
            from diffusers import DiffusionPipeline
        except ImportError:
            raise ImportError(
                "diffusers package not found. "
                "Please install it: pip install diffusers"
            )
        
        logger.info(f"Loading model: {self.config.model_name}")
        
        torch_dtype = self._get_torch_dtype()
        
        # Prepare pipeline kwargs
        pipe_kwargs = {
            "torch_dtype": torch_dtype,
        }
        
        # Add safetensors variant for compatible models
        if self.config.use_safetensors:
            pipe_kwargs["use_safetensors"] = True
            if torch_dtype == torch.float16:
                pipe_kwargs["variant"] = "fp16"
        
        self.pipe = DiffusionPipeline.from_pretrained(
            self.config.model_name,
            **pipe_kwargs
        )
        
        self.pipe.to(self.config.device)
        
        # Disable per-step progress bar inside the pipeline
        self.pipe.set_progress_bar_config(disable=True)
        
        # Enable memory optimizations if available
        if self.config.enable_xformers:
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                logger.info("Enabled xformers memory efficient attention")
            except Exception as e:
                logger.warning(f"Could not enable xformers: {e}")
        
        self._initialized = True
        logger.info("Model loaded successfully!")
    
    def generate_image(self, prompt: str, seed: int) -> Any:
        """
        Generate a single image from a prompt.
        
        Args:
            prompt: Text prompt for image generation
            seed: Random seed for reproducibility
            
        Returns:
            Generated PIL Image
        """
        if not self._initialized:
            self.initialize()
        
        generator = torch.Generator(device=self.config.device).manual_seed(seed)
        
        # Build pipe arguments
        pipe_kwargs = {
            'prompt': prompt,
            'generator': generator,
            'num_inference_steps': self.config.num_inference_steps,
            'width': self.config.image_width,
            'height': self.config.image_height
        }
        
        # Only add guidance_scale if it's specified
        if self.config.guidance_scale is not None:
            pipe_kwargs['guidance_scale'] = self.config.guidance_scale
        
        image = self.pipe(**pipe_kwargs).images[0]
        
        return image
    
    def save_image(self, image: Any, filepath: Path) -> bool:
        """
        Save an image to a file.
        
        Args:
            image: PIL Image to save
            filepath: Path to save the image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            image.save(filepath)
            return True
        except Exception as e:
            logger.error(f"Failed to save image to {filepath}: {e}")
            return False
    
    def unload(self) -> None:
        """Unload the model to free memory."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            self._initialized = False
            torch.cuda.empty_cache()
            logger.info("Model unloaded")


class ImageGenerationManager:
    """
    Manager for batch image generation with streaming output.
    
    This class handles:
    - Loading scene graphs from input JSON
    - Managing generation tasks
    - Handling resume/checkpoint functionality
    - Streaming output to JSON file
    """
    
    def __init__(
        self,
        generator: BaseImageGenerator,
        output_dir: str,
        output_json_path: str,
        config: GeneratorConfig,
        dataset_name: str = "synthetic"
    ):
        """
        Initialize the generation manager.
        
        Images are saved in per-class subdirs: output_dir / 001.ClassName / 000.png, 001.png, ...
        
        Args:
            generator: Image generator instance
            output_dir: Directory to save generated images (root for class subdirs)
            output_json_path: Path to save output JSON
            config: Generator configuration
            dataset_name: Name of the dataset
        """
        self.generator = generator
        self.output_dir = Path(output_dir)
        self.output_json_path = Path(output_json_path)
        self.config = config
        self.dataset_name = dataset_name
        self._task_id_to_class_and_index: Optional[Dict[int, Tuple[str, int]]] = None
        self._class_folder_to_task_ids: Optional[Dict[str, List[int]]] = None
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ImageGenerationManager initialized")
        logger.info(f"  Output directory: {self.output_dir} (per-class subdirs: 001.ClassName/000.png)")
        logger.info(f"  Output JSON: {self.output_json_path}")
    
    def load_scene_graphs(self, input_path: str) -> List[SceneGraphItem]:
        """
        Load scene graphs from input JSON file.
        
        Args:
            input_path: Path to the input JSON file
            
        Returns:
            List of SceneGraphItem objects
        """
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        results = data.get("results", [])
        items = [SceneGraphItem.from_dict(item) for item in results]
        
        logger.info(f"Loaded {len(items)} scene graph items from {input_path}")
        return items
    
    def set_tasks_for_by_class(self, tasks: List[ImageGenerationTask]) -> None:
        """
        Build task_id -> (class_folder, local_index) for per-class layout.
        Must be called after create_tasks().
        """
        by_label = defaultdict(list)
        for t in tasks:
            by_label[t.label].append(t)
        self._task_id_to_class_and_index = {}
        self._class_folder_to_task_ids = {}
        for label in sorted(by_label.keys()):
            sorted_tasks = sorted(by_label[label], key=lambda t: t.task_id)
            label_name = sorted_tasks[0].label_name
            # Sanitize path separators so class names like "F-16A/B" don't
            # create nested directories on disk. Matches the real dataset
            # convention (e.g. FGVC-Aircraft uses "F-16A-B").
            safe_label_name = label_name.replace("/", "-").replace("\\", "-")
            class_folder = f"{label + 1:03d}.{safe_label_name}"
            task_ids = [t.task_id for t in sorted_tasks]
            self._class_folder_to_task_ids[class_folder] = task_ids
            for local_index, t in enumerate(sorted_tasks):
                self._task_id_to_class_and_index[t.task_id] = (class_folder, local_index)
    
    def create_tasks(
        self,
        items: List[SceneGraphItem]
    ) -> List[ImageGenerationTask]:
        """
        Create image generation tasks from scene graph items.
        
        Args:
            items: List of SceneGraphItem objects
            
        Returns:
            List of ImageGenerationTask objects
        """
        tasks = [
            ImageGenerationTask.from_scene_graph_item(item)
            for item in items
        ]
        logger.info(f"Created {len(tasks)} generation tasks")
        return tasks
    
    def get_existing_image_ids(self) -> Set[int]:
        """
        Scan output directory and get IDs of existing images (per-class subdirs).
        
        Returns:
            Set of existing image IDs
        """
        existing_ids = set()
        if not self.output_dir.exists() or self._class_folder_to_task_ids is None:
            return existing_ids
        
        # Per-class layout: output_dir / 001.ClassName / 000.png, 001.png, ...
        subdir_pattern = re.compile(r'^\d{3}\..+')
        ext = self.config.image_format
        file_pattern = re.compile(r'^(\d{3})\.' + re.escape(ext) + r'$')
        for subdir in self.output_dir.iterdir():
            if not subdir.is_dir() or not subdir_pattern.match(subdir.name):
                continue
            class_folder = subdir.name
            task_ids = self._class_folder_to_task_ids.get(class_folder)
            if not task_ids:
                continue
            for f in subdir.iterdir():
                if not f.is_file():
                    continue
                match = file_pattern.match(f.name)
                if match:
                    local_index = int(match.group(1))
                    if local_index < len(task_ids):
                        existing_ids.add(task_ids[local_index])
        return existing_ids
    
    def load_existing_json_results(self) -> Dict[int, ImageGenerationResult]:
        """
        Load existing results from output JSON file.
        
        Returns:
            Dictionary mapping task_id to ImageGenerationResult
        """
        existing_results = {}
        
        if not self.output_json_path.exists():
            return existing_results
        
        try:
            with open(self.output_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for item in data.get("results", []):
                result = ImageGenerationResult.from_dict(item)
                existing_results[result.id] = result
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Could not load existing JSON file: {e}")
        
        return existing_results
    
    def build_image_filename(self, task_id: int) -> str:
        """Build relative path for an image (class_subdir/local_index.ext)."""
        class_folder, local_index = self._task_id_to_class_and_index[task_id]
        return f"{class_folder}/{local_index:03d}.{self.config.image_format}"
    
    def build_image_path(self, task_id: int) -> Path:
        """Build full path for an image."""
        class_folder, local_index = self._task_id_to_class_and_index[task_id]
        return self.output_dir / class_folder / f"{local_index:03d}.{self.config.image_format}"


class StreamingJsonWriter:
    """
    Handles true streaming output of generation results to JSON file.
    
    This class writes results incrementally as they are generated,
    ensuring that partial results are preserved even if the process
    is interrupted.
    """
    
    def __init__(
        self,
        output_path: str,
        dataset_name: str,
        image_dir: str,
        total_expected: int = 0
    ):
        """
        Initialize the streaming writer.
        
        Args:
            output_path: Path to the output JSON file
            dataset_name: Name of the dataset
            image_dir: Directory containing generated images
            total_expected: Expected total number of results
        """
        self.output_path = Path(output_path)
        self.dataset_name = dataset_name
        self.image_dir = image_dir
        self.total_expected = total_expected
        self.result_count = 0
        self.first_item = True
        self.file_handle = None
        
        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
    
    def open(self) -> None:
        """Open the file and write the JSON header."""
        self.file_handle = open(self.output_path, 'w', encoding='utf-8')
        # Write JSON header
        self.file_handle.write('{\n')
        self.file_handle.write(f'  "dataset": {json.dumps(self.dataset_name)},\n')
        self.file_handle.write(f'  "total": {self.total_expected},\n')
        self.file_handle.write(f'  "image_dir": {json.dumps(self.image_dir)},\n')
        self.file_handle.write('  "results": [\n')
        self.file_handle.flush()
    
    def add_result(self, result: ImageGenerationResult) -> None:
        """
        Add a result and immediately write to file (streaming).
        
        Args:
            result: ImageGenerationResult to add
        """
        if self.file_handle is None:
            self.open()
        
        # Write comma separator if not first item
        if not self.first_item:
            self.file_handle.write(',\n')
        else:
            self.first_item = False
        
        # Convert result to JSON string with proper indentation
        result_dict = result.to_dict()
        json_str = json.dumps(result_dict, indent=2, ensure_ascii=False)
        
        # Add indentation to each line (4 spaces for items in results array)
        indented_lines = []
        for line in json_str.split('\n'):
            if line.strip():
                indented_lines.append('    ' + line)
            else:
                indented_lines.append(line)
        indented_json = '\n'.join(indented_lines)
        
        self.file_handle.write(indented_json)
        self.file_handle.flush()  # Flush immediately for true streaming
        self.result_count += 1
    
    def flush(self) -> None:
        """Flush the file buffer to disk."""
        if self.file_handle:
            self.file_handle.flush()
    
    def close(self) -> int:
        """
        Close the file with proper JSON ending.
        
        Returns:
            Total number of results written
        """
        if self.file_handle:
            # Write JSON footer
            self.file_handle.write('\n  ]\n')
            self.file_handle.write('}\n')
            self.file_handle.close()
            self.file_handle = None
        
        return self.result_count
    
    def finalize(self) -> int:
        """
        Finalize and close the output file.
        
        Note: For streaming output, results are written in generation order.
        If sorted order is needed, use finalize_sorted() instead.
        
        Returns:
            Total number of results
        """
        return self.close()
    
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False


def _generate_on_single_gpu(
    gpu_id: int,
    all_tasks: List[ImageGenerationTask],
    assigned_tasks: List[ImageGenerationTask],
    output_dir: str,
    temp_json_path: str,
    config_dict: Dict[str, Any],
    dataset_name: str,
    existing_image_ids: Set[int],
    existing_results: Dict[int, ImageGenerationResult]
) -> Tuple[int, int, int]:
    """
    Generate images on a single GPU (worker function for multiprocessing).
    
    Args:
        gpu_id: GPU device ID (e.g., 0, 1, 2)
        all_tasks: All tasks (for writing complete results)
        assigned_tasks: Tasks assigned to this GPU for generation
        output_dir: Directory to save generated images
        temp_json_path: Path to temporary JSON file for this GPU
        config_dict: GeneratorConfig as dictionary
        dataset_name: Name of the dataset
        existing_image_ids: Set of existing image IDs
        existing_results: Dictionary of existing results
        
    Returns:
        Tuple of (generated_count, skipped_count, failed_count)
    """
    # Reconstruct config with GPU-specific device
    config = GeneratorConfig(**config_dict)
    config.device = f"cuda:{gpu_id}"
    
    # Initialize generator for this GPU
    generator = DiffusionImageGenerator(config)
    manager = ImageGenerationManager(
        generator=generator,
        output_dir=output_dir,
        output_json_path=temp_json_path,
        config=config,
        dataset_name=dataset_name
    )
    manager.set_tasks_for_by_class(all_tasks)
    
    # Filter assigned tasks that need generation
    pending_tasks = [t for t in assigned_tasks if t.task_id not in existing_image_ids]
    
    # Initialize streaming writer
    # Each GPU writes all existing results, but only generates assigned tasks
    # This ensures complete results in each temp file, and merge will deduplicate
    writer = StreamingJsonWriter(
        output_path=temp_json_path,
        dataset_name=dataset_name,
        image_dir=str(manager.output_dir),
        total_expected=len(all_tasks)
    )
    
    writer.open()
    
    # Write all existing results first (for complete output in temp file)
    # Count skipped only for assigned tasks
    skipped_count = 0
    assigned_task_ids = {t.task_id for t in assigned_tasks}
    for task in all_tasks:
        if task.task_id in existing_results:
            writer.add_result(existing_results[task.task_id])
            if task.task_id in assigned_task_ids:
                skipped_count += 1
        elif task.task_id in existing_image_ids:
            image_path = str(manager.build_image_path(task.task_id))
            result = ImageGenerationResult.from_task(task, image_path)
            writer.add_result(result)
            if task.task_id in assigned_task_ids:
                skipped_count += 1
    
    # Skip model loading if nothing to generate on this GPU
    if len(pending_tasks) == 0:
        total = writer.finalize()
        return 0, skipped_count, 0
    
    # Initialize the generator model
    generator.initialize()
    
    # Generate images for assigned tasks
    generated_count = 0
    failed_count = 0
    
    logger.info(f"GPU {gpu_id}: Processing {len(pending_tasks)} tasks...")
    
    for task in tqdm(pending_tasks, desc=f"GPU {gpu_id}", position=gpu_id):
        try:
            # Calculate seed for this task
            item_seed = config.seed + task.task_id
            
            # Generate image
            image = generator.generate_image(task.caption, item_seed)
            
            # Save image
            image_path = manager.build_image_path(task.task_id)
            if generator.save_image(image, image_path):
                result = ImageGenerationResult.from_task(task, str(image_path))
                writer.add_result(result)
                generated_count += 1
            else:
                result = ImageGenerationResult.from_task(task, None)
                writer.add_result(result)
                failed_count += 1
                
        except Exception as e:
            logger.error(f"GPU {gpu_id}: Error generating image for task {task.task_id}: {e}")
            result = ImageGenerationResult.from_task(task, None)
            writer.add_result(result)
            failed_count += 1
    
    # Finalize output
    writer.finalize()
    
    # Unload model
    generator.unload()
    
    logger.info(f"GPU {gpu_id}: Completed - Generated: {generated_count}, Skipped: {skipped_count}, Failed: {failed_count}")
    
    return generated_count, skipped_count, failed_count


def _merge_json_files(
    temp_json_paths: List[str],
    output_json_path: str,
    dataset_name: str,
    image_dir: str,
    total_expected: int
) -> None:
    """
    Merge multiple temporary JSON files into a single output file.
    
    Since each GPU writes all results (including existing ones), we need to
    merge them intelligently, preferring results with image_path when available.
    
    Args:
        temp_json_paths: List of paths to temporary JSON files
        output_json_path: Path to final output JSON file
        dataset_name: Name of the dataset
        image_dir: Directory containing images
        total_expected: Expected total number of results
    """
    all_results = {}
    
    # Load all results from temporary files
    # Prefer results with image_path (newly generated) over those without
    for temp_path in temp_json_paths:
        temp_path_obj = Path(temp_path)
        if not temp_path_obj.exists():
            logger.warning(f"Temporary file not found: {temp_path}")
            continue
        
        try:
            with open(temp_path_obj, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for item in data.get("results", []):
                result = ImageGenerationResult.from_dict(item)
                result_id = result.id
                
                # Prefer result with image_path, or keep existing if both have paths
                if result_id not in all_results:
                    all_results[result_id] = result
                elif result.image_path and not all_results[result_id].image_path:
                    # Replace with result that has image_path
                    all_results[result_id] = result
                elif result.image_path and all_results[result_id].image_path:
                    # Both have paths, prefer the one that's not None
                    if result.image_path != all_results[result_id].image_path:
                        # Keep the first one found (should be same path anyway)
                        pass
        except Exception as e:
            logger.error(f"Error loading temporary file {temp_path}: {e}")
    
    # Write merged results to output file
    writer = StreamingJsonWriter(
        output_path=output_json_path,
        dataset_name=dataset_name,
        image_dir=image_dir,
        total_expected=total_expected
    )
    
    writer.open()
    
    # Write results sorted by ID
    for result_id in sorted(all_results.keys()):
        writer.add_result(all_results[result_id])
    
    writer.finalize()
    
    # Clean up temporary files
    for temp_path in temp_json_paths:
        try:
            Path(temp_path).unlink()
            logger.debug(f"Deleted temporary file: {temp_path}")
        except Exception as e:
            logger.warning(f"Could not delete temporary file {temp_path}: {e}")


def generate_images_batch(
    input_json_path: str,
    output_dir: str,
    output_json_path: str,
    config: Optional[GeneratorConfig] = None,
    dataset_name: str = "synthetic",
    resume: bool = True,
    gpu_ids: Optional[List[int]] = None
) -> Tuple[int, int]:
    """
    Batch generate images from scene graphs JSON file.
    
    This is the main entry point for batch image generation.
    Supports single GPU or multi-GPU parallel generation.
    Results are written to JSON file in streaming mode (immediately flushed
    after each image generation).
    
    Args:
        input_json_path: Path to input scene graphs JSON file
        output_dir: Directory to save generated images
        output_json_path: Path to save output JSON with image paths
        config: Generator configuration (uses defaults if None)
        dataset_name: Name of the dataset
        resume: Whether to resume from existing progress
        gpu_ids: List of GPU IDs to use (e.g., [0, 1, 2]). If None, uses single GPU from config.
        
    Returns:
        Tuple of (total_generated, total_skipped)
    """
    # Use default config if not provided
    if config is None:
        config = GeneratorConfig()
    
    # Initialize manager for loading tasks and checking existing files
    # Don't initialize the generator model here to avoid CUDA initialization issues
    # The model will be initialized in each GPU process
    temp_config = GeneratorConfig(**config.to_dict())
    temp_config.device = "cpu"  # Use CPU for manager operations
    temp_generator = DiffusionImageGenerator(temp_config)
    manager = ImageGenerationManager(
        generator=temp_generator,
        output_dir=output_dir,
        output_json_path=output_json_path,
        config=config,  # Keep original config for actual generation
        dataset_name=dataset_name
    )
    
    # Load scene graphs
    items = manager.load_scene_graphs(input_json_path)
    tasks = manager.create_tasks(items)
    manager.set_tasks_for_by_class(tasks)
    
    # Check for existing progress if resuming
    existing_image_ids = set()
    existing_results = {}
    
    if resume:
        logger.info("Checking for existing generated images...")
        existing_image_ids = manager.get_existing_image_ids()
        if existing_image_ids:
            logger.info(f"Found {len(existing_image_ids)} existing images")
        
        existing_results = manager.load_existing_json_results()
        if existing_results:
            logger.info(f"Found {len(existing_results)} existing JSON results")
    
    # Determine GPU configuration
    if gpu_ids is None or len(gpu_ids) == 0:
        # Single GPU mode - extract from device parameter
        gpu_ids = [0] if config.device.startswith("cuda") else []
        if config.device.startswith("cuda:"):
            try:
                gpu_id = int(config.device.split(":")[1])
                gpu_ids = [gpu_id]
            except (ValueError, IndexError):
                gpu_ids = [0]
    else:
        # GPU IDs were explicitly provided - update config.device to match
        # This ensures the device string matches the specified GPU ID
        if len(gpu_ids) == 1 and config.device.startswith("cuda"):
            config.device = f"cuda:{gpu_ids[0]}"
    
    num_gpus = len(gpu_ids)
    
    logger.info(f"Total tasks: {len(tasks)}")
    logger.info(f"Already completed: {len(existing_image_ids)}")
    logger.info(f"To generate: {len(tasks) - len(existing_image_ids)}")
    logger.info(f"Using {num_gpus} GPU(s): {gpu_ids}")
    
    # Skip model loading if nothing to generate
    pending_tasks = [t for t in tasks if t.task_id not in existing_image_ids]
    if len(pending_tasks) == 0:
        logger.info("All images already generated, skipping model loading")
        # Write existing results to output JSON
        writer = StreamingJsonWriter(
            output_path=output_json_path,
            dataset_name=dataset_name,
            image_dir=str(manager.output_dir),
            total_expected=len(tasks)
        )
        writer.open()
        skipped_count = 0
        for task in tasks:
            if task.task_id in existing_results:
                writer.add_result(existing_results[task.task_id])
                skipped_count += 1
            elif task.task_id in existing_image_ids:
                image_path = str(manager.build_image_path(task.task_id))
                result = ImageGenerationResult.from_task(task, image_path)
                writer.add_result(result)
                skipped_count += 1
        writer.finalize()
        return 0, skipped_count
    
    # Multi-GPU parallel generation
    if num_gpus > 1:
        logger.info(f"Starting multi-GPU parallel generation on {num_gpus} GPUs...")
        
        # Split tasks evenly across GPUs
        tasks_per_gpu = len(pending_tasks) // num_gpus
        remainder = len(pending_tasks) % num_gpus
        
        task_chunks = []
        start_idx = 0
        for i in range(num_gpus):
            chunk_size = tasks_per_gpu + (1 if i < remainder else 0)
            end_idx = start_idx + chunk_size
            task_chunks.append(pending_tasks[start_idx:end_idx])
            start_idx = end_idx
        
        # Create temporary JSON paths for each GPU
        output_path_obj = Path(output_json_path)
        temp_json_paths = [
            str(output_path_obj.parent / f"{output_path_obj.stem}_gpu{gpu_id}{output_path_obj.suffix}")
            for gpu_id in gpu_ids
        ]
        
        # Prepare arguments for multiprocessing
        config_dict = config.to_dict()
        process_args = []
        for gpu_id, gpu_tasks, temp_json_path in zip(gpu_ids, task_chunks, temp_json_paths):
            process_args.append((
                gpu_id,
                tasks,  # All tasks for writing complete results
                gpu_tasks,  # Assigned tasks for this GPU
                output_dir,
                temp_json_path,
                config_dict,
                dataset_name,
                existing_image_ids,
                existing_results
            ))
        
        # Run generation on multiple GPUs in parallel
        # Must use 'spawn' method for CUDA/multiprocessing compatibility
        # 'fork' method causes "Cannot re-initialize CUDA in forked subprocess" error
        try:
            ctx = multiprocessing.get_context('spawn')
        except AttributeError:
            # Fallback for older Python versions (use default, but may have issues)
            ctx = multiprocessing
            logger.warning("Using default multiprocessing context. If you encounter CUDA errors, upgrade Python.")
        
        with ctx.Pool(processes=num_gpus) as pool:
            results = pool.starmap(_generate_on_single_gpu, process_args)
        
        # Aggregate results
        total_generated = sum(r[0] for r in results)
        total_skipped = sum(r[1] for r in results)
        total_failed = sum(r[2] for r in results)
        
        # Merge temporary JSON files
        logger.info("Merging results from all GPUs...")
        _merge_json_files(
            temp_json_paths=temp_json_paths,
            output_json_path=output_json_path,
            dataset_name=dataset_name,
            image_dir=str(manager.output_dir),
            total_expected=len(tasks)
        )
        
        logger.info(f"\nImage generation completed!")
        logger.info(f"  Total items: {len(tasks)}")
        logger.info(f"  Newly generated: {total_generated}")
        logger.info(f"  Skipped (existing): {total_skipped}")
        logger.info(f"  Failed: {total_failed}")
        logger.info(f"  Output JSON: {output_json_path}")
        logger.info(f"  Images directory: {output_dir}")
        
        return total_generated, total_skipped
    
    else:
        # Single GPU mode (original implementation)
        logger.info("Starting single GPU image generation...")
        
        generator = DiffusionImageGenerator(config)
        manager = ImageGenerationManager(
            generator=generator,
            output_dir=output_dir,
            output_json_path=output_json_path,
            config=config,
            dataset_name=dataset_name
        )
        manager.set_tasks_for_by_class(tasks)
        
        # Initialize streaming writer
        writer = StreamingJsonWriter(
            output_path=output_json_path,
            dataset_name=dataset_name,
            image_dir=str(manager.output_dir),
            total_expected=len(tasks)
        )
        
        writer.open()
        
        # Write existing results first
        skipped_count = 0
        for task in tasks:
            if task.task_id in existing_results:
                writer.add_result(existing_results[task.task_id])
                skipped_count += 1
            elif task.task_id in existing_image_ids:
                image_path = str(manager.build_image_path(task.task_id))
                result = ImageGenerationResult.from_task(task, image_path)
                writer.add_result(result)
                skipped_count += 1
        
        # Initialize the generator model
        generator.initialize()
        
        # Generate images
        generated_count = 0
        failed_count = 0
        
        for task in tqdm(pending_tasks, desc="Generating images"):
            try:
                item_seed = config.seed + task.task_id
                image = generator.generate_image(task.caption, item_seed)
                image_path = manager.build_image_path(task.task_id)
                if generator.save_image(image, image_path):
                    result = ImageGenerationResult.from_task(task, str(image_path))
                    writer.add_result(result)
                    generated_count += 1
                else:
                    result = ImageGenerationResult.from_task(task, None)
                    writer.add_result(result)
                    failed_count += 1
            except Exception as e:
                logger.error(f"Error generating image for task {task.task_id}: {e}")
                result = ImageGenerationResult.from_task(task, None)
                writer.add_result(result)
                failed_count += 1
        
        writer.finalize()
        generator.unload()
        
        logger.info(f"\nImage generation completed!")
        logger.info(f"  Total items: {len(tasks)}")
        logger.info(f"  Newly generated: {generated_count}")
        logger.info(f"  Skipped (existing): {skipped_count}")
        logger.info(f"  Failed: {failed_count}")
        logger.info(f"  Output JSON: {output_json_path}")
        logger.info(f"  Images directory: {output_dir}")
        
        return generated_count, skipped_count

