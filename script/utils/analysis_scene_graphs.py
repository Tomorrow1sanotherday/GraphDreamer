import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


class SceneGraphAnalyzer:
    """场景图分析器，用于分析场景图JSON文件中每个subject的数量"""
    
    def __init__(self, file_path: str):
        """
        初始化分析器
        
        Args:
            file_path: JSON文件路径
        """
        self.file_path = Path(file_path)
        self.data = None
        self.subject_counts = None
        
    def load_data(self) -> None:
        """加载JSON文件数据"""
        if not self.file_path.exists():
            raise FileNotFoundError(f"文件不存在: {self.file_path}")
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
    
    def count_subjects(self) -> Dict[str, int]:
        """
        统计每个subject的数量
        
        Returns:
            字典，key为subject名称，value为出现次数
        """
        if self.data is None:
            raise ValueError("请先调用load_data()加载数据")
        
        if 'results' not in self.data:
            raise ValueError("JSON文件格式错误：缺少'results'字段")
        
        subject_counter = Counter()
        
        for item in self.data['results']:
            if 'label_name' not in item:
                continue
            subject_name = item['label_name']
            subject_counter[subject_name] += 1
        
        self.subject_counts = dict(subject_counter)
        return self.subject_counts
    
    def check_subject_count(self, expected_count: int = 30) -> Tuple[bool, Dict[str, bool], List[str]]:
        """
        检查每个subject是否都有指定数量
        
        Args:
            expected_count: 期望的每个subject的数量，默认为30
            
        Returns:
            Tuple[是否全部符合, 每个subject是否符合的字典, 不符合的subject列表]
        """
        if self.subject_counts is None:
            self.count_subjects()
        
        all_valid = True
        subject_validity = {}
        invalid_subjects = []
        
        for subject, count in self.subject_counts.items():
            is_valid = (count == expected_count)
            subject_validity[subject] = is_valid
            
            if not is_valid:
                all_valid = False
                invalid_subjects.append(subject)
        
        return all_valid, subject_validity, invalid_subjects
    
    def get_statistics(self) -> Dict:
        """
        获取统计信息
        
        Returns:
            包含统计信息的字典
        """
        if self.subject_counts is None:
            self.count_subjects()
        
        total_subjects = len(self.subject_counts)
        total_entries = sum(self.subject_counts.values())
        counts = list(self.subject_counts.values())
        
        stats = {
            'total_subject_types': total_subjects,
            'total_entries': total_entries,
            'min_count': min(counts) if counts else 0,
            'max_count': max(counts) if counts else 0,
            'avg_count': sum(counts) / len(counts) if counts else 0,
            'subject_counts': self.subject_counts
        }
        
        return stats
    
    def analyze(self, expected_count: int = 30) -> Dict:
        """
        执行完整分析
        
        Args:
            expected_count: 期望的每个subject的数量，默认为30
            
        Returns:
            包含分析结果的字典
        """
        self.load_data()
        self.count_subjects()
        
        all_valid, subject_validity, invalid_subjects = self.check_subject_count(expected_count)
        statistics = self.get_statistics()
        
        result = {
            'file_path': str(self.file_path),
            'expected_count': expected_count,
            'all_subjects_valid': all_valid,
            'statistics': statistics,
            'subject_validity': subject_validity,
            'invalid_subjects': invalid_subjects,
            'invalid_subjects_detail': {
                subject: self.subject_counts[subject] 
                for subject in invalid_subjects
            } if invalid_subjects else {}
        }
        
        return result
    
    def print_report(self, expected_count: int = 30) -> None:
        """
        打印分析报告
        
        Args:
            expected_count: 期望的每个subject的数量，默认为30
        """
        result = self.analyze(expected_count)
        
        print("=" * 60)
        print("场景图分析报告")
        print("=" * 60)
        print(f"文件路径: {result['file_path']}")
        print(f"期望每个subject数量: {expected_count}")
        print(f"\n总体统计:")
        print(f"  - Subject类型总数: {result['statistics']['total_subject_types']}")
        print(f"  - 总条目数: {result['statistics']['total_entries']}")
        print(f"  - 每个subject最小数量: {result['statistics']['min_count']}")
        print(f"  - 每个subject最大数量: {result['statistics']['max_count']}")
        print(f"  - 每个subject平均数量: {result['statistics']['avg_count']:.2f}")
        print(f"\n验证结果:")
        print(f"  - 所有subject是否符合要求: {'是' if result['all_subjects_valid'] else '否'}")
        
        if result['invalid_subjects']:
            print(f"\n不符合要求的subject ({len(result['invalid_subjects'])}个):")
            for subject in result['invalid_subjects']:
                actual_count = result['invalid_subjects_detail'][subject]
                print(f"  - {subject}: {actual_count}个 (期望{expected_count}个)")
        else:
            print(f"\n✓ 所有subject都符合要求，每个都有{expected_count}个")
        
        print("=" * 60)


def main():
    """主函数，用于命令行执行"""
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python analysis_scene_graphs.py <json_file_path> [expected_count]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    expected_count = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    
    analyzer = SceneGraphAnalyzer(file_path)
    analyzer.print_report(expected_count)


if __name__ == "__main__":
    main()

