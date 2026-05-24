# GraphDreamer Synthetic Only 全量结果总结

本报告只统计 results 目录下所有评估文件中的 Synthetic Only 条目，不混入其他实验设置。

## 数据范围与口径

- 共纳入 39 个 Synthetic Only run，来源于 results 目录下 39 个 clip_eval JSON。
- 每个 run 都评估同一套测试集：5794 张测试图像，200 个类别。
- Synthetic Only 训练样本规模在不同 run 中有 3 种设置：6000、6200、12000。
- 类别排名按 39 个 run 的类别准确率简单平均计算，每个 run 权重相同。
- 误判去向分析依赖逐样本文件，共使用 28 个 detailed_predictions_synthetic_only.csv。

## 整体表现

- 39 个 Synthetic Only run 的平均总体准确率为 64.54%，中位数为 64.26%。
- 最好 run 是 [results/20260221_023857/clip_eval_20260221_025149.json](../results/20260221_023857/clip_eval_20260221_025149.json)，准确率 68.09%。
- 最差 run 是 [results/20260130_015313/clip_eval_20260130_021032.json](../results/20260130_015313/clip_eval_20260130_021032.json)，准确率 54.54%。
- 从类别平均准确率分布看，200 个类别中有 10 个类别平均准确率低于 10%，17 个类别低于 20%，72 个类别高于 80%。
- 跨类别的平均准确率均值为 64.24%，中位数为 66.89%，说明整体上存在明显长尾难类。

### 最好与最差的 run

最好 5 个 run：

| 排名 | run | 准确率 |
|---|---|---:|
| 1 | [results/20260221_023857/clip_eval_20260221_025149.json](../results/20260221_023857/clip_eval_20260221_025149.json) | 68.09% |
| 2 | [results/20260318_221302/clip_eval_20260318_224821.json](../results/20260318_221302/clip_eval_20260318_224821.json) | 67.66% |
| 3 | [results/20260224_133055/clip_eval_20260224_134454.json](../results/20260224_133055/clip_eval_20260224_134454.json) | 67.21% |
| 4 | [results/20260224_220944/clip_eval_20260224_222322.json](../results/20260224_220944/clip_eval_20260224_222322.json) | 67.21% |
| 5 | [results/20260319_224050/clip_eval_20260319_225618.json](../results/20260319_224050/clip_eval_20260319_225618.json) | 67.05% |

最差 5 个 run：

| 排名 | run | 准确率 |
|---|---|---:|
| 1 | [results/20260130_015313/clip_eval_20260130_021032.json](../results/20260130_015313/clip_eval_20260130_021032.json) | 54.54% |
| 2 | [results/20260130_015228/clip_eval_20260130_020446.json](../results/20260130_015228/clip_eval_20260130_020446.json) | 58.09% |
| 3 | [results/20260204_212705/clip_eval_20260204_214816.json](../results/20260204_212705/clip_eval_20260204_214816.json) | 62.00% |
| 4 | [results/20260204_215459/clip_eval_20260204_220758.json](../results/20260204_215459/clip_eval_20260204_220758.json) | 62.00% |
| 5 | [results/20260213_191901/clip_eval_20260213_193635.json](../results/20260213_191901/clip_eval_20260213_193635.json) | 62.72% |

## 最差 20 类

下表按 39 个 Synthetic Only run 的类别平均准确率排序，同时列出该类别在单次 run 中的最好与最差表现，以及它进入单次 run Bottom 20 的频率。

| 排名 | 类别 | 平均准确率 | 单次最低 | 单次最高 | 进入 Bottom 20 次数 |
|---|---|---:|---:|---:|---:|
| 1 | Sayornis | 0.26% | 0.00% | 3.33% | 39/39 |
| 2 | Baird_Sparrow | 0.77% | 0.00% | 10.00% | 39/39 |
| 3 | Blue_winged_Warbler | 1.79% | 0.00% | 10.00% | 39/39 |
| 4 | Parakeet_Auklet | 2.68% | 0.00% | 21.74% | 38/39 |
| 5 | Chuck_will_Widow | 3.25% | 0.00% | 11.54% | 39/39 |
| 6 | Yellow_bellied_Flycatcher | 5.92% | 0.00% | 27.59% | 37/39 |
| 7 | Rhinoceros_Auklet | 5.98% | 0.00% | 33.33% | 32/39 |
| 8 | Bronzed_Cowbird | 7.35% | 0.00% | 23.33% | 36/39 |
| 9 | Harris_Sparrow | 8.97% | 0.00% | 20.00% | 30/39 |
| 10 | Common_Raven | 9.23% | 0.00% | 46.67% | 35/39 |
| 11 | Herring_Gull | 11.54% | 0.00% | 30.00% | 27/39 |
| 12 | Whip_poor_Will | 14.84% | 0.00% | 57.89% | 27/39 |
| 13 | Tree_Sparrow | 15.73% | 0.00% | 36.67% | 27/39 |
| 14 | Clay_colored_Sparrow | 16.80% | 0.00% | 34.48% | 25/39 |
| 15 | Green_Kingfisher | 18.21% | 0.00% | 33.33% | 17/39 |
| 16 | Henslow_Sparrow | 19.74% | 0.00% | 36.67% | 18/39 |
| 17 | Western_Gull | 19.83% | 0.00% | 70.00% | 22/39 |
| 18 | Nelson_Sharp_tailed_Sparrow | 20.51% | 0.00% | 43.33% | 17/39 |
| 19 | Le_Conte_Sparrow | 20.78% | 3.45% | 68.97% | 20/39 |
| 20 | Ringed_Kingfisher | 22.48% | 3.33% | 60.00% | 21/39 |

## 最稳定的难类

- Sayornis、Baird_Sparrow、Blue_winged_Warbler、Chuck_will_Widow 在 39 个 run 中每一次都进入 Bottom 20。
- Parakeet_Auklet 在 39 个 run 中有 38 次进入 Bottom 20。
- Yellow_bellied_Flycatcher 和 Bronzed_Cowbird 分别有 37 次、36 次进入 Bottom 20。
- 这说明最差类不是由某几次异常 run 造成，而是 Synthetic Only 设置下稳定存在的系统性弱项。

## 代表性误判模式

以下误判去向基于 28 个带逐样本 CSV 的 Synthetic Only run 统计：

- Sayornis 最常被误判为 Olive_sided_Flycatcher。
- Baird_Sparrow 最常被误判为 Grasshopper_Sparrow。
- Blue_winged_Warbler 最常被误判为 Wilson_Warbler。
- Parakeet_Auklet 最常被误判为 Least_Auklet。
- Chuck_will_Widow 最常被误判为 Nighthawk。
- Yellow_bellied_Flycatcher 最常被误判为 Least_Flycatcher。
- Rhinoceros_Auklet 最常被误判为 Crested_Auklet。
- Bronzed_Cowbird 最常被误判为 Shiny_Cowbird。
- Harris_Sparrow 最常被误判为 Brewer_Sparrow。
- Common_Raven 最常被误判为 American_Crow。

这些误判高度集中在外观非常接近的细粒度近邻类别上，尤其是 Sparrow、Warbler、Auklet、Kingfisher、Crow 以及夜行鸟相关类别，说明 Synthetic Only 的主要问题不是整体识别失效，而是细粒度区分能力不足。

## 表现最好的 10 类

| 排名 | 类别 | 平均准确率 |
|---|---|---:|
| 1 | White_breasted_Nuthatch | 100.00% |
| 2 | White_Pelican | 99.87% |
| 3 | American_Goldfinch | 99.57% |
| 4 | Red_bellied_Woodpecker | 99.49% |
| 5 | Pileated_Woodpecker | 99.06% |
| 6 | Cedar_Waxwing | 98.72% |
| 7 | Green_Jay | 97.91% |
| 8 | Brown_Creeper | 97.88% |
| 9 | Rose_breasted_Grosbeak | 97.86% |
| 10 | European_Goldfinch | 97.44% |

## 结论

- 如果只看 Synthetic Only，全量结果已经清楚地暴露出一批稳定难类，而不是单次 run 的偶然波动。
- 最差类别高度集中在细粒度相似物种，说明当前合成数据对“类间细微差别”的表达不够强。
- 后续若要继续优化，最优先的方向应该是针对这些稳定难类做数据质量检查、近邻类对比增强和定向难例挖掘。