# 垃圾分类模型性能对比分析

本文档对比了两种不同的垃圾分类模型在相同数据集上的性能：一种是基于HOG特征的传统机器学习模型SVM（支持向量机），另一种是深度学习模型MobileNetV2。两个模型都在包含2527张图片的数据集上进行训练，并使用20%的数据（506张图片）作为测试集/验证集进行评估。

## 测试结果摘要

### 1. SVM (HOG + 支持向量机)

```
SVM Classification Report (on test set, 506 samples):

              precision    recall  f1-score   support

   cardboard       0.77      0.67      0.72        81
       glass       0.53      0.65      0.58       100
       metal       0.61      0.54      0.57        82
       paper       0.72      0.83      0.77       119
     plastic       0.56      0.55      0.55        97
       trash       1.00      0.33      0.50        27

    accuracy                           0.64       506
   macro avg       0.70      0.59      0.62       506
weighted avg       0.66      0.64      0.64       506
```

### 2. MobileNetV2

```
MobileNetV2 Classification Report (on validation set, 506 samples):

              precision    recall  f1-score   support

   cardboard       0.88      0.85      0.86        67
       glass       0.64      0.80      0.71       105
       metal       0.73      0.67      0.70        87
       paper       0.77      0.85      0.81       125
     plastic       0.65      0.66      0.66        86
       trash       0.83      0.14      0.24        36

    accuracy                           0.73       506
   macro avg       0.75      0.66      0.66       506
weighted avg       0.74      0.73      0.71       506
```

## 详细分析

### 1. 总体性能

*   **准确率 (Accuracy)**:
    *   MobileNetV2 (0.73) 显著优于 SVM (0.64)。这表明 MobileNetV2 在整体上能更准确地将垃圾图片分类到正确的类别。
*   **宏平均 (Macro Avg) F1-score**:
    *   MobileNetV2 (0.66) 也略高于 SVM (0.62)。宏平均F1分数平等对待所有类别，无论其样本量大小。
*   **加权平均 (Weighted Avg) F1-score**:
    *   MobileNetV2 (0.71) 同样优于 SVM (0.64)。加权平均F1分数考虑了每个类别的样本数量，通常更能反映模型在实际应用中的整体表现，特别是当类别不平衡时。

从总体指标来看，MobileNetV2 的性能更胜一筹。

### 2. 各类别性能分析

| 类别      | 模型        | Precision | Recall | F1-score | Support |
| :-------- | :---------- | :-------- | :----- | :------- | :------ |
| cardboard | SVM         | 0.77      | 0.67   | 0.72     | 81      |
|           | MobileNetV2 | 0.88      | 0.85   | 0.86     | 67      |
| glass     | SVM         | 0.53      | 0.65   | 0.58     | 100     |
|           | MobileNetV2 | 0.64      | 0.80   | 0.71     | 105     |
| metal     | SVM         | 0.61      | 0.54   | 0.57     | 82      |
|           | MobileNetV2 | 0.73      | 0.67   | 0.70     | 87      |
| paper     | SVM         | 0.72      | 0.83   | 0.77     | 119     |
|           | MobileNetV2 | 0.77      | 0.85   | 0.81     | 125     |
| plastic   | SVM         | 0.56      | 0.55   | 0.55     | 97      |
|           | MobileNetV2 | 0.65      | 0.66   | 0.66     | 86      |
| trash     | SVM         | 1.00      | 0.33   | 0.50     | 27      |
|           | MobileNetV2 | 0.83      | 0.14   | 0.24     | 36      |

**关键观察点:**

*   **cardboard (纸板)**: MobileNetV2 在这个类别上表现出色 (F1=0.86)，远超 SVM (F1=0.72)。
*   **glass (玻璃)**: MobileNetV2 (F1=0.71) 同样优于 SVM (F1=0.58)，尤其在召回率上有明显提升。
*   **metal (金属)**: MobileNetV2 (F1=0.70) 优于 SVM (F1=0.57)。
*   **paper (纸张)**: 两个模型在此类别上表现都相对较好，MobileNetV2 (F1=0.81) 略优于 SVM (F1=0.77)。这是两个模型F1分数最高的常见类别之一。
*   **plastic (塑料)**: MobileNetV2 (F1=0.66) 优于 SVM (F1=0.55)。这两个类别对两个模型来说似乎都有些挑战。
*   **trash (其他垃圾)**: 这是一个非常有趣的类别。
    *   SVM 的 Precision 达到了 1.00，这意味着所有被 SVM 预测为 "trash" 的样本确实是 "trash"。但是其 Recall 非常低 (0.33)，说明 SVM 漏掉了很多真实的 "trash" 样本，将它们错分到了其他类别。F1分数为0.50。
    *   MobileNetV2 在 "trash" 类别上的 Precision 也很高 (0.83)，但 Recall 更低 (0.14)，导致 F1 分数只有 0.24。
    *   两个模型在这个类别上的 Recall 都很低，表明识别 "其他垃圾" 是一个难点。MobileNetV2 似乎更加保守，更少地将样本预测为 "trash"。这可能是因为 "trash" 类别的样本量相对较少（SVM测试集27个，MobileNetV2测试集36个），模型难以充分学习其特征，或者这个类别的多样性非常大。

### 3. 模型特点与原因分析

*   **SVM (HOG)**:
    *   **优点**: 实现相对简单，计算成本较低。HOG特征对于物体的形状和纹理有一定的捕捉能力。在某些特定类别上，如果特征明显且易于区分，可能表现尚可 (如其对'trash'的极高Precision)。
    *   **缺点**: HOG是手动设计的特征，可能无法捕捉到图像中所有细微和复杂的模式。其性能高度依赖于HOG参数的选择以及SVM超参数的调优。从结果看，其泛化能力不如MobileNetV2。对于视觉上相似度高或内部差异大的类别（如 glass, plastic, metal）表现较差。
*   **MobileNetV2**:
    *   **优点**: 深度学习模型能够自动从原始像素中学习层次化的特征，通常能捕捉到更复杂和抽象的视觉模式。MobileNetV2是为移动和嵌入式设备设计的轻量级网络，在保持较高准确率的同时，计算效率也较高。它在大多数类别上的表现都优于SVM。
    *   **缺点**: 需要更多的训练数据（尽管此处数据集大小相同）和计算资源进行训练。对于样本量极少的类别，也可能难以充分学习 (如'trash'的低Recall)。

### 4. 结论与建议

*   **MobileNetV2 是本次对比中表现更优的模型**，无论是在整体准确率还是在大多数类别的 F1 分数上都超过了 SVM。这符合深度学习模型在图像分类任务上通常优于传统方法的预期。
*   **"trash" (其他垃圾) 类别对两个模型都是一个挑战**，尤其是召回率非常低。这可能源于：
    *   **样本不足**: 相对其他类别，"trash" 的样本量较少。
    *   **类别定义宽泛**: "其他垃圾" 可能包含各种各样外观差异极大的物品，导致模型难以学习到统一的判别特征。
*   **改进方向**:
    *   **数据增强**: 对于MobileNetV2，可以尝试更丰富的数据增强策略，特别是针对样本较少的类别。
    *   **处理类别不平衡**: 可以尝试过采样少数类（如 "trash"）或欠采样多数类，或者使用类别加权的损失函数。
    *   **模型微调**: 对于MobileNetV2，可以尝试更长时间的训练，或者调整学习率等超参数。
    *   **SVM调优**: 对于SVM，可以进行更细致的超参数搜索 (GridSearchCV for C and gamma)，并尝试不同的HOG参数。
    *   **特征工程 (SVM)**: 探索其他类型的特征或HOG的变种。
    *   **"trash" 类别的重新审视**: 如果可能，可以考虑是否能将 "trash" 类别进一步细分，或者收集更多该类别的代表性样本。

总的来说，MobileNetV2 提供了一个更好的性能基线。进一步的工作可以集中在提升MobileNetV2在难点类别上的表现以及解决类别不平衡问题。 