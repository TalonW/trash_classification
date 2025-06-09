# 垃圾分类项目综合报告

## 1. 项目简介与问题背景

本项目旨在开发一个能够自动识别和分类不同类型垃圾的系统。随着城市化进程的加快和生活垃圾的急剧增加，有效的垃圾分类对于环境保护、资源回收利用以及可持续发展至关重要。本项目探索并对比了两种主流的图像分类技术：一种是基于传统计算机视觉特征（HOG）和支持向量机（SVM）的方法，另一种是基于深度学习的卷积神经网络模型（MobileNetV2）。

项目的核心任务是根据垃圾图片，将其准确分类到预定义的类别中，如纸板（cardboard）、玻璃（glass）、金属（metal）、纸张（paper）、塑料（plastic）和其他垃圾（trash）。通过对比分析不同技术路线的性能，为实际应用选择合适的技术方案提供参考。

## 2. 项目文件结构

项目的核心文件和目录结构如下：

```
.
├── data/
│   └── trashnet/              # 包含各类垃圾图片的根目录，每个子目录代表一个类别
│       ├── cardboard/
│       ├── glass/
│       ├── metal/
│       ├── paper/
│       ├── plastic/
│       └── trash/
├── garbage_classification.py  # 主程序脚本，包含模型训练、评估和命令行接口
├── gradcam_utils.py           # Grad-CAM 可视化工具脚本
├── svm_hog.pkl                # 训练好的SVM模型及相关参数
├── mobilenetv2_best.pth       # 训练好的MobileNetV2模型权重
├── svm_classification_report.txt      # SVM模型在测试集上的分类报告
├── mobilenet_classification_report.txt # MobileNetV2模型在验证集上的分类报告
├── svm_confusion.png          # SVM模型的混淆矩阵图
├── mobilenet_confusion.png    # MobileNetV2模型的混淆矩阵图
├── gradcam_sample.png         # MobileNetV2的Grad-CAM可视化示例图
└── model_comparison_analysis.md # SVM与MobileNetV2的详细性能对比分析报告 (本文档将包含其内容)
└── project_summary_for_llm.md # (本文档) 综合项目信息，用于大语言模型生成报告
```

## 3. 技术路线

本项目主要探索并实现了以下两种技术路线进行垃圾分类：

### 3.1. 传统计算机视觉方法：HOG + SVM

*   **特征提取 (HOG - Histogram of Oriented Gradients)**:
    *   对每张输入的垃圾图片进行预处理（灰度化、尺寸归一化）。
    *   提取图像的HOG特征。HOG特征通过计算图像局部区域中梯度方向的直方图来描述物体的边缘和形状信息，对于物体识别具有较好的鲁棒性。
*   **分类器 (SVM - Support Vector Machine)**:
    *   使用提取到的HOG特征作为输入，训练一个支持向量机分类器。
    *   SVM通过在高维空间中寻找一个最优超平面，将不同类别的特征点分开。本项目中使用了带有径向基函数（RBF）核的SVM。
    *   在训练前，对HOG特征进行了标准化处理。

### 3.2. 深度学习方法：MobileNetV2

*   **模型架构 (MobileNetV2)**:
    *   MobileNetV2是一种轻量级的卷积神经网络（CNN），专为移动和嵌入式视觉应用设计。它通过使用深度可分离卷积和线性瓶颈（inverted residuals and linear bottlenecks）等技术，在保持较高准确率的同时显著减少了模型参数量和计算复杂度。
    *   本项目加载了在ImageNet上预训练的MobileNetV2模型权重，并对其顶部分类层进行了修改，以适应本项目中的垃圾分类类别数量。
*   **迁移学习**:
    *   采用了迁移学习的策略，冻结了预训练模型的特征提取部分（卷积层）的参数，只训练新添加的分类层。这样可以利用ImageNet大数据集上学到的通用图像特征，加快模型收敛速度，并提高在小数据集上的泛化能力。
*   **数据增强**:
    *   在训练过程中，对输入的训练图片应用了一系列数据增强技术（如随机裁剪、水平翻转、亮度对比度调整、旋转等），以增加训练数据的多样性，提高模型的鲁棒性。
*   **Grad-CAM 可视化**:
    *   为了理解MobileNetV2模型的决策依据，集成了Grad-CAM（Gradient-weighted Class Activation Mapping）技术。Grad-CAM能够生成热力图，显示出图像中对模型分类决策贡献最大的区域。

## 4. 核心代码实现

### 4.1. 主程序脚本: `garbage_classification.py`

```python
#!/usr/bin/env python
# coding: utf-8
"""
一键训练 / 评估垃圾分类模型（SVM 或 MobileNetV2）
用法示例:
  python garbage_classification.py --data_dir data/trashnet --model svm
  python garbage_classification.py --data_dir data/trashnet --model mobilenet --epochs 15
"""
import argparse, pathlib, itertools, time, json, numpy as np, matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.io import imread
from PIL import Image
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from gradcam_utils import get_gradcam

# ---------- 共用 ----------
def plot_confusion(cm, classes, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar(im, shrink=0.75)
    tick_marks = np.arange(len(classes))
    ax.set(xticks=tick_marks, yticks=tick_marks,
           xticklabels=classes, yticklabels=classes,
           ylabel="True", xlabel="Predicted")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

# ---------- SVM + HOG ----------
def train_svm(data_dir, img_size=128, cell=8):
    data_dir = pathlib.Path(data_dir)
    classes = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
    X, y = [], []
    print("▶ Extracting HOG features ...")
    for label, cls in enumerate(classes):
        for img_path in (data_dir/cls).glob("*"):
            img = imread(img_path)
            if img.ndim == 3:
                img = rgb2gray(img)
            img = np.array(Image.fromarray((img*255).astype(np.uint8)).resize((img_size, img_size))) / 255.
            feat = hog(img, pixels_per_cell=(cell, cell), cells_per_block=(2, 2), feature_vector=True)
            X.append(feat); y.append(label)
    X, y = np.array(X), np.array(y)
    
    # 划分训练集和测试集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 标准化特征
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 训练SVM
    svm = SVC(kernel="rbf", C=10, gamma='scale', probability=True)
    print("▶ Training SVM ...")
    svm.fit(X_train, y_train)
    
    # 在测试集上评估
    print("▶ Evaluating SVM on test set ...")
    preds = svm.predict(X_test)
    report_str_svm = classification_report(y_test, preds, target_names=classes, zero_division=0)
    print(report_str_svm)
    with open("svm_classification_report.txt", "w") as f:
        f.write(f"SVM Classification Report (on test set, {len(y_test)} samples):\n\n")
        f.write(report_str_svm)
    print("✔ SVM classification report saved to svm_classification_report.txt")
    
    # 绘制混淆矩阵
    cm = confusion_matrix(y_test, preds)
    plot_confusion(cm, classes, "svm_confusion.png")
    
    # 保存模型和标准化器
    torch.save({
        "svm": svm,
        "scaler": scaler,
        "classes": classes,
        "img_size": img_size,
        "cell": cell
    }, "svm_hog.pkl")
    print("✔ 模型保存在 svm_hog.pkl, 混淆矩阵 svm_confusion.png")
    
    # 返回训练好的模型和标准化器，方便后续使用
    return svm, scaler, classes

# ---------- MobileNetV2 ----------
class ImageNetNorm:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

def get_transforms(img_size=224, train=True):
    if train:
        aug = A.Compose([
            A.RandomResizedCrop((img_size, img_size), scale=(0.8, 1.0)),
            A.HorizontalFlip(),
            A.RandomBrightnessContrast(0.2, 0.2),
            A.Rotate(limit=15),
            A.Normalize(ImageNetNorm.mean, ImageNetNorm.std), ToTensorV2()
        ])
    else:
        aug = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(ImageNetNorm.mean, ImageNetNorm.std), ToTensorV2()
        ])
    return aug

def apply_transform(img, transform):
    return transform(image=np.array(img))["image"]

def train_mobilenet(data_dir, epochs=10, batch=32, lr=1e-4, img_size=224):
    train_tf = get_transforms(img_size, True)
    val_tf = get_transforms(img_size, False)
    
    class CustomImageFolder(datasets.ImageFolder):
        def __init__(self, root, transform=None):
            super().__init__(root, transform=None) # Base class handles loading, we apply aug_transform in __getitem__
            self.aug_transform = transform
            
        def __getitem__(self, index):
            img, label = super().__getitem__(index) # Gets PIL image and label
            if self.aug_transform is not None:
                # albumentations expects numpy array
                img = apply_transform(img, self.aug_transform) 
            return img, label
    
    # Initialize datasets
    full_dataset_train = CustomImageFolder(data_dir, transform=train_tf)
    full_dataset_val = CustomImageFolder(data_dir, transform=val_tf) # Use val_tf for the 'validation' part of full_dataset_val

    # Get class names once from the full dataset structure
    dataset_classes = full_dataset_train.classes # or full_dataset_val.classes, should be the same

    # 简单随机划分
    n = len(full_dataset_train) # Total number of images
    idxs = torch.randperm(n)
    split = int(0.8*n)
    train_idx, val_idx = idxs[:split], idxs[split:]
    
    ds_train = torch.utils.data.Subset(full_dataset_train, train_idx)
    ds_val = torch.utils.data.Subset(full_dataset_val, val_idx) # This subset will use val_tf due to full_dataset_val
    
    train_loader = DataLoader(ds_train, batch_size=batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(ds_val, batch_size=batch, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.mobilenet_v2(weights="DEFAULT")
    for p in model.features.parameters():
        p.requires_grad = False
    model.classifier[1] = nn.Linear(model.last_channel, len(dataset_classes)) # Use len(dataset_classes)
    model.to(device)

    opt = optim.Adam(model.parameters(), lr)
    criterion = nn.CrossEntropyLoss()

    print("▶ Training MobileNetV2 ...")
    best_acc = 0
    for epoch in range(epochs):
        model.train(); running = 0
        for x, y in tqdm(train_loader, desc=f"E{epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); out = model(x); loss = criterion(out, y)
            loss.backward(); opt.step(); running += loss.item()*y.size(0)
        train_loss = running/len(ds_train) # Corrected: len(ds_train) not len(train_loader.dataset) which is full_dataset_train

        # --- 验证 ---
        model.eval(); correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x); pred = out.argmax(1)
                correct += (pred==y).sum().item()
        acc = correct/len(ds_val) # Corrected: len(ds_val) not len(val_loader.dataset) which is full_dataset_val
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} val_acc={acc:.4f}")
        if acc > best_acc:
            torch.save(model.state_dict(), "mobilenetv2_best.pth"); best_acc = acc
    print("✔ 最佳模型保存为 mobilenetv2_best.pth")
    
    # 报告
    y_true, y_pred = [], []
    model.eval(); model.load_state_dict(torch.load("mobilenetv2_best.pth"))
    for x, y in DataLoader(ds_val, batch_size=batch): # Use the same ds_val for reporting
        x = x.to(device)
        out = model(x); pred = out.argmax(1).cpu()
        y_true.extend(y.numpy()); y_pred.extend(pred.numpy())
    
    report_str_mobilenet = classification_report(y_true, y_pred, target_names=dataset_classes, zero_division=0)
    print(report_str_mobilenet)
    with open("mobilenet_classification_report.txt", "w") as f:
        f.write(f"MobileNetV2 Classification Report (on validation set, {len(y_true)} samples):\n\n")
        f.write(report_str_mobilenet)
    print("✔ MobileNetV2 classification report saved to mobilenet_classification_report.txt")

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion(cm, dataset_classes, "mobilenet_confusion.png")
    
    # Grad‑CAM 可视化示例
    # To get an image from ds_val (which is a Subset), we need to access its underlying dataset and original index
    # However, it's simpler to just re-create a dataset instance for one image if needed for GradCAM
    # Or, get an item directly from the Subset if transformations are consistent.
    # For simplicity, let's grab one from the validation loader (it will be a transformed tensor)
    # We need to ensure it's a single image for GradCAM.
    if len(ds_val) > 0:
        img_tensor, _ = ds_val[0] # This should be a transformed tensor
        if img_tensor.ndim == 3: # Ensure it's CHW
            img_tensor_unsqueezed = img_tensor.unsqueeze(0).to(device) # Add batch dim and move to device
            cam = get_gradcam(model, img_tensor_unsqueezed)
            plt.imsave("gradcam_sample.png", cam, cmap="jet")
            print("✔ Grad‑CAM 示例 gradcam_sample.png 已生成")
        else:
            print("⚠ Could not generate Grad-CAM sample: image tensor dimension mismatch.")
    else:
        print("⚠ Could not generate Grad-CAM sample: validation set is empty.")


# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="垃圾图像根目录")
    parser.add_argument("--model", choices=["svm", "mobilenet"], default="mobilenet")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    if args.model == "svm":
        train_svm(args.data_dir)
    else:
        train_mobilenet(args.data_dir, epochs=args.epochs)
```

### 4.2. Grad-CAM 工具脚本: `gradcam_utils.py`

```python
"""
简单版 Grad‑CAM（用于 CNN / MobileNetV2）
调用: heatmap = get_gradcam(model, input_tensor, class_idx)
"""
import torch, torch.nn.functional as F

def _find_last_conv(model):
    # 对于 MobileNetV2，我们需要找到最后一个卷积层
    last_conv_layer = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and name.startswith('features'):
            last_conv_layer = module
    if last_conv_layer is None:
        # Fallback for other generic CNNs: find literally the last Conv2d
        for child in reversed(list(model.modules())):
            if isinstance(child, torch.nn.Conv2d):
                return child
        raise ValueError("No Conv2d layer found for Grad-CAM.")
    return last_conv_layer

def get_gradcam(model, x, class_idx=None):
    model.eval()
    conv = _find_last_conv(model)
    feats, grads = [], []

    def forward_hook(_, __, output): 
        feats.append(output.detach())
    def backward_hook(_, grad_in, grad_out): 
        # grad_out is a tuple, we need the first element
        grads.append(grad_out[0].detach())

    fh = conv.register_forward_hook(forward_hook)
    bh = conv.register_backward_hook(backward_hook)

    out = model(x)
    if class_idx is None:
        class_idx = out.argmax(dim=1)
    
    one_hot = torch.zeros_like(out)
    # Ensure class_idx is a scalar or a 1-element tensor for indexing
    if class_idx.ndim > 0: # If it's a tensor
        idx_to_use = class_idx[0].item() if class_idx.numel() > 0 else 0
    else: # If it's a scalar Python number
        idx_to_use = class_idx
        
    one_hot[0, idx_to_use] = 1 # Assuming batch size is 1 for Grad-CAM input 'x'
    
    model.zero_grad() # Zero out any existing gradients before backward pass
    out.backward(gradient=one_hot, retain_graph=False) # retain_graph can be False

    fh.remove(); bh.remove()
    
    if len(grads) == 0 or len(feats) == 0:
        print("Warning: Grads or Feats list is empty. Cannot generate CAM.")
        return torch.zeros(x.shape[2:]).cpu().numpy() # Return a blank map
        
    weights = grads[0].mean((2, 3), keepdim=True) # Global Average Pooling on gradients
    cam = (weights * feats[0]).sum(1, keepdim=True) # Weighted sum of feature maps
    cam = F.relu(cam) # Apply ReLU
    
    # Resize CAM to original image size
    cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
    
    # Normalize CAM
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8) # Add epsilon to avoid division by zero
    
    return cam[0, 0].cpu().numpy()
```

## 5. 模型性能对比分析

(此处嵌入 `model_comparison_analysis.md` 的全部内容)

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