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
        f.write(f"SVM Classification Report (on test set, {len(y_test)} samples):\\n\\n")
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
            super().__init__(root, transform=None)
            self.aug_transform = transform
            
        def __getitem__(self, index):
            img, label = super().__getitem__(index)
            if self.aug_transform is not None:
                img = apply_transform(img, self.aug_transform)
            return img, label
    
    ds_train = CustomImageFolder(data_dir, transform=train_tf)
    ds_val = CustomImageFolder(data_dir, transform=val_tf)
    # 简单随机划分
    n = len(ds_train); idxs = torch.randperm(n)
    split = int(0.8*n)
    train_idx, val_idx = idxs[:split], idxs[split:]
    ds_train = torch.utils.data.Subset(ds_train, train_idx)
    ds_val = torch.utils.data.Subset(ds_val, val_idx)
    train_loader = DataLoader(ds_train, batch_size=batch, shuffle=True, num_workers=0)  # 设置 num_workers=0
    val_loader = DataLoader(ds_val, batch_size=batch, shuffle=False, num_workers=0)  # 设置 num_workers=0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.mobilenet_v2(weights="DEFAULT")
    for p in model.features.parameters():
        p.requires_grad = False
    model.classifier[1] = nn.Linear(model.last_channel, len(ds_train.dataset.classes))
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
        train_loss = running/len(train_loader.dataset)
        # --- 验证 ---
        model.eval(); correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x); pred = out.argmax(1)
                correct += (pred==y).sum().item()
        acc = correct/len(val_loader.dataset)
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} val_acc={acc:.4f}")
        if acc > best_acc:
            torch.save(model.state_dict(), "mobilenetv2_best.pth"); best_acc = acc
    print("✔ 最佳模型保存为 mobilenetv2_best.pth")
    # 报告
    y_true, y_pred = [], []
    model.eval(); model.load_state_dict(torch.load("mobilenetv2_best.pth"))
    for x, y in DataLoader(ds_val, batch_size=batch):
        x = x.to(device)
        out = model(x); pred = out.argmax(1).cpu()
        y_true.extend(y.numpy()); y_pred.extend(pred.numpy())
    
    # Get classes correctly from the subset's underlying dataset
    if isinstance(ds_val, torch.utils.data.Subset):
        classes_for_report = ds_val.dataset.classes
    else: # Should be CustomImageFolder if not a Subset (though it is a Subset here)
        classes_for_report = ds_val.classes

    report_str_mobilenet = classification_report(y_true, y_pred, target_names=classes_for_report, zero_division=0)
    print(report_str_mobilenet)
    with open("mobilenet_classification_report.txt", "w") as f:
        f.write(f"MobileNetV2 Classification Report (on validation set, {len(y_true)} samples):\\n\\n")
        f.write(report_str_mobilenet)
    print("✔ MobileNetV2 classification report saved to mobilenet_classification_report.txt")

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion(cm, classes_for_report, "mobilenet_confusion.png")
    # Grad‑CAM 可视化示例
    img, _ = ds_val[0]; cam = get_gradcam(model, img.unsqueeze(0).to(device))
    plt.imsave("gradcam_sample.png", cam, cmap="jet")
    print("✔ 混淆矩阵 mobilenet_confusion.png, Grad‑CAM 示例 gradcam_sample.png 已生成")

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
