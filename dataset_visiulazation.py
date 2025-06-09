import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
from pathlib import Path
import matplotlib
import cv2
import torch
from skimage.feature import hog
from skimage import exposure
from skimage.color import rgb2gray
from torchvision import models, transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from gradcam_utils import get_gradcam

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 类别名称中英文映射
CLASS_NAMES = {
    'cardboard': '纸板',
    'glass': '玻璃',
    'metal': '金属',
    'paper': '纸张',
    'plastic': '塑料',
    'trash': '其他垃圾'
}

def create_class_distribution_plot(data_dir, save_path):
    """创建类别分布柱状图"""
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    class_counts = []
    chinese_names = [CLASS_NAMES.get(cls, cls) for cls in classes]
    
    for class_name in classes:
        class_path = os.path.join(data_dir, class_name)
        count = len([f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
        class_counts.append(count)
    
    plt.figure(figsize=(12, 6))
    plt.bar(chinese_names, class_counts)
    plt.title('数据集类别分布')
    plt.xlabel('类别')
    plt.ylabel('图片数量')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def create_sample_images_grid(data_dir, save_path):
    """创建每个类别的样例图（横向排列）"""
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    chinese_names = [CLASS_NAMES.get(cls, cls) for cls in classes]
    fig, axes = plt.subplots(1, len(classes), figsize=(3*len(classes), 3))
    
    for i, (class_name, chinese_name) in enumerate(zip(classes, chinese_names)):
        class_path = os.path.join(data_dir, class_name)
        images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        sample_image = random.sample(images, 1)[0]
        
        img_path = os.path.join(class_path, sample_image)
        img = Image.open(img_path)
        axes[i].imshow(img)
        axes[i].set_title(chinese_name)
        axes[i].axis('off')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def create_processed_comparison(data_dir, save_path, samples_per_class=2):
    """创建原图和处理后图片的对比图"""
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    chinese_names = [CLASS_NAMES.get(cls, cls) for cls in classes]
    fig, axes = plt.subplots(len(classes), samples_per_class*2, figsize=(15, 3*len(classes)))
    
    for i, (class_name, chinese_name) in enumerate(zip(classes, chinese_names)):
        class_path = os.path.join(data_dir, class_name)
        images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        sample_images = random.sample(images, min(samples_per_class, len(images)))
        
        for j, img_name in enumerate(sample_images):
            img_path = os.path.join(class_path, img_name)
            img = Image.open(img_path)
            
            # 显示原图
            axes[i, j*2].imshow(img)
            axes[i, j*2].axis('off')
            if j == 0:
                axes[i, j*2].set_ylabel(f"{chinese_name}\n原图")
            
            # 显示处理后的图片（这里使用简单的灰度转换作为示例）
            processed_img = img.convert('L')
            axes[i, j*2+1].imshow(processed_img, cmap='gray')
            axes[i, j*2+1].axis('off')
            if j == 0:
                axes[i, j*2+1].set_ylabel("处理后")
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

# SVM+HOG 路线处理对比

def compare_resize(img, save_path):
    # 原图
    w, h = img.size
    # 目标缩放，长边128，保持比例
    scale = 128 / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = img.resize((new_w, new_h))
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(img)
    axs[0].set_title('原图')
    axs[0].axis('off')
    axs[1].imshow(resized)
    axs[1].set_title('缩放后')
    axs[1].axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def compare_gray(img, save_path):
    gray = img.convert('L')
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(img)
    axs[0].set_title('原图')
    axs[0].axis('off')
    axs[1].imshow(gray, cmap='gray')
    axs[1].set_title('灰度化')
    axs[1].axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def compare_norm(img, save_path):
    arr = np.array(img).astype(np.float32) / 255.0
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(img)
    axs[0].set_title('原图')
    axs[0].axis('off')
    axs[1].imshow(arr)
    axs[1].set_title('归一化')
    axs[1].axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def compare_hog(img, save_path):
    gray = np.array(img.convert('L')) / 255.0
    fd, hog_image = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(img)
    axs[0].set_title('原图')
    axs[0].axis('off')
    axs[1].imshow(hog_image, cmap='gray')
    axs[1].set_title('HOG 可视化')
    axs[1].axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# MobileNetV2 路线处理对比

def compare_crop(img, save_path):
    aug = A.RandomResizedCrop((224, 224), scale=(0.8, 1.0), p=1.0)
    cropped = aug(image=np.array(img))['image']
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(img)
    axs[0].set_title('原图')
    axs[0].axis('off')
    axs[1].imshow(cropped)
    axs[1].set_title('随机裁剪')
    axs[1].axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def compare_flip(img, save_path):
    aug = A.HorizontalFlip(p=1.0)
    flipped = aug(image=np.array(img))['image']
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(img)
    axs[0].set_title('原图')
    axs[0].axis('off')
    axs[1].imshow(flipped)
    axs[1].set_title('水平翻转')
    axs[1].axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def compare_brightness(img, save_path):
    aug = A.RandomBrightnessContrast(0.2, 0.2, p=1.0)
    bright = aug(image=np.array(img))['image']
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(img)
    axs[0].set_title('原图')
    axs[0].axis('off')
    axs[1].imshow(bright)
    axs[1].set_title('亮度/对比度增强')
    axs[1].axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def compare_rotate(img, save_path):
    aug = A.Rotate(limit=15, p=1.0)
    rotated = aug(image=np.array(img))['image']
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(img)
    axs[0].set_title('原图')
    axs[0].axis('off')
    axs[1].imshow(rotated)
    axs[1].set_title('旋转')
    axs[1].axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def compare_norm_mobilenet(img, save_path):
    arr = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normed = (arr - mean) / std
    normed = np.clip((normed - normed.min()) / (normed.max() - normed.min()), 0, 1)  # 归一化到0-1方便显示
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(img)
    axs[0].set_title('原图')
    axs[0].axis('off')
    axs[1].imshow(normed)
    axs[1].set_title('归一化')
    axs[1].axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def compare_augmented(img, save_path):
    aug = A.Compose([
        A.RandomResizedCrop((224, 224), scale=(0.8, 1.0), p=1.0),
        A.HorizontalFlip(p=1.0),
        A.RandomBrightnessContrast(0.2, 0.2, p=1.0),
        A.Rotate(limit=15, p=1.0)
    ])
    aug_img = aug(image=np.array(img))['image']
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(img)
    axs[0].set_title('原图')
    axs[0].axis('off')
    axs[1].imshow(aug_img)
    axs[1].set_title('增强后彩色图')
    axs[1].axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def compare_gradcam(img, save_path):
    # 载入mobilenetv2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.mobilenet_v2(weights="DEFAULT").to(device)
    model.eval()
    # 预处理
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(input_tensor)
    cam = get_gradcam(model, input_tensor)
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(img)
    axs[0].set_title('原图')
    axs[0].axis('off')
    axs[1].imshow(img.resize((224, 224)))
    axs[1].imshow(cam, cmap='jet', alpha=0.5)
    axs[1].set_title('Grad-CAM')
    axs[1].axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    data_dir = "data/trashnet"
    save_dir = "dataset_pics"
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成类别分布图
    create_class_distribution_plot(
        data_dir,
        os.path.join(save_dir, "class_distribution.png")
    )
    
    # 生成样例图网格
    create_sample_images_grid(
        data_dir,
        os.path.join(save_dir, "sample_images.png")
    )
    
    # 生成处理对比图
    create_processed_comparison(
        data_dir,
        os.path.join(save_dir, "processed_comparison.png")
    )

    # 选一张样例图片
    sample_img_path = os.path.join(data_dir, 'paper', 'paper99.jpg')
    img = Image.open(sample_img_path).convert('RGB')
    # SVM+HOG 路线
    compare_resize(img, os.path.join(save_dir, 'compare_resize.png'))
    compare_gray(img, os.path.join(save_dir, 'compare_gray.png'))
    compare_norm(img, os.path.join(save_dir, 'compare_norm.png'))
    compare_hog(img, os.path.join(save_dir, 'compare_hog.png'))
    # MobileNetV2 路线
    compare_crop(img, os.path.join(save_dir, 'compare_crop.png'))
    compare_flip(img, os.path.join(save_dir, 'compare_flip.png'))
    compare_brightness(img, os.path.join(save_dir, 'compare_brightness.png'))
    compare_rotate(img, os.path.join(save_dir, 'compare_rotate.png'))
    compare_norm_mobilenet(img, os.path.join(save_dir, 'compare_norm_mobilenet.png'))
    compare_augmented(img, os.path.join(save_dir, 'compare_augmented.png'))
    compare_gradcam(img, os.path.join(save_dir, 'compare_gradcam.png'))

if __name__ == "__main__":
    main()
