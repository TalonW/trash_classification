import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import cv2 # For resizing heatmap

# 从项目中的其他文件导入
from gradcam_utils import get_gradcam
from garbage_classification import get_transforms, ImageNetNorm, apply_transform # CustomImageFolder is not strictly needed here if we list files

# --- 配置参数 ---
DATA_DIR = pathlib.Path("data/trashnet") # 你的垃圾图片数据目录
MODEL_PATH = "mobilenetv2_best.pth"     # 预训练模型路径
IMG_SIZE = 224                          # 图像大小 (应与训练时一致)
OUTPUT_DIR = pathlib.Path("gradcam_outputs") # Grad-CAM 输出目录
NUM_SAMPLES_PER_CLASS = 1               # 每个类别选择多少样本进行可视化

# 确保输出目录存在
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_model(num_classes, device):
    """加载预训练的 MobileNetV2 模型"""
    model = models.mobilenet_v2(weights=None) # 不加载预训练ImageNet权重，因为我们要加载自己的
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"模型已从 {MODEL_PATH} 加载。")
    except FileNotFoundError:
        print(f"错误: 模型文件 {MODEL_PATH} 未找到。请确保路径正确。")
        return None
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return None
        
    model.to(device)
    model.eval()
    return model

def get_image_paths(data_dir, num_samples_per_class):
    """获取每个类别的样本图片路径"""
    image_paths = []
    class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    for class_name in class_names:
        class_path = data_dir / class_name
        samples = list(class_path.glob("*.*"))[:num_samples_per_class] # 支持常见图像后缀
        image_paths.extend([(str(p), class_name) for p in samples])
    return image_paths, class_names

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    if not DATA_DIR.exists():
        print(f"错误: 数据目录 {DATA_DIR} 不存在。请检查路径。")
        return

    sample_image_paths_with_class, class_names = get_image_paths(DATA_DIR, NUM_SAMPLES_PER_CLASS)
    num_classes = len(class_names)

    if num_classes == 0:
        print(f"错误: 在 {DATA_DIR} 中没有找到类别子目录。")
        return
    print(f"找到类别: {class_names}")

    model = load_model(num_classes, device)
    if model is None:
        return

    # 获取验证集/测试集的图像转换 (不进行数据增强)
    val_transforms = get_transforms(img_size=IMG_SIZE, train=False)

    for img_path_str, class_name in sample_image_paths_with_class:
        img_path = pathlib.Path(img_path_str)
        print(f"\n处理图像: {img_path} (类别: {class_name})")

        try:
            # 1. 加载原始图像 (用于叠加)
            original_pil_img = Image.open(img_path).convert("RGB")
            original_np_img = np.array(original_pil_img)

            # 2. 预处理图像以输入模型
            #    apply_transform 需要 numpy 数组
            transformed_tensor = apply_transform(original_np_img, val_transforms)
            input_tensor = transformed_tensor.unsqueeze(0).to(device)

            # 3. 生成 Grad-CAM (这将使用 gradcam_utils.py 中的调试打印)
            #    让 get_gradcam 内部决定 class_idx (即模型预测的类别)
            heatmap_np = get_gradcam(model, input_tensor, class_idx=None) 
            
            if heatmap_np is None or heatmap_np.size == 0 or np.all(heatmap_np == 0):
                print(f"警告: 为 {img_path.name} 生成的 Grad-CAM 为空或全零。跳过保存。")
                plt.imsave(OUTPUT_DIR / f"gradcam_raw_{class_name}_{img_path.stem}_EMPTY.png", np.zeros((IMG_SIZE, IMG_SIZE)), cmap="gray")
                continue


            # 4. 保存原始 Grad-CAM 热力图 (灰度)
            raw_heatmap_path = OUTPUT_DIR / f"gradcam_raw_{class_name}_{img_path.stem}.png"
            plt.imsave(raw_heatmap_path, heatmap_np, cmap="gray") # 保存为灰度图
            print(f"原始 Grad-CAM 热力图已保存到: {raw_heatmap_path}")

            # 5. 创建并保存叠加图像
            #    将热力图调整到原始图像大小
            heatmap_resized = cv2.resize(heatmap_np, (original_pil_img.width, original_pil_img.height))
            
            #    将热力图转换为彩色 (0-255, uint8)
            heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3] * 255  # 取RGB通道，忽略alpha
            heatmap_colored = heatmap_colored.astype(np.uint8)

            #    叠加: alpha * original + beta * heatmap + gamma
            alpha = 0.6
            beta = 0.4
            superimposed_img = cv2.addWeighted(heatmap_colored, alpha, original_np_img, beta, 0)
            superimposed_img_pil = Image.fromarray(superimposed_img)

            overlay_path = OUTPUT_DIR / f"gradcam_overlay_{class_name}_{img_path.stem}.png"
            superimposed_img_pil.save(overlay_path)
            print(f"Grad-CAM 叠加图像已保存到: {overlay_path}")

        except FileNotFoundError:
            print(f"错误: 图像文件 {img_path} 未找到。")
        except Exception as e:
            print(f"处理图像 {img_path} 时发生错误: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main() 