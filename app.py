import os
import io
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, render_template, send_from_directory
import albumentations as A
from albumentations.pytorch import ToTensorV2
from matplotlib import colormaps
from gradcam_utils import get_gradcam

# --- 应用初始化 ---
app = Flask(__name__)
# 设置上传文件夹和允许的文件扩展名
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER

# --- 模型加载 ---
# 确定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义类别（需要根据你的数据集进行修改）
# 根据之前的分析，这里的类别是 6 个
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
class_names_cn = ['纸板', '玻璃', '金属', '纸张', '塑料', '其他垃圾']
num_classes = len(class_names)

# 加载 MobileNetV2 模型结构
model = models.mobilenet_v2(weights=None) # 不使用预训练权重，因为我们要加载自己的权重
model.classifier[1] = nn.Linear(model.last_channel, num_classes)

# 加载训练好的模型权重
model_path = 'mobilenetv2_best.pth'
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✔ 模型权重 '{model_path}' 加载成功。")
except FileNotFoundError:
    print(f"✘ 错误: 模型文件 '{model_path}' 未找到。请确保模型文件存在。")
    exit()
except Exception as e:
    print(f"✘ 加载模型时出错: {e}")
    exit()

model.to(device)
model.eval()

# 为 Grad-CAM 启用梯度计算
for param in model.parameters():
    param.requires_grad = True

# --- 图像预处理 ---
# 定义图像变换
class ImageNetNorm:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

def get_transforms(img_size=224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(ImageNetNorm.mean, ImageNetNorm.std),
        ToTensorV2()
    ])

val_tf = get_transforms()

def apply_transform(img, transform):
    return transform(image=np.array(img))["image"]

# --- Grad-CAM & 结果可视化 ---
def apply_heatmap(original_image, heatmap, colormap_name='jet', alpha=0.5):
    """ 将热力图叠加到原始图像上 """
    colormap = colormaps.get_cmap(colormap_name)
    heatmap_colored = colormap(heatmap)[:, :, :3] # 取 RGB 通道
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    # 确保原始图像是 numpy array 且为 RGB
    original_np = np.array(original_image.convert('RGB'))
    
    # 调整热力图尺寸以匹配原始图像
    heatmap_resized = np.array(Image.fromarray(heatmap_colored).resize(original_np.shape[:2][::-1], Image.BILINEAR))

    # 叠加
    overlay = (original_np * (1 - alpha) + heatmap_resized * alpha).astype(np.uint8)
    return Image.fromarray(overlay)

# --- Flask 路由 ---
def allowed_file(filename):
    """ 检查文件扩展名是否被允许 """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_and_predict():
    if request.method == 'POST':
        # 检查是否有文件在请求中
        if 'file' not in request.files:
            return render_template('index.html', error='没有文件部分')
        
        file = request.files['file']
        
        # 如果用户没有选择文件，浏览器也会提交一个空文件名
        if file.filename == '':
            return render_template('index.html', error='未选择文件')

        if file and allowed_file(file.filename):
            # 读取图像文件
            img_bytes = file.read()
            original_image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            
            # 预处理图像
            tensor = apply_transform(original_image, val_tf).unsqueeze(0).to(device)
            
            # 模型预测
            with torch.no_grad():
                outputs = model(tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                predicted_idx = torch.argmax(probabilities).item()
                predicted_class = class_names[predicted_idx]
                confidence = probabilities[predicted_idx].item()

            # 生成 Grad-CAM
            heatmap = get_gradcam(model, tensor, class_idx=predicted_idx)
            
            # 将 Grad-CAM 应用到原图
            gradcam_image = apply_heatmap(original_image, heatmap)

            # 保存结果图片以便在前端显示
            result_filename = f"result_{file.filename}"
            # 确保 static 目录存在
            if not os.path.exists(app.config['STATIC_FOLDER']):
                os.makedirs(app.config['STATIC_FOLDER'])
            gradcam_image.save(os.path.join(app.config['STATIC_FOLDER'], result_filename))
            
            # 准备要在模板中显示的结果
            result = {
                'class_name': predicted_class.upper(),
                'class_name_cn': class_names_cn[predicted_idx],
                'confidence': f"{confidence:.2%}",
                'image_filename': result_filename
            }
            
            return render_template('index.html', result=result)

    # 对于 GET 请求，只显示主页面
    return render_template('index.html')

@app.route('/static/<path:filename>')
def send_static(filename):
    """ 提供静态文件服务 (例如, 显示结果图片) """
    return send_from_directory(app.config['STATIC_FOLDER'], filename)

if __name__ == '__main__':
    # 确保上传目录存在
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    # 确保静态目录存在
    if not os.path.exists(STATIC_FOLDER):
        os.makedirs(STATIC_FOLDER)
        
    app.run(debug=True) 