# 垃圾分类系统

基于计算机视觉的垃圾分类系统，支持 SVM+HOG 和 MobileNetV2 两种模型方案。

## 功能特点

- 支持两种模型方案：
  - SVM + HOG 特征：轻量级方案，适合资源受限场景
  - MobileNetV2：深度学习方案，准确率更高
- 提供完整的数据可视化：
  - 数据集类别分布
  - 各类别样例展示
  - 数据处理效果对比
  - Grad-CAM 可视化
- 支持多种数据增强方法：
  - 随机裁剪
  - 水平翻转
  - 亮度/对比度调整
  - 旋转
- 提供 Web 界面进行实时预测

## 环境要求

- Python 3.8+
- CUDA（可选，用于 GPU 加速）

## 安装

1. 克隆仓库：
```bash
git clone https://github.com/你的用户名/garbage-classification.git
cd garbage-classification
```

2. 创建虚拟环境（推荐）：
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 数据准备

将垃圾分类数据集放在 `data/trashnet` 目录下，目录结构如下：
```
data/trashnet/
    ├── cardboard/
    ├── glass/
    ├── metal/
    ├── paper/
    ├── plastic/
    └── trash/
```

### 2. 模型训练

#### SVM + HOG 模型
```bash
python garbage_classification.py --data_dir data/trashnet --model svm
```

#### MobileNetV2 模型
```bash
python garbage_classification.py --data_dir data/trashnet --model mobilenet --epochs 15
```

### 3. 数据可视化

生成数据集可视化和处理效果对比图：
```bash
python dataset_visiulazation.py
```
生成的图片将保存在 `dataset_pics` 目录下。

### 4. Web 界面

启动 Web 服务：
```bash
python app.py
```
然后在浏览器中访问 `http://localhost:5000` 即可使用 Web 界面进行实时预测。

## 项目结构

```
garbage-classification/
├── app.py                    # Web 应用入口
├── garbage_classification.py # 模型训练脚本
├── dataset_visiulazation.py  # 数据可视化脚本
├── gradcam_utils.py         # Grad-CAM 工具
├── requirements.txt         # 项目依赖
├── data/                    # 数据集目录
│   └── trashnet/
├── dataset_pics/           # 可视化结果
├── static/                 # Web 静态文件
└── templates/              # Web 模板
```

## 模型性能

### SVM + HOG
- 优点：模型轻量，训练快速
- 缺点：特征表达能力有限

### MobileNetV2
- 优点：准确率高，特征提取能力强
- 缺点：需要更多计算资源

## 可视化效果

项目提供了丰富的数据可视化功能：
- 数据集类别分布图
- 各类别样例展示
- 数据处理效果对比
- Grad-CAM 可视化

所有可视化结果保存在 `dataset_pics` 目录下。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License
