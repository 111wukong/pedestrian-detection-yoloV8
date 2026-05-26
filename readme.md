# Pedestrian Detection with YOLOv8 🚶‍♂️

基于 YOLOv8 的行人检测项目，支持训练、推理和 Web 服务部署。

> 📖 详细博客：[CSDN 文章](https://editor.csdn.net/md/?articleId=139529123)

---

## 📁 项目结构

```
pedestrian-detection-yoloV8/
├── inference.py              # 图像推理脚本
├── inference-video.py        # 视频推理脚本
├── train.py                  # 模型训练脚本
├── yolov8_service.py         # Web 服务 (Flask)
├── web/                      # 前端页面
│   ├── index.html
│   └── assests/              # 静态资源
├── data.yaml                 # 数据集配置
├── requirements.txt          # Python 依赖
├── LICENSE                   # MIT 许可证
└── README.md
```

> 注意：模型权重文件 (`.pt`)、图片 (`.jpg/.png`)、训练输出 (`runs/`) 和测试图片 (`test-img/`) 已加入 `.gitignore`，不会被提交到仓库。

---

## 🚀 快速开始

### 环境要求

- Python 3.8+
- CUDA 11.8+ (可选，GPU 加速)
- 建议使用虚拟环境

### 安装

```bash
# 克隆项目
git clone https://github.com/111wukong/pedestrian-detection-yoloV8.git
cd pedestrian-detection-yoloV8

# 安装依赖
pip install -r requirements.txt
```

### 运行推理

```bash
# 图像推理
python inference.py

# 视频推理
python inference-video.py

# Web 服务
python yolov8_service.py
```

### 下载模型

YOLOv8 预训练模型会在首次运行时自动下载，或手动下载：

```bash
# 下载 YOLOv8n 模型
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

---

## 📖 使用指南

### 1. 准备数据集

假设你的数据集已经转换为 YOLO 格式（即每个图像都有对应的 YOLO 格式的注释文件）。数据集应该有以下结构：

```
dataset/
  ├── images/
  │   ├── train/
  │   │   ├── img1.jpg
  │   │   ├── img2.jpg
  │   │   └── ...
  │   ├── val/
  │   │   ├── img1.jpg
  │   │   └── ...
  └── labels/
      ├── train/
      │   ├── img1.txt
      │   ├── img2.txt
      │   └── ...
      ├── val/
      │   ├── img1.txt
      │   └── ...
```

### 2. 配置 YOLOv8 模型

创建一个 YAML 文件来配置你的数据集，例如 `data.yaml`：

```yaml
train: dataset/images/train
val: dataset/images/val

nc: 1  # 类别数量，这里是1类：行人
names: ['person']
```

### 3. 训练模型

使用以下 Python 脚本或命令行指令来训练 YOLOv8 模型：

```python
from ultralytics import YOLO

# 加载YOLOv8模型
model = YOLO('yolov8n.pt')  # 选择合适的模型大小：yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt

# 开始训练
model.train(data='data.yaml', epochs=100, imgsz=640, batch=16)
```

或者使用命令行：

```bash
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=100 imgsz=640 batch=16
```

### 4. 验证和测试

在训练期间，模型会自动使用验证数据集进行评估。你可以在训练完成后进行进一步的评估：

```python
# 评估模型
metrics = model.val()
print(metrics)
```

### 5. 推理

使用训练好的模型进行推理：

```python
# 使用训练好的模型进行推理
results = model('path/to/image.jpg')  # 可以是单张图片路径或目录
results.show()  # 显示检测结果
results.save('path/to/save')  # 保存检测结果
```

或者使用命令行：

```bash
yolo task=detect mode=predict model=path/to/best.pt source=path/to/image.jpg
```

### 完整示例

```python
from ultralytics import YOLO

# 加载YOLOv8模型
model = YOLO('yolov8n.pt')

# 开始训练
model.train(data='data.yaml', epochs=100, imgsz=640, batch=16)

# 评估模型
metrics = model.val()
print(metrics)

# 使用训练好的模型进行推理
results = model('path/to/image.jpg')
results.show()  # 显示检测结果
results.save('path/to/save')  # 保存检测结果
```

### 注意事项

1. **超参数调整**：训练过程中可以根据需要调整超参数（如学习率、批量大小等）。
2. **数据增强**：使用数据增强技术可以提高模型的泛化能力。
3. **模型选择**：根据你的计算资源选择合适的模型大小（如 nano, small, medium, large, xlarge）。
4. **结果分析**：分析模型的评估指标（如 mAP, Precision, Recall）以调整模型和数据。

---

## 🔬 零样本实现行人检测（标签过滤方法）

在这种方法中，不对模型进行重新训练，而是在模型输出的基础上，通过筛选、过滤标签来达到特定的识别目标。

### 方法原理

1. **模型输出**：首先使用一个预训练好的目标检测模型来对图像进行检测。这些模型已经在大型数据集上进行了训练，学习到了各种不同类别的目标的特征。
2. **目标标签过滤**：从模型的输出结果中提取目标的标签信息。这些标签通常包含了检测到的目标类别（如人、车、狗等）、位置（边界框坐标）、置信度分数等信息。
3. **选择感兴趣的类别**：在标签过滤的过程中，根据任务需求选择感兴趣的目标类别。例如，只对行人感兴趣，可以只保留标签为"行人"的目标检测结果，而过滤掉其他类别的目标。
4. **阈值处理**：除了选择感兴趣的类别外，还可以根据置信度分数来进行阈值处理。可以设定阈值来过滤掉低置信度的检测结果，以确保只保留可信度较高的目标。
5. **结果可视化或保存**：最后，将经过标签过滤处理后的目标检测结果进行可视化或保存。

### Demo

```python
import cv2
from ultralytics import YOLO

# 加载YOLOv8模型
model = YOLO('yolov8n.pt')  # 你可以选择其他模型，例如yolov8s.pt, yolov8m.pt等
image_path = 'test-img/ms.jpg'  # 替换为你的图像路径
image = cv2.imread(image_path)

# 使用模型进行检测
results = model(image)

# 筛选出标签为"person"的检测结果（COCO数据集中，类别0通常为'person'）
person_results = [result for result in results[0].boxes if result.cls[0] == 0]

# 绘制检测到的"person"的边界框
for box in person_results:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    confidence = box.conf[0]
    label = f"person {confidence:.2f}"
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 4)

# 保存结果图像
output_path_person_only = 'person_only_detected_image1.jpg'
cv2.imwrite(output_path_person_only, image)
print(f"检测结果已保存到 {output_path_person_only}")
```

---

## ⚖️ 两种方法对比

### 迁移学习

| 优点 | 缺点 |
|------|------|
| 目标定制化：针对特定任务和数据集优化 | 时间和资源消耗大 |
| 灵活性：可调整架构、超参数和训练策略 | 需要大量标注数据 |
| 更适应新任务：提高泛化能力和适应性 | 可能过拟合小数据集 |

### 标签过滤

| 优点 | 缺点 |
|------|------|
| 简单快速，无需重新训练 | 受限于原模型预训练特征 |
| 资源消耗低 | 无法完全定制化 |
| 保留原模型在大型数据集上的丰富特征 | 复杂场景可能导致误差传播 |

---

## 📄 许可证

本项目基于 [MIT License](LICENSE) 开源。

---

## 💡 总结

没有最好的方法，只有最合适的方法。
