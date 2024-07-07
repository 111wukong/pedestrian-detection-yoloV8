from ultralytics import YOLO

def train_yolo():
    # 定义模型
    model = YOLO('yolov8n.pt')  # 使用预训练的 YOLOv8 模型

    # 训练模型
    model.train(
        data='./data.yaml',  # 数据集配置文件路径
        epochs=10,        # 训练轮数
        imgsz=640,         # 输入图像大小
        batch=16,          # 批次大小
        name='yolov8_person_detection',  # 训练运行名称
        cache=True         # 是否缓存数据集
    )

if __name__ == '__main__':
    train_yolo()
