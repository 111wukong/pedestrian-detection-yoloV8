from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

def run_inference(image_path, model_path='./last.pt'):
    # 加载训练好的模型
    model = YOLO(model_path)

    # 进行推理
    results = model(image_path)

    # 显示结果
    if isinstance(results, list):
        for result in results:
            result.show()
    else:
        results.show()

    # 选择第一张图像的结果
    result_img = results[0].plot()

    # 使用 OpenCV 显示结果图像
    cv2.imshow('Inference Result', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存结果图像
    save_path = 'inference_result.jpg'
    cv2.imwrite(save_path, result_img)
    print(f"Result saved to {save_path}")

if __name__ == '__main__':
    # 替换为你想要检测的图像路径
    image_path = './test-img/ms.jpg'
    run_inference(image_path)
