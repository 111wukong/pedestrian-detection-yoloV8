# import cv2
# from ultralytics import YOLO
#
# def run_video_inference(video_path, model_path='/best.pt', output_path='output_video.avi'):
#     # 加载训练好的模型
#     model = YOLO(model_path)
#
#     # 打开视频文件
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print("Error: Could not open video.")
#         return
#
#     # 获取视频的宽度、高度和帧率
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#
#     # 定义视频写入对象
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         # 使用模型进行检测
#         results = model(frame)
#
#         # 绘制检测结果
#         result_img = results[0].plot()
#
#         # 写入结果到输出视频
#         out.write(result_img)
#
#         # 显示结果（可选）
#         cv2.imshow('Inference Result', result_img)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     # 释放资源
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()
#     print(f"Detection results saved to {output_path}")
#
# if __name__ == '__main__':
#     # 替换为你想要检测的视频路径
#     video_path = './test-video/test.mp4'
#     run_video_inference(video_path)
# #图片的行人检测
# import cv2
# from ultralytics import YOLO
#
# # 加载YOLOv8模型
# model = YOLO('yolov8n.pt')  # 你可以选择其他模型，例如yolov8s.pt, yolov8m.pt等
#
# # 读取图像
# image_path = 'test-img/js.jpg'  # 替换为你的图像路径
# image = cv2.imread(image_path)
#
# # 使用模型进行检测
# results = model(image)
# # 筛选出标签为"person"的检测结果（COCO数据集中，类别0通常为'person'）
# person_results = [result for result in results[0].boxes if result.cls[0] == 0]
#
# # 绘制检测到的"person"的边界框
# for box in person_results:
#     x1, y1, x2, y2 = map(int, box.xyxy[0])
#     confidence = box.conf[0]
#     label = f"person {confidence:.2f}"
#     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 5)
#     cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)  # 调整字体大小为1
#
# # 保存结果图像
# output_path_person_only = 'person_only_detected_image1.jpg'
# cv2.imwrite(output_path_person_only, image)
#
# print(f"检测结果已保存到 {output_path_person_only}")




# import cv2
# import matplotlib.pyplot as plt
# from ultralytics import YOLO
#
# def run_video_inference(video_path, model_path='yolov8n.pt', output_path='output_video.avi'):
#     # 加载YOLOv8模型
#     model = YOLO(model_path)
#
#     # 打开视频文件
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print("Error: Could not open video.")
#         return
#
#     # 获取视频的宽度、高度和帧率
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#
#     # 定义视频写入对象
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
#
#     # 加载模型
#     model = YOLO(model_path)
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         # 使用模型进行检测
#         results = model(frame)
#
#         # 筛选出标签为"person"的检测结果（COCO数据集中，类别0通常为'person'）
#         person_results = [result for result in results[0].boxes if result.cls[0] == 0]
#
#         # 绘制检测到的"person"的边界框
#         for box in person_results:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             confidence = box.conf[0]
#             label = f"person {confidence:.2f}"
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
#             cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 4)
#
#         # 使用Matplotlib显示结果图像
#         plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#         plt.axis('off')  # 关闭坐标轴
#         plt.show()
#
#         # 写入结果到输出视频
#         out.write(frame)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     # 释放资源
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()
#     print(f"Detection results saved to {output_path}")
#
# if __name__ == '__main__':
#     # 替换为你想要检测的视频路径
#     video_path = 'test-img/xr.mp4'
#     run_video_inference(video_path)




import cv2
from ultralytics import YOLO

# 加载YOLOv8模型
model = YOLO('yolov8n.pt')  # 你可以选择其他模型，例如yolov8s.pt, yolov8m.pt等

# 读取图像
image_path = 'test-img/fire1.png'  # 替换为你的图像路径
image = cv2.imread(image_path)

# 使用模型进行检测
results = model(image)

# 筛选出标签为"fire"和"smoke"的检测结果
fire_smoke_results = [result for result in results[0].boxes if result.cls[0] in [0, 1]]  # 假设火焰的类别索引为0，烟雾的类别索引为1

# 绘制检测到的"fire"和"smoke"的边界框
for box in fire_smoke_results:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    confidence = box.conf[0]
    label = f"{model.names[int(box.cls[0])]} {confidence:.2f}"  # 获取类别名称
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)  # 绘制边界框为绿色
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # 调整字体大小为0.5，颜色为绿色

# 保存结果图像
output_path_fire_smoke_detected = 'fire_smoke_detected_image.jpg'
cv2.imwrite(output_path_fire_smoke_detected, image)

print(f"火焰和烟雾检测结果已保存到 {output_path_fire_smoke_detected}")
