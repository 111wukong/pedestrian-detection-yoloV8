from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, Response
from ultralytics import YOLO
import cv2
import numpy as np
import logging
import io
import tempfile

# 配置日志
logging.basicConfig(level=logging.INFO)

# 创建FastAPI应用
app = FastAPI()

# 加载训练好的YOLOv8模型
model = YOLO('runs/best.pt')


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    logging.info("Received file: %s", file.filename)
    try:
        # 读取上传的图像文件
        contents = await file.read()
        logging.info("File read successfully, size: %d bytes", len(contents))

        np_image = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Image decoding failed.")

        logging.info("Image decoded successfully, shape: %s", image.shape)

        # 使用模型进行预测
        results = model(image)

        # 获取预测结果并绘制
        result_img = results[0].plot()

        # 将结果图像编码为JPEG格式并保存到临时文件
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        cv2.imwrite(temp_file.name, result_img)
        logging.info("Image encoded and written to temporary file: %s", temp_file.name)

        # 返回临时文件作为响应
        return FileResponse(temp_file.name, media_type="image/jpeg", filename="result.jpg")
    except Exception as e:
        logging.error("Error during prediction: %s", str(e))
        return Response(content=str(e), media_type="text/plain", status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
