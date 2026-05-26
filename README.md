# 🚶 YOLOv8 Pedestrian Detection

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/111wukong/pedestrian-detection-yoloV8/actions/workflows/ci.yml/badge.svg)](https://github.com/111wukong/pedestrian-detection-yoloV8/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED?logo=docker)](https://hub.docker.com/)

**Real-time pedestrian detection powered by Ultralytics YOLOv8.**
Upload an image. Get instant results. Zero configuration.

[Quick Start](#-quick-start) •
[Demo](#-gradio-demo) •
[Architecture](#-project-architecture) •
[API](#-api-examples) •
[Model](#-model-information)

</div>

---

## 📸 Gradio Demo

Launch the web demo with a single command:

```bash
python app.py
```

Open **http://localhost:7860** and you'll see:

<p align="center">
  <img src="docs/demo-screenshot.png" alt="Gradio Demo Screenshot" width="700">
  <br>
  <em>Upload an image → Adjust confidence → See results — all in the browser.</em>
</p>

The demo includes:

- 📤 Drag-and-drop image upload
- 🎯 Adjustable confidence threshold slider
- 📸 Side-by-side original vs. annotated comparison
- 📊 Per-class detection summary with confidence scores
- 🖼️ Built-in example images to try instantly

---

## 🚀 Quick Start

### Option 1: Pip Install (Recommended)

```bash
# Clone the repo
git clone https://github.com/111wukong/pedestrian-detection-yoloV8.git
cd pedestrian-detection-yoloV8

# (Optional) Create a virtual environment
python -m venv venv && source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch the Gradio demo
python app.py
```

The model (`yolov8n.pt`) auto-downloads on first run. No manual download needed.

### Option 2: Docker

```bash
# Build the image
docker build -t pedestrian-yolov8 .

# Run the container
docker run -p 7860:7860 pedestrian-yolov8
```

Then visit **http://localhost:7860**.

### Option 3: Use a Custom Trained Model

```bash
export MODEL_PATH=path/to/your/best.pt
python app.py
```

---

## 📁 Project Architecture

```
pedestrian-detection-yoloV8/
├── app.py                    # Gradio Web Demo (main entry)
├── yolov8_service.py         # FastAPI inference service
├── inference.py              # Single-image inference script
├── inference-video.py        # Video inference script
├── train.py                  # Model training script
├── data.yaml                 # Dataset configuration
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker build recipe
├── .dockerignore             # Docker build exclusions
├── .gitignore                # Git tracking exclusions
├── .github/
│   └── workflows/
│       └── ci.yml            # GitHub Actions CI pipeline
├── web/                      # Static frontend (legacy)
│   └── index.html
├── test-img/                 # Sample test images
└── runs/                     # Training output (gitignored)
```

**Core flow:**

```
┌──────────────┐     ┌───────────────┐     ┌──────────────┐
│  Gradio UI   │────▶│  YOLOv8 Model │────▶│  Annotated   │
│  (app.py)    │     │  (ultralytics) │     │  Image +     │
│              │     │               │     │  Summary     │
└──────────────┘     └───────────────┘     └──────────────┘
```

---

## 📖 API Examples

### Gradio App (Recommended)

```python
from app import detect
import cv2

image = cv2.imread("test-img/ms.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

annotated, summary = detect(image_rgb, confidence_threshold=0.25)
print(summary)
```

### FastAPI Service

```python
# Start the server:
# uvicorn yolov8_service:app --host 0.0.0.0 --port 8000

import requests

with open("image.jpg", "rb") as f:
    resp = requests.post("http://localhost:8000/predict/", files={"file": f})

with open("result.jpg", "wb") as out:
    out.write(resp.content)
```

### Command-Line Inference

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model("test-img/ms.jpg")

# Show and save
results[0].show()
results[0].save("output.jpg")
```

---

## 🧠 Model Information

| Property | Value |
|----------|-------|
| **Architecture** | YOLOv8 Nano |
| **Base Model** | `yolov8n.pt` (COCO pre-trained) |
| **Parameters** | 3.2M |
| **Input Size** | 640x640 |
| **Classes** | 80 COCO classes (including `person`) |
| **Framework** | PyTorch + Ultralytics |

The default model (`yolov8n.pt`) is pre-trained on the **COCO dataset** and detects 80 classes out of the box. Pedestrian detection uses class `0` (`person`).

### Training Your Own Model

If you want a pedestrian-only model optimized for your use case:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
)
```

Then use it with the demo:

```bash
export MODEL_PATH=runs/detect/train/weights/best.pt
python app.py
```

---

## 🏗️ Detection Approaches

This project demonstrates **two strategies** for pedestrian detection:

### 1. Transfer Learning

Train a custom YOLOv8 model on your pedestrian dataset.

| Pros | Cons |
|------|------|
| Task-specific optimization | Requires GPU + training time |
| Better accuracy on your data | Needs labeled dataset |
| Full control over hyperparameters | Risk of overfitting |

### 2. Zero-Shot (Label Filtering)

Use the pre-trained COCO model and filter for `person` class.

| Pros | Cons |
|------|------|
| Zero training required | Limited to COCO classes |
| Minimal compute | Cannot customize object classes |
| Instant deployment | No domain-specific improvements |

> Our Gradio demo uses zero-shot by default — it just works. For production, fine-tune with your own data.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, open an issue first to discuss.

1. Fork the repo
2. Create a branch (`git checkout -b feature/amazing`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push (`git push origin feature/amazing`)
5. Open a PR

---

## 📄 License

MIT © [111wukong](https://github.com/111wukong). See [LICENSE](LICENSE).

---

<div align="center">
Made with YOLOv8
</div>
