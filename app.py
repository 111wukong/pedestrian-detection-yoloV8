"""
Pedestrian Detection with YOLOv8 — Gradio Web Demo
===================================================
A clean, interactive web UI for real-time pedestrian detection
using YOLOv8. Upload an image, adjust confidence threshold,
and instantly see annotated results.

Usage:
    python app.py                     # opens http://localhost:7860
    MODEL_PATH=best.pt python app.py  # use a custom model
"""

import os
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
# Use environment variable or default to the lightweight YOLOv8n.
# YOLOv8n auto-downloads on first run (via ultralytics).
MODEL_PATH = os.environ.get("MODEL_PATH", "yolov8n.pt")

print(f"🚀 Loading YOLOv8 model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print(f"✅ Model loaded — {len(model.names)} classes available")


# ---------------------------------------------------------------------------
# Detection function
# ---------------------------------------------------------------------------
def detect(
    image: np.ndarray | None,
    confidence_threshold: float = 0.25,
) -> tuple[np.ndarray | None, str]:
    """
    Run YOLOv8 detection on an uploaded image.

    Parameters
    ----------
    image : np.ndarray or None
        Input image in RGB format (from Gradio Image component).
    confidence_threshold : float
        Minimum confidence score for a detection to be shown.

    Returns
    -------
    tuple
        (annotated_image, detection_summary_text)
    """
    if image is None:
        return None, "⚠️  Please upload an image first."

    # Convert to BGR for OpenCV / YOLO compat (plot returns BGR, we convert back)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Run inference
    results = model(image_bgr, conf=confidence_threshold)

    # ── Build annotated image ──────────────────────────────────────────
    annotated_bgr = results[0].plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    # ── Build detection summary ─────────────────────────────────────────
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        summary = "🔍 No objects detected."
    else:
        # Count by class
        class_counts: dict[str, int] = {}
        lines: list[str] = []
        for i, box in enumerate(boxes, start=1):
            cls_name = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            lines.append(f"  {i}. **{cls_name}** — {conf:.1%}")

        summary = "### 📊 Detection Summary\n\n"
        summary += "| Class | Count |\n|-------|-------|\n"
        for name, count in sorted(class_counts.items()):
            summary += f"| {name} | {count} |\n"
        summary += "\n### 🏷️ Details\n\n" + "\n".join(lines)

    return annotated_rgb, summary


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(
    title="YOLOv8 Pedestrian Detection",
    theme=gr.themes.Soft(),
    css="""
    .detection-output { min-height: 320px; }
    .gr-textbox { font-family: 'SF Mono', 'Fira Code', monospace; }
    """,
) as demo:
    gr.Markdown(
        """
        # 🚶 YOLOv8 Pedestrian Detection

        Upload an image and instantly detect pedestrians (and 79 other COCO
        classes).  Powered by **Ultralytics YOLOv8** — no GPU required.

        Adjust the confidence slider to filter weak detections.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="📤 Upload Image",
                type="numpy",
                height=400,
            )
            confidence = gr.Slider(
                minimum=0.05,
                maximum=0.95,
                value=0.25,
                step=0.05,
                label="🎯 Confidence Threshold",
            )
            detect_btn = gr.Button("🔍 Detect", variant="primary", size="lg")

        with gr.Column(scale=1):
            output_image = gr.Image(
                label="📸 Detection Result",
                type="numpy",
                height=400,
                elem_classes=["detection-output"],
            )

    with gr.Row():
        output_text = gr.Markdown(label="📋 Details", value="*Results will appear here.*")

    # ── Example images ──────────────────────────────────────────────────
    gr.Markdown("### 🖼️ Try It Out")
    gr.Examples(
        examples=[
            ["test-img/ms.jpg", 0.25],
            ["test-img/js.jpg", 0.25],
            ["test-img/wbb.jpg", 0.25],
        ],
        inputs=[input_image, confidence],
    )

    # ── Wire up events ──────────────────────────────────────────────────
    detect_btn.click(
        fn=detect,
        inputs=[input_image, confidence],
        outputs=[output_image, output_text],
    )
    # Also trigger on image change
    input_image.change(
        fn=detect,
        inputs=[input_image, confidence],
        outputs=[output_image, output_text],
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True,
    )
