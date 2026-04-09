"""
Script de fine-tuning do YOLOv8 para instrumentos cirúrgicos ginecológicos.

Uso:
    python models/yolov8/train/train.py
"""

from ultralytics import YOLO
from pathlib import Path

CONFIG = Path(__file__).parent.parent / "configs" / "data.yaml"
OUTPUT = Path(__file__).parent.parent / "weights"


def train(
    base_model: str = "yolov8n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
):
    model = YOLO(base_model)
    results = model.train(
        data=str(CONFIG),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(OUTPUT),
        name="surgical_tools_v1",
        patience=10,
        save=True,
        device=0,           # 0 = GPU, "cpu" = CPU
    )
    print(f"Treinamento concluído. Melhor modelo em: {results.save_dir}")


if __name__ == "__main__":
    train()
