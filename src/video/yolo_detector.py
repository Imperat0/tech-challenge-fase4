"""
Detecção de objetos e anomalias cirúrgicas usando YOLOv8.
Requisito obrigatório: modelo customizado para instrumentos ginecológicos.

Objetivos cobertos:
- Cirurgias: detecção de complicações ou sangramento anômalo
- Instrumentos cirúrgicos ginecológicos
"""

from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path


DEFAULT_WEIGHTS = Path(__file__).parent.parent.parent / "models" / "yolov8" / "weights" / "best.pt"
FALLBACK_WEIGHTS = "yolov8n.pt"  # Modelo base enquanto o customizado não está treinado


def load_model(weights_path: str | Path | None = None) -> YOLO:
    """Carrega o modelo YOLOv8. Usa o customizado se disponível."""
    path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS
    if path.exists():
        return YOLO(str(path))
    return YOLO(FALLBACK_WEIGHTS)


def detect_in_frame(model: YOLO, frame: np.ndarray, conf: float = 0.4) -> list[dict]:
    """Executa detecção em um frame e retorna lista de detecções."""
    results = model(frame, conf=conf, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class_id": int(box.cls),
                "class_name": model.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xyxy[0].tolist(),
            })
    return detections


def process_video_yolo(
    video_path: str,
    weights_path: str | None = None,
    sample_rate: int = 15,
    conf: float = 0.4,
) -> list[dict]:
    """
    Processa vídeo com YOLOv8 e retorna detecções por frame.

    Args:
        video_path: Caminho para o vídeo.
        weights_path: Pesos do modelo. Usa default se None.
        sample_rate: Analisar 1 frame a cada N frames.
        conf: Limiar de confiança mínimo.

    Returns:
        Lista de frames com detecções e flag de alerta.
    """
    model = load_model(weights_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    results = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % sample_rate == 0:
            detections = detect_in_frame(model, frame, conf)
            results.append({
                "frame": frame_count,
                "timestamp_s": round(frame_count / fps, 2),
                "detections": detections,
                "alert": len(detections) > 0,
            })

        frame_count += 1

    cap.release()
    return results
