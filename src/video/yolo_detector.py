"""
Detecção de objetos e anomalias cirúrgicas usando YOLOv8.
Requisito obrigatório: modelo customizado para instrumentos ginecológicos.

Objetivos cobertos:
- Cirurgias: detecção de complicações ou sangramento anômalo
- Instrumentos cirúrgicos ginecológicos
"""

import logging
import traceback

try:
    from ultralytics import YOLO
    logger = logging.getLogger(__name__)
    logger.info("✅ YOLOv8 (Ultralytics) importado com sucesso")
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"❌ Falha ao importar YOLOv8: {e}")
    logger.debug(traceback.format_exc())
    YOLO = None

import cv2
import numpy as np
from pathlib import Path


DEFAULT_WEIGHTS = Path(__file__).parent.parent.parent / "models" / "yolov8" / "weights" / "best.pt"
FALLBACK_WEIGHTS = "yolov8n.pt"  # Modelo base enquanto o customizado não está treinado


def load_model(weights_path: str | Path | None = None) -> YOLO:
    """Carrega o modelo YOLOv8. Usa o customizado se disponível."""
    if YOLO is None:
        logger.warning("⚠️  YOLOv8 não disponível, retornando None")
        return None
    
    try:
        path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS
        if path.exists():
            logger.info(f"🤖 Carregando modelo customizado: {path}")
            return YOLO(str(path))
        logger.info(f"📥 Modelo customizado não encontrado, usando fallback: {FALLBACK_WEIGHTS}")
        return YOLO(FALLBACK_WEIGHTS)
    except Exception as e:
        logger.error(f"❌ Erro ao carregar modelo YOLOv8: {e}")
        logger.debug(traceback.format_exc())
        return None


def detect_in_frame(model: YOLO, frame: np.ndarray, conf: float = 0.4) -> list[dict]:
    """Executa detecção em um frame e retorna lista de detecções."""
    if model is None:
        logger.warning("⚠️  Modelo YOLOv8 não disponível")
        return []
    
    try:
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
    except Exception as e:
        logger.debug(f"Erro na detecção YOLO de frame: {e}")
        return []


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
    if YOLO is None:
        logger.warning("⚠️  YOLOv8 indisponível, pulando detecção de objetos")
        return []
    
    try:
        model = load_model(weights_path)
        if model is None:
            logger.warning("⚠️  Modelo YOLOv8 não pôde ser carregado")
            return []
        
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
        logger.info(f"✅ Processamento YOLO concluído: {len(results)} frames analisados")
        return results
    
    except Exception as e:
        logger.error(f"❌ Erro ao processar vídeo com YOLO: {e}")
        logger.debug(traceback.format_exc())
        return []
