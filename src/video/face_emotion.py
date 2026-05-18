"""
Análise de expressão facial e emoção usando DeepFace.
Aula 02 - Reconhecimento facial e análise de expressões emocionais em vídeos.

Objetivos cobertos:
- Identificar sinais não-verbais de desconforto ou medo em consultas
- Triagem de violência: detecção de linguagem corporal indicativa de abuso
"""

import logging
import traceback

try:
    from deepface import DeepFace
    logger = logging.getLogger(__name__)
    logger.info("✅ DeepFace importado com sucesso")
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"❌ Falha ao importar DeepFace: {e}")
    logger.debug(traceback.format_exc())
    DeepFace = None

import cv2
import numpy as np


EMOTIONS_OF_CONCERN = {"fear", "sad", "disgust", "angry"}


def analyze_frame_emotions(frame: np.ndarray) -> list[dict]:
    """Analisa emoções de todos os rostos detectados em um frame."""
    if DeepFace is None:
        logger.warning("⚠️  DeepFace não disponível, retornando lista vazia")
        return []
    
    try:
        results = DeepFace.analyze(
            frame,
            actions=["emotion"],
            enforce_detection=False,
            silent=True,
        )
        return results if isinstance(results, list) else [results]
    except Exception as e:
        logger.debug(f"Erro na análise de emoção de frame: {e}")
        return []


def process_video_emotions(video_path: str, sample_rate: int = 30) -> list[dict]:
    """
    Processa um vídeo e retorna análise emocional por intervalo de frames.

    Args:
        video_path: Caminho para o vídeo.
        sample_rate: Analisar 1 frame a cada N frames.

    Returns:
        Lista de dicts com frame_number, timestamp, emotions e flag de alerta.
    """
    if DeepFace is None:
        logger.warning("⚠️  DeepFace indisponível, pulando análise de emoção facial")
        return []
    
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        results = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % sample_rate == 0:
                analyses = analyze_frame_emotions(frame)
                for analysis in analyses:
                    dominant = analysis.get("dominant_emotion", "")
                    alert = dominant in EMOTIONS_OF_CONCERN
                    results.append({
                        "frame": frame_count,
                        "timestamp_s": round(frame_count / fps, 2),
                        "dominant_emotion": dominant,
                        "emotions": analysis.get("emotion", {}),
                        "alert": alert,
                    })

            frame_count += 1

        cap.release()
        return results
    
    except Exception as e:
        logger.error(f"❌ Erro ao processar vídeo com DeepFace: {e}")
        logger.debug(traceback.format_exc())
        return []
