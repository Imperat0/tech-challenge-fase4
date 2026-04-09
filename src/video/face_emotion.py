"""
Análise de expressão facial e emoção usando DeepFace.
Aula 02 - Reconhecimento facial e análise de expressões emocionais em vídeos.

Objetivos cobertos:
- Identificar sinais não-verbais de desconforto ou medo em consultas
- Triagem de violência: detecção de linguagem corporal indicativa de abuso
"""

from deepface import DeepFace
import cv2
import numpy as np


EMOTIONS_OF_CONCERN = {"fear", "sad", "disgust", "angry"}


def analyze_frame_emotions(frame: np.ndarray) -> list[dict]:
    """Analisa emoções de todos os rostos detectados em um frame."""
    try:
        results = DeepFace.analyze(
            frame,
            actions=["emotion"],
            enforce_detection=False,
            silent=True,
        )
        return results if isinstance(results, list) else [results]
    except Exception:
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
