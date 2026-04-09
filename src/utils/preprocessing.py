"""Funções auxiliares de pré-processamento compartilhadas entre módulos."""

import cv2
import numpy as np
from pathlib import Path


def resize_frame(frame: np.ndarray, width: int = 640) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = width / w
    return cv2.resize(frame, (width, int(h * scale)))


def extract_frames(video_path: str, output_dir: str, every_n: int = 30) -> list[str]:
    """Extrai frames de um vídeo e salva como imagens. Retorna lista de caminhos."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    paths = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % every_n == 0:
            path = out / f"frame_{frame_count:06d}.jpg"
            cv2.imwrite(str(path), frame)
            paths.append(str(path))
        frame_count += 1

    cap.release()
    return paths


def anonymize_face(frame: np.ndarray, face_bbox: tuple) -> np.ndarray:
    """Borra região facial para anonimização — LGPD compliance."""
    x, y, w, h = face_bbox
    roi = frame[y:y+h, x:x+w]
    blurred = cv2.GaussianBlur(roi, (51, 51), 0)
    frame[y:y+h, x:x+w] = blurred
    return frame
