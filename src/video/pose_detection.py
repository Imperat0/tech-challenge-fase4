"""
Detecção de pose e análise de linguagem corporal usando MediaPipe.
Aula 03 - Detecção de atividades e reconhecimento de ações em vídeos.

Objetivos cobertos:
- Triagem de violência: postura defensiva, tremor, posição retraída
- Fisioterapia: análise de movimentos e recuperação pós-parto
"""

import mediapipe as mp
import cv2
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# Índices dos landmarks relevantes para postura defensiva
SHOULDER_L = 11
SHOULDER_R = 12
ELBOW_L = 13
ELBOW_R = 14
WRIST_L = 15
WRIST_R = 16
HIP_L = 23
HIP_R = 24


def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Calcula o ângulo em graus entre três pontos (a-b-c), sendo b o vértice."""
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def detect_defensive_posture(landmarks) -> bool:
    """
    Heurística simples: braços cruzados / ombros encurvados para frente
    indicam postura defensiva associada a medo ou trauma.
    """
    lm = landmarks.landmark
    shoulder_l = np.array([lm[SHOULDER_L].x, lm[SHOULDER_L].y])
    shoulder_r = np.array([lm[SHOULDER_R].x, lm[SHOULDER_R].y])
    wrist_l = np.array([lm[WRIST_L].x, lm[WRIST_L].y])
    wrist_r = np.array([lm[WRIST_R].x, lm[WRIST_R].y])

    # Pulsos cruzados (x do pulso esquerdo > pulso direito = braços cruzados)
    wrists_crossed = wrist_l[0] > wrist_r[0]

    # Ombros encurvados: distância horizontal menor que o esperado
    shoulder_width = abs(shoulder_l[0] - shoulder_r[0])
    hunched = shoulder_width < 0.15  # threshold normalizado 0-1

    return wrists_crossed or hunched


def process_video_pose(video_path: str, sample_rate: int = 30) -> list[dict]:
    """
    Processa vídeo extraindo dados de pose e sinalizando posturas de risco.

    Returns:
        Lista com frame, timestamp, postura detectada e flag de alerta.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    results = []
    frame_count = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % sample_rate == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)

                if result.pose_landmarks:
                    defensive = detect_defensive_posture(result.pose_landmarks)
                    results.append({
                        "frame": frame_count,
                        "timestamp_s": round(frame_count / fps, 2),
                        "pose_detected": True,
                        "defensive_posture": defensive,
                        "alert": defensive,
                    })
                else:
                    results.append({
                        "frame": frame_count,
                        "timestamp_s": round(frame_count / fps, 2),
                        "pose_detected": False,
                        "defensive_posture": False,
                        "alert": False,
                    })

            frame_count += 1

    cap.release()
    return results
