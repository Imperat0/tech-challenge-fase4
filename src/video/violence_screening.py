"""
Triagem de violência doméstica via fusão de sinais de emoção + pose.

Combina os resultados de face_emotion.py e pose_detection.py
para gerar um score consolidado de risco por timestamp.
"""


def fuse_video_signals(
    emotion_results: list[dict],
    pose_results: list[dict],
) -> list[dict]:
    """
    Funde análise emocional e de pose em um score de risco unificado.

    Regra heurística:
    - Emoção de risco (medo/tristeza/raiva) = +1 ponto
    - Postura defensiva detectada = +1 ponto
    - Score >= 1 gera alerta

    Returns:
        Lista com timestamp, score de risco e flag de alerta.
    """
    # Indexar por frame
    emotion_by_frame = {r["frame"]: r for r in emotion_results}
    pose_by_frame = {r["frame"]: r for r in pose_results}

    all_frames = sorted(set(emotion_by_frame) | set(pose_by_frame))
    fused = []

    for frame in all_frames:
        em = emotion_by_frame.get(frame, {})
        po = pose_by_frame.get(frame, {})

        score = int(em.get("alert", False)) + int(po.get("alert", False))
        fused.append({
            "frame": frame,
            "timestamp_s": em.get("timestamp_s") or po.get("timestamp_s", 0),
            "dominant_emotion": em.get("dominant_emotion", "N/A"),
            "defensive_posture": po.get("defensive_posture", False),
            "risk_score": score,
            "alert": score >= 1,
        })

    return fused
