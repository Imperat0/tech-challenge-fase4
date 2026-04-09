"""
Orquestrador Multimodal — Tech Challenge Fase 4
PosTech FIAP | IA para Devs

Fluxo principal:
    Vídeo → [YOLOv8 + DeepFace + MediaPipe] ─┐
    Áudio → [Whisper + Azure Speech + Prosódia] ┼→ Fusão → GPT-4o → Relatório → Alerta
    Texto → [Azure Language + GPT-4o] ─────────┘
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


def run_pipeline(
    video_path: str | None = None,
    audio_path: str | None = None,
    clinical_text: str | None = None,
    patient_id: str = "ANONIMO",
    output_dir: str = "reports",
) -> dict:
    """
    Executa o pipeline multimodal completo.

    Args:
        video_path: Vídeo de consulta ou cirurgia.
        audio_path: Gravação de consulta. Se None e video_path fornecido,
                    extrai o áudio do vídeo automaticamente.
        clinical_text: Laudo ou texto de prontuário para análise.
        patient_id: Identificador anonimizado da paciente.
        output_dir: Pasta para salvar o relatório gerado.

    Returns:
        Relatório consolidado como dict.
    """
    from src.video.face_emotion import process_video_emotions
    from src.video.pose_detection import process_video_pose
    from src.video.yolo_detector import process_video_yolo
    from src.video.violence_screening import fuse_video_signals
    from src.audio.transcriber import transcribe_with_whisper, extract_audio_from_video
    from src.audio.emotion_audio import analyze_audio_emotion
    from src.text.gpt_analysis import analyze_medical_text, generate_clinical_report
    from src.text.report_generator import build_report, save_report
    from src.cloud.alerts import send_alert

    results = {}

    # ── 1. ANÁLISE DE VÍDEO ─────────────────────────────────────────────────
    if video_path:
        logger.info("Iniciando análise de vídeo: %s", video_path)

        emotion_results = process_video_emotions(video_path)
        pose_results = process_video_pose(video_path)
        yolo_results = process_video_yolo(video_path)
        violence_signals = fuse_video_signals(emotion_results, pose_results)

        all_video_alerts = [r for r in violence_signals if r["alert"]] + \
                           [r for r in yolo_results if r["alert"]]

        results["video"] = {
            "emotion_analysis": emotion_results,
            "pose_analysis": pose_results,
            "yolo_detections": yolo_results,
            "violence_screening": violence_signals,
            "total_alerts": len(all_video_alerts),
        }
        logger.info("Vídeo: %d alertas detectados", len(all_video_alerts))

    # ── 2. ANÁLISE DE ÁUDIO ─────────────────────────────────────────────────
    if audio_path is None and video_path:
        logger.info("Extraindo áudio do vídeo...")
        try:
            audio_path = extract_audio_from_video(video_path)
        except Exception as e:
            logger.warning("Não foi possível extrair áudio: %s", e)

    if audio_path:
        logger.info("Iniciando análise de áudio: %s", audio_path)

        transcript = transcribe_with_whisper(audio_path)
        emotion_audio = analyze_audio_emotion(audio_path)

        results["audio"] = {
            "transcript": transcript,
            "emotion": emotion_audio,
            "alert": emotion_audio.get("alert", False),
        }
        logger.info("Áudio transcrito. Emoção detectada: %s", emotion_audio.get("predicted_emotion"))

        # Usa o texto transcrito para análise de violência se não houver texto clínico
        if clinical_text is None and transcript.get("text"):
            clinical_text = transcript["text"]

    # ── 3. ANÁLISE DE TEXTO ─────────────────────────────────────────────────
    if clinical_text:
        logger.info("Iniciando análise de texto clínico...")

        maternal_analysis = analyze_medical_text(clinical_text, analysis_type="maternal")
        violence_text_analysis = analyze_medical_text(clinical_text, analysis_type="violence")

        results["text"] = {
            "maternal_risk": maternal_analysis,
            "violence_indicators": violence_text_analysis,
            "risk_level": maternal_analysis.get("risk_level", "low"),
            "recommendations": maternal_analysis.get("recommendations", []),
        }
        logger.info("Texto: risco maternal = %s", results["text"]["risk_level"])

    # ── 4. FUSÃO E RELATÓRIO ─────────────────────────────────────────────────
    logger.info("Gerando relatório consolidado...")

    report = build_report(
        video_results=results.get("video", {}).get("violence_screening", []),
        audio_results=results.get("audio", {"alert": False}),
        text_analysis=results.get("text", {"risk_level": "low", "recommendations": []}),
        patient_id=patient_id,
    )

    report_path = save_report(report, output_dir)
    logger.info("Relatório salvo em: %s", report_path)

    # ── 5. ALERTA ────────────────────────────────────────────────────────────
    if report.get("requires_immediate_attention"):
        send_alert(report, channel=os.getenv("ALERT_CHANNEL", "log"))
        logger.warning("ALERTA ENVIADO: risco %s", report["overall_risk"])

    return report


if __name__ == "__main__":
    import sys

    video = sys.argv[1] if len(sys.argv) > 1 else None
    result = run_pipeline(video_path=video)
    print(f"\nRisco geral: {result['overall_risk'].upper()}")
    print(f"Atenção imediata: {result['requires_immediate_attention']}")
