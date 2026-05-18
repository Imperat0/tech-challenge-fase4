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
import traceback
from pathlib import Path
from dotenv import load_dotenv
from src.utils.logger import setup_logger, log_exception

load_dotenv()
logger = setup_logger(__name__)


def run_pipeline(
    video_path: str | None = None,
    audio_path: str | None = None,
    clinical_text: str | None = None,
    patient_id: str = "ANONIMO",
    output_dir: str = "reports",
) -> dict:
    """
    Executa o pipeline multimodal completo com logging detalhado de erros.

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
    logger.info("=" * 70)
    logger.info("🎯 INICIANDO PIPELINE MULTIMODAL")
    logger.info(f"📍 Patient ID: {patient_id} | Saída: {output_dir}")
    
    try:
        from src.video.face_emotion import process_video_emotions
        from src.video.pose_detection import process_video_pose
        from src.video.yolo_detector import process_video_yolo
        from src.video.violence_screening import fuse_video_signals
        from src.audio.transcriber import transcribe_with_whisper, extract_audio_from_video
        from src.audio.emotion_audio import analyze_audio_emotion
        from src.text.gpt_analysis import analyze_medical_text, generate_clinical_report
        from src.text.report_generator import build_report, save_report
        from src.cloud.alerts import send_alert
        
        logger.info("✅ Todos os módulos importados com sucesso")
    except ImportError as e:
        logger.error(f"❌ [IMPORTAÇÃO] Erro ao carregar módulos: {e}")
        logger.debug(traceback.format_exc())
        raise

    results = {}

    # ── 1. ANÁLISE DE VÍDEO ─────────────────────────────────────────────────
    if video_path:
        try:
            logger.info("┌─ 1️⃣  ANÁLISE DE VÍDEO")
            logger.info(f"   Arquivo: {video_path}")
            
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")
            
            file_size = os.path.getsize(video_path) / (1024 * 1024)
            logger.info(f"   Tamanho: {file_size:.2f} MB")
            
            logger.info("   → Detectando emoções...")
            emotion_results = process_video_emotions(video_path)
            logger.info(f"   ✓ Emoções: {len(emotion_results)} frames processados")
            
            logger.info("   → Detectando pose...")
            pose_results = process_video_pose(video_path)
            logger.info(f"   ✓ Pose: {len(pose_results)} frames processados")
            
            logger.info("   → Detectando objetos (YOLO)...")
            yolo_results = process_video_yolo(video_path)
            logger.info(f"   ✓ YOLO: {len(yolo_results)} detecções")
            
            logger.info("   → Sintetizando violência...")
            violence_signals = fuse_video_signals(emotion_results, pose_results)
            logger.info(f"   ✓ Sinais: {len(violence_signals)} gerados")

            all_video_alerts = [r for r in violence_signals if r["alert"]] + \
                               [r for r in yolo_results if r["alert"]]

            results["video"] = {
                "emotion_analysis": emotion_results,
                "pose_analysis": pose_results,
                "yolo_detections": yolo_results,
                "violence_screening": violence_signals,
                "total_alerts": len(all_video_alerts),
            }
            logger.info(f"└─ 🎥 Vídeo: ✅ {len(all_video_alerts)} ALERTAS DETECTADOS")

        except FileNotFoundError as e:
            error_msg = log_exception(logger, e, "VÍDEO_ARQUIVO")
            logger.warning(f"⚠️  Análise de vídeo ignorada: {error_msg}")
            results["video"] = {"error": str(e), "total_alerts": 0}
        except Exception as e:
            error_msg = log_exception(logger, e, "VÍDEO_PROCESSAMENTO")
            logger.warning(f"⚠️  Falha na análise de vídeo: {error_msg}")
            results["video"] = {"error": str(e), "total_alerts": 0}

    # ── 2. ANÁLISE DE ÁUDIO ─────────────────────────────────────────────────
    audio_path_extracted = None
    if audio_path is None and video_path:
        try:
            logger.info("┌─ 2️⃣  EXTRAÇÃO DE ÁUDIO DO VÍDEO")
            audio_path_extracted = extract_audio_from_video(video_path)
            audio_path = audio_path_extracted
            logger.info(f"└─ 🎙️  Áudio: ✅ Extraído em {audio_path}")
        except Exception as e:
            error_msg = log_exception(logger, e, "AUDIO_EXTRACAO")
            logger.warning(f"⚠️  Não foi possível extrair áudio: {error_msg}")

    if audio_path:
        try:
            logger.info("┌─ 2️⃣  ANÁLISE DE ÁUDIO")
            logger.info(f"   Arquivo: {audio_path}")
            
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Áudio não encontrado: {audio_path}")
            
            logger.info("   → Transcrevendo áudio (Whisper)...")
            transcript = transcribe_with_whisper(audio_path)
            logger.info(f"   ✓ Transcrição: {len(transcript.get('text', ''))} caracteres")
            
            logger.info("   → Analisando emoção do áudio...")
            emotion_audio = analyze_audio_emotion(audio_path)
            logger.info(f"   ✓ Emoção: {emotion_audio.get('predicted_emotion')}")

            results["audio"] = {
                "transcript": transcript,
                "emotion": emotion_audio,
                "alert": emotion_audio.get("alert", False),
            }
            logger.info(f"└─ 🎵 Áudio: ✅ Processado | Alerta: {results['audio']['alert']}")

            # Usa o texto transcrito para análise de violência se não houver texto clínico
            if clinical_text is None and transcript.get("text"):
                clinical_text = transcript["text"]
                logger.info("   💡 Usando texto transcrito para análise clínica")

        except FileNotFoundError as e:
            error_msg = log_exception(logger, e, "AUDIO_ARQUIVO")
            logger.warning(f"⚠️  Análise de áudio ignorada: {error_msg}")
            results["audio"] = {"error": str(e), "alert": False}
        except Exception as e:
            error_msg = log_exception(logger, e, "AUDIO_PROCESSAMENTO")
            logger.warning(f"⚠️  Falha na análise de áudio: {error_msg}")
            results["audio"] = {"error": str(e), "alert": False}

    # ── 3. ANÁLISE DE TEXTO ─────────────────────────────────────────────────
    if clinical_text:
        try:
            logger.info("┌─ 3️⃣  ANÁLISE DE TEXTO CLÍNICO")
            logger.info(f"   Tamanho: {len(clinical_text)} caracteres")
            
            logger.info("   → Analisando risco maternal...")
            maternal_analysis = analyze_medical_text(clinical_text, analysis_type="maternal")
            logger.info(f"   ✓ Risco maternal: {maternal_analysis.get('risk_level')}")
            
            logger.info("   → Analisando indicadores de violência...")
            violence_text_analysis = analyze_medical_text(clinical_text, analysis_type="violence")
            logger.info(f"   ✓ Violência detectada: {violence_text_analysis.get('violence_detected', False)}")

            results["text"] = {
                "maternal_risk": maternal_analysis,
                "violence_indicators": violence_text_analysis,
                "risk_level": maternal_analysis.get("risk_level", "low"),
                "recommendations": maternal_analysis.get("recommendations", []),
            }
            logger.info(f"└─ 📄 Texto: ✅ Processado | Risco: {results['text']['risk_level'].upper()}")

        except Exception as e:
            error_msg = log_exception(logger, e, "TEXTO_PROCESSAMENTO")
            logger.warning(f"⚠️  Falha na análise de texto: {error_msg}")
            results["text"] = {"error": str(e), "risk_level": "low", "recommendations": []}

    # ── 4. FUSÃO E RELATÓRIO ─────────────────────────────────────────────────
    try:
        logger.info("┌─ 4️⃣  GERANDO RELATÓRIO CONSOLIDADO")
        
        logger.info("   → Consolidando dados...")
        report = build_report(
            video_results=results.get("video", {}).get("violence_screening", []),
            audio_results=results.get("audio", {"alert": False}),
            text_analysis=results.get("text", {"risk_level": "low", "recommendations": []}),
            patient_id=patient_id,
        )
        
        logger.info("   → Salvando relatório...")
        report_path = save_report(report, output_dir)
        logger.info(f"   ✓ Relatório: {report_path}")
        logger.info(f"└─ 📊 Relatório: ✅ Gerado | Risco geral: {report.get('overall_risk').upper()}")

    except Exception as e:
        error_msg = log_exception(logger, e, "RELATORIO_GERACAO")
        logger.error(f"❌ Falha ao gerar relatório: {error_msg}")
        logger.debug(traceback.format_exc())
        # Retorna dict mínimo com erro
        return {
            "overall_risk": "unknown",
            "requires_immediate_attention": False,
            "error": str(e),
            "message": "Falha ao gerar relatório consolidado"
        }

    # ── 5. ALERTA ────────────────────────────────────────────────────────────
    try:
        if report.get("requires_immediate_attention"):
            logger.info("┌─ 5️⃣  ENVIANDO ALERTA")
            send_alert(report, channel=os.getenv("ALERT_CHANNEL", "log"))
            logger.warning(f"└─ 🚨 ALERTA ENVIADO: Risco {report['overall_risk'].upper()}")
        else:
            logger.info("└─ ✅ Pipeline concluído sem alertas")
    except Exception as e:
        error_msg = log_exception(logger, e, "ALERTA_ENVIO")
        logger.warning(f"⚠️  Falha ao enviar alerta: {error_msg}")

    logger.info("=" * 70)
    logger.info("✅ PIPELINE CONCLUÍDO COM SUCESSO")
    
    return report


if __name__ == "__main__":
    import sys

    video = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        result = run_pipeline(video_path=video)
        print(f"\nRisco geral: {result['overall_risk'].upper()}")
        print(f"Atenção imediata: {result['requires_immediate_attention']}")
    except Exception as e:
        logger.error(f"❌ Pipeline falhou: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)
