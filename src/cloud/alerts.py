"""
Sistema de alertas para equipe médica.
Dispara notificações quando riscos são detectados no pipeline multimodal.
"""

import os
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def send_alert(report: dict, channel: str = "log") -> bool:
    """
    Envia alerta para a equipe médica.

    Args:
        report: Relatório gerado pelo report_generator.
        channel: "log" | "webhook" | "email" (configurar conforme ambiente)

    Returns:
        True se o alerta foi enviado com sucesso.
    """
    if not report.get("requires_immediate_attention"):
        return False

    message = _format_alert_message(report)

    if channel == "log":
        logger.warning("ALERTA MÉDICO: %s", message)
        return True

    if channel == "webhook":
        return _send_webhook(message)

    logger.info("Canal '%s' não implementado. Alerta: %s", channel, message)
    return False


def _format_alert_message(report: dict) -> str:
    return (
        f"[{report['timestamp']}] "
        f"Paciente {report['patient_id']} | "
        f"Risco: {report['overall_risk'].upper()} | "
        f"Alertas de vídeo: {report['video_summary']['alert_frames']} frames | "
        f"Recomendações: {'; '.join(report.get('recommendations', []))}"
    )


def _send_webhook(message: str) -> bool:
    """Envia alerta via webhook (configure ALERT_WEBHOOK_URL no .env)."""
    import requests

    url = os.environ.get("ALERT_WEBHOOK_URL")
    if not url:
        logger.error("ALERT_WEBHOOK_URL não configurada.")
        return False

    try:
        resp = requests.post(url, json={"text": message}, timeout=5)
        resp.raise_for_status()
        return True
    except requests.RequestException as e:
        logger.error("Falha ao enviar webhook: %s", e)
        return False
