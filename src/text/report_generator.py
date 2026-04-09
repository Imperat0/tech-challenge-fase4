"""
Geração de relatórios automáticos especializados.
Consolida resultados multimodais em relatório estruturado.
"""

import json
from datetime import datetime
from pathlib import Path


def build_report(
    video_results: list[dict],
    audio_results: dict,
    text_analysis: dict,
    patient_id: str = "ANONIMO",
) -> dict:
    """
    Consolida todos os resultados em um relatório estruturado.

    Returns:
        Dict com relatório completo pronto para exportar.
    """
    video_alerts = [r for r in video_results if r.get("alert")]
    audio_alert = audio_results.get("alert", False)
    text_risk = text_analysis.get("risk_level", "low")

    # Score geral de risco
    risk_scores = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    base_score = risk_scores.get(text_risk, 0)
    if len(video_alerts) > 3:
        base_score += 1
    if audio_alert:
        base_score += 1

    overall_risk = ["low", "medium", "high", "critical"][min(base_score, 3)]

    return {
        "report_id": f"RPT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "patient_id": patient_id,
        "timestamp": datetime.now().isoformat(),
        "overall_risk": overall_risk,
        "requires_immediate_attention": overall_risk in ("high", "critical"),
        "video_summary": {
            "total_frames_analyzed": len(video_results),
            "alert_frames": len(video_alerts),
            "alert_timestamps": [r["timestamp_s"] for r in video_alerts],
        },
        "audio_summary": audio_results,
        "text_summary": text_analysis,
        "recommendations": text_analysis.get("recommendations", []),
    }


def save_report(report: dict, output_dir: str = "reports") -> str:
    """Salva relatório em JSON. Retorna o caminho do arquivo."""
    out = Path(output_dir)
    out.mkdir(exist_ok=True)
    path = out / f"{report['report_id']}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return str(path)
