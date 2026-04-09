"""
Geração de relatórios automáticos especializados.
Consolida resultados multimodais em relatório estruturado.

Integra os limiares clínicos exportados pelo Notebook 01
(data/datasets/maternal_risk/clinical_thresholds.json).
"""

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Limiares clínicos padrão — substituídos pelos do Notebook 01 quando disponíveis
_DEFAULT_THRESHOLDS = {
    "SystolicBP_high_risk" : 140,
    "DiastolicBP_high_risk": 90,
    "BS_high_risk"         : 7.8,
    "BodyTemp_fever"       : 100.0,
    "HeartRate_tachycardia": 100,
}

_THRESHOLDS_PATH = (
    Path(__file__).parent.parent.parent
    / "data/datasets/maternal_risk/clinical_thresholds.json"
)

_thresholds_cache: dict | None = None


def load_clinical_thresholds() -> dict:
    """
    Carrega limiares clínicos do Notebook 01.
    Usa valores padrão se o arquivo não existir (antes de rodar o notebook).
    """
    global _thresholds_cache
    if _thresholds_cache is not None:
        return _thresholds_cache

    if _THRESHOLDS_PATH.exists():
        try:
            _thresholds_cache = json.loads(_THRESHOLDS_PATH.read_text())
            logger.info("Limiares clínicos carregados do Notebook 01.")
            return _thresholds_cache
        except Exception as e:
            logger.warning("Falha ao carregar limiares: %s", e)

    logger.debug("Usando limiares clínicos padrão.")
    _thresholds_cache = _DEFAULT_THRESHOLDS.copy()
    return _thresholds_cache


def check_vital_signs(vitals: dict) -> dict:
    """
    Verifica sinais vitais contra limiares clínicos.

    Args:
        vitals: Dict com chaves SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate.

    Returns:
        Dict com flags de alerta por indicador.
    """
    th = load_clinical_thresholds()
    flags = {}

    if "SystolicBP" in vitals:
        flags["hipertensao_sistolica"] = vitals["SystolicBP"] >= th["SystolicBP_high_risk"]
    if "DiastolicBP" in vitals:
        flags["hipertensao_diastolica"] = vitals["DiastolicBP"] >= th["DiastolicBP_high_risk"]
    if "BS" in vitals:
        flags["diabetes_gestacional"] = vitals["BS"] >= th["BS_high_risk"]
    if "BodyTemp" in vitals:
        flags["febre"] = vitals["BodyTemp"] >= th["BodyTemp_fever"]
    if "HeartRate" in vitals:
        flags["taquicardia"] = vitals["HeartRate"] > th["HeartRate_tachycardia"]

    flags["any_alert"] = any(flags.values())
    return flags


def build_report(
    video_results : list[dict],
    audio_results : dict,
    text_analysis : dict,
    patient_id    : str = "ANONIMO",
    vitals        : dict | None = None,
) -> dict:
    """
    Consolida todos os resultados em um relatório estruturado.

    Args:
        video_results : Lista de frames com campo 'alert'.
        audio_results : Dict da análise de áudio (alert, predicted_emotion).
        text_analysis : Dict da análise de texto (risk_level, recommendations).
        patient_id    : Identificador anonimizado da paciente.
        vitals        : Dict opcional com sinais vitais para checagem clínica.

    Returns:
        Dict com relatório completo pronto para exportar e alertar.
    """
    video_alerts = [r for r in video_results if r.get("alert")]
    audio_alert  = audio_results.get("alert", False)
    text_risk    = text_analysis.get("risk_level", "low")

    # Score base pelo texto clínico (maior peso — laudo é o mais confiável)
    risk_scores = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    base_score  = risk_scores.get(text_risk, 0)

    # Sinal de vídeo: > 3 frames com alerta = score +1
    if len(video_alerts) > 3:
        base_score += 1

    # Sinal de áudio
    if audio_alert:
        base_score += 1

    # Sinais vitais (se fornecidos) — usa limiares do Notebook 01
    vitals_flags = {}
    if vitals:
        vitals_flags = check_vital_signs(vitals)
        if vitals_flags.get("any_alert"):
            base_score += 1

    overall_risk = ["low", "medium", "high", "critical"][min(base_score, 3)]

    return {
        "report_id"                  : f"RPT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "patient_id"                 : patient_id,
        "timestamp"                  : datetime.now().isoformat(),
        "overall_risk"               : overall_risk,
        "requires_immediate_attention": overall_risk in ("high", "critical"),
        "scores": {
            "text" : risk_scores.get(text_risk, 0),
            "video": 1 if len(video_alerts) > 3 else 0,
            "audio": 1 if audio_alert else 0,
            "vitals": 1 if vitals_flags.get("any_alert") else 0,
        },
        "video_summary" : {
            "total_frames_analyzed": len(video_results),
            "alert_frames"         : len(video_alerts),
            "alert_timestamps"     : [r.get("timestamp_s") for r in video_alerts],
        },
        "audio_summary" : audio_results,
        "text_summary"  : text_analysis,
        "vitals_flags"  : vitals_flags,
        "recommendations": text_analysis.get("recommendations", []),
    }


def save_report(report: dict, output_dir: str = "reports") -> str:
    """Salva relatório em JSON. Retorna o caminho do arquivo."""
    out  = Path(output_dir)
    out.mkdir(exist_ok=True)
    path = out / f"{report['report_id']}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return str(path)
