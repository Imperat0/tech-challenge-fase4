"""Testes de integração do pipeline multimodal."""

import pytest
import json
import os
import tempfile
from src.text.report_generator import build_report, save_report, check_vital_signs, load_clinical_thresholds


def test_build_report_low_risk():
    report = build_report(
        video_results=[],
        audio_results={"alert": False},
        text_analysis={"risk_level": "low", "recommendations": []},
        patient_id="TEST-001",
    )
    assert report["overall_risk"] == "low"
    assert report["requires_immediate_attention"] is False


def test_build_report_critical():
    video_alerts = [{"frame": i, "timestamp_s": float(i), "alert": True} for i in range(5)]
    report = build_report(
        video_results=video_alerts,
        audio_results={"alert": True},
        text_analysis={"risk_level": "high", "recommendations": ["Atenção imediata"]},
        patient_id="TEST-002",
    )
    assert report["requires_immediate_attention"] is True


def test_build_report_with_vitals():
    """Sinais vitais acima do limiar devem aumentar o score."""
    report = build_report(
        video_results=[],
        audio_results={"alert": False},
        text_analysis={"risk_level": "low", "recommendations": []},
        vitals={"SystolicBP": 160, "DiastolicBP": 100},  # hipertensão
        patient_id="TEST-003",
    )
    assert report["vitals_flags"]["hipertensao_sistolica"] is True
    assert report["vitals_flags"]["any_alert"] is True
    # Score text=0 + vitals=1 → medium
    assert report["overall_risk"] in ("medium", "high", "critical")


def test_check_vital_signs_normal():
    flags = check_vital_signs({"SystolicBP": 115, "DiastolicBP": 75, "HeartRate": 80})
    assert flags["any_alert"] is False


def test_check_vital_signs_high_bp():
    flags = check_vital_signs({"SystolicBP": 145})
    assert flags["hipertensao_sistolica"] is True
    assert flags["any_alert"] is True


def test_check_vital_signs_diabetes():
    flags = check_vital_signs({"BS": 9.5})
    assert flags["diabetes_gestacional"] is True


def test_load_clinical_thresholds_returns_dict():
    thresholds = load_clinical_thresholds()
    assert isinstance(thresholds, dict)
    assert "SystolicBP_high_risk" in thresholds


def test_save_report():
    report = build_report(
        video_results=[],
        audio_results={"alert": False},
        text_analysis={"risk_level": "low", "recommendations": []},
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = save_report(report, tmp)
        assert os.path.exists(path)
        with open(path) as f:
            saved = json.load(f)
        assert saved["overall_risk"] == report["overall_risk"]
        assert "scores" in saved
