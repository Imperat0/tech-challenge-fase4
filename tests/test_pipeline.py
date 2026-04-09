"""Testes de integração do pipeline multimodal."""

import pytest
from src.text.report_generator import build_report, save_report
import tempfile
import json
import os


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
