"""Testes unitários para módulos de áudio."""

import pytest
from src.audio.emotion_audio import classify_emotion_heuristic


def test_classify_sad():
    features = {
        "pitch_mean_hz": 120.0,
        "pitch_std_hz": 10.0,
        "energy_mean": 0.02,
        "energy_std": 0.01,
        "speech_rate_proxy": 0.05,
        "mfcc_means": [0.0] * 13,
    }
    result = classify_emotion_heuristic(features)
    assert result["predicted_emotion"] == "sad"
    assert result["alert"] is True


def test_classify_neutral():
    features = {
        "pitch_mean_hz": 200.0,
        "pitch_std_hz": 20.0,
        "energy_mean": 0.06,
        "energy_std": 0.02,
        "speech_rate_proxy": 0.07,
        "mfcc_means": [0.0] * 13,
    }
    result = classify_emotion_heuristic(features)
    assert result["predicted_emotion"] == "neutral"
    assert result["alert"] is False
