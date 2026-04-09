"""Testes unitários para módulos de vídeo."""

import numpy as np
import pytest
from src.video.violence_screening import fuse_video_signals


def test_fuse_signals_no_alerts():
    emotions = [{"frame": 0, "timestamp_s": 0.0, "dominant_emotion": "happy", "alert": False}]
    poses = [{"frame": 0, "timestamp_s": 0.0, "defensive_posture": False, "alert": False}]
    result = fuse_video_signals(emotions, poses)
    assert result[0]["risk_score"] == 0
    assert result[0]["alert"] is False


def test_fuse_signals_both_alerts():
    emotions = [{"frame": 0, "timestamp_s": 0.0, "dominant_emotion": "fear", "alert": True}]
    poses = [{"frame": 0, "timestamp_s": 0.0, "defensive_posture": True, "alert": True}]
    result = fuse_video_signals(emotions, poses)
    assert result[0]["risk_score"] == 2
    assert result[0]["alert"] is True


def test_fuse_signals_partial_alert():
    emotions = [{"frame": 5, "timestamp_s": 0.5, "dominant_emotion": "sad", "alert": True}]
    poses = [{"frame": 5, "timestamp_s": 0.5, "defensive_posture": False, "alert": False}]
    result = fuse_video_signals(emotions, poses)
    assert result[0]["risk_score"] == 1
    assert result[0]["alert"] is True
