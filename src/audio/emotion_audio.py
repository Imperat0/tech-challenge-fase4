"""
Análise de emoção e prosódia no áudio de consultas médicas.

Detecta sinais vocais de:
- Depressão pós-parto (voz monótona, ritmo lento)
- Ansiedade gestacional (fala acelerada, tom alto)
- Trauma por violência doméstica (hesitação, tremor vocal)

Dataset de referência: RAVDESS, CREMA-D
"""

import librosa
import numpy as np


EMOTION_LABELS = {
    0: "neutral",
    1: "calm",
    2: "happy",
    3: "sad",
    4: "angry",
    5: "fearful",
    6: "disgust",
    7: "surprised",
}

EMOTIONS_OF_CONCERN = {"sad", "fearful", "angry", "disgust"}


def extract_prosody_features(audio_path: str) -> dict:
    """
    Extrai características prosódicas do áudio:
    - Pitch (F0): variação de tom
    - Energy (RMS): volume / energia
    - Speech rate: velocidade da fala (via ZCR)
    - MFCCs: coeficientes cepstrais para classificação de emoção
    """
    y, sr = librosa.load(audio_path, sr=None)

    # Pitch
    f0, _, _ = librosa.pyin(y, fmin=50, fmax=400, sr=sr)
    f0_clean = f0[~np.isnan(f0)]
    pitch_mean = float(np.mean(f0_clean)) if len(f0_clean) > 0 else 0.0
    pitch_std = float(np.std(f0_clean)) if len(f0_clean) > 0 else 0.0

    # Energia (volume)
    rms = librosa.feature.rms(y=y)[0]
    energy_mean = float(np.mean(rms))
    energy_std = float(np.std(rms))

    # Zero Crossing Rate (proxy de velocidade de fala)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    speech_rate_proxy = float(np.mean(zcr))

    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = mfccs.mean(axis=1).tolist()

    return {
        "pitch_mean_hz": pitch_mean,
        "pitch_std_hz": pitch_std,
        "energy_mean": energy_mean,
        "energy_std": energy_std,
        "speech_rate_proxy": speech_rate_proxy,
        "mfcc_means": mfcc_means,
    }


def classify_emotion_heuristic(features: dict) -> dict:
    """
    Classificação heurística baseada em prosódia.
    Substitua por modelo treinado com RAVDESS para maior precisão.

    Heurísticas:
    - Pitch baixo + energia baixa = possível tristeza/depressão
    - Pitch alto + energia alta = possível ansiedade/medo
    - Pitch variável alto = possível raiva
    """
    p_mean = features["pitch_mean_hz"]
    p_std = features["pitch_std_hz"]
    e_mean = features["energy_mean"]

    if p_mean < 150 and e_mean < 0.05:
        emotion = "sad"
    elif p_mean > 250 and e_mean > 0.1:
        emotion = "fearful"
    elif p_std > 50 and e_mean > 0.08:
        emotion = "angry"
    else:
        emotion = "neutral"

    return {
        "predicted_emotion": emotion,
        "alert": emotion in EMOTIONS_OF_CONCERN,
        "features": features,
    }


def analyze_audio_emotion(audio_path: str) -> dict:
    """Pipeline completo: extrai features e classifica emoção."""
    features = extract_prosody_features(audio_path)
    return classify_emotion_heuristic(features)
