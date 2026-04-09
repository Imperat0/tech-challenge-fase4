"""
Análise de emoção e prosódia no áudio de consultas médicas.

Detecta sinais vocais de:
- Depressão pós-parto (voz monótona, ritmo lento)
- Ansiedade gestacional (fala acelerada, tom alto)
- Trauma por violência doméstica (hesitação, tremor vocal)

Datasets de referência: RAVDESS, CREMA-D
Modelo treinado pelo Notebook 02 (emotion_classifier_binary.pkl).
Fallback para heurísticas de prosódia quando o modelo não estiver disponível.
"""

import pickle
import logging
from pathlib import Path

import librosa
import numpy as np

logger = logging.getLogger(__name__)

EMOTIONS_OF_CONCERN = {"sad", "fearful", "angry", "disgust"}

# Caminho padrão do modelo treinado no Notebook 02
_DEFAULT_MODEL_PATH = (
    Path(__file__).parent.parent.parent
    / "data/datasets/violence_audio/emotion_classifier_binary.pkl"
)

# Cache do modelo carregado (evita recarregar a cada chamada)
_model_cache: dict = {}


def _load_model(model_path: str | Path | None = None) -> dict | None:
    """Carrega o modelo pkl treinado. Retorna None se não encontrado."""
    path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
    key  = str(path)

    if key in _model_cache:
        return _model_cache[key]

    if not path.exists():
        logger.debug("Modelo de emoção não encontrado em %s. Usando heurísticas.", path)
        return None

    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        _model_cache[key] = data
        logger.info("Modelo de emoção vocal carregado: %s", path.name)
        return data
    except Exception as e:
        logger.warning("Falha ao carregar modelo de emoção: %s", e)
        return None


def extract_features_vector(audio_path: str, duration: float = 3.0) -> np.ndarray | None:
    """
    Extrai vetor de 42 features de prosódia para o classificador sklearn.
    Mesmo pipeline do Notebook 02:
      MFCCs (13 média + 13 desvio) + RMS (média + desvio) + ZCR + Centroid + Chroma(12)
    """
    try:
        y, sr = librosa.load(audio_path, duration=duration, res_type="kaiser_fast")
        if len(y) < sr * 0.5:
            return None

        mfccs  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        rms    = librosa.feature.rms(y=y)[0]
        zcr    = librosa.feature.zero_crossing_rate(y)[0].mean()
        sc     = librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12).mean(axis=1)

        return np.concatenate([
            mfccs.mean(axis=1), mfccs.std(axis=1),
            [rms.mean(), rms.std(), zcr, sc],
            chroma,
        ])
    except Exception as e:
        logger.warning("Falha na extração de features: %s", e)
        return None


def extract_prosody_features(audio_path: str) -> dict:
    """
    Extrai características prosódicas interpretáveis (pitch, energia, ZCR, MFCCs).
    Retorna dict para uso nas heurísticas e logs.
    """
    y, sr = librosa.load(audio_path, sr=None)

    f0, _, _ = librosa.pyin(y, fmin=50, fmax=400, sr=sr)
    f0_clean  = f0[~np.isnan(f0)]
    pitch_mean = float(np.mean(f0_clean)) if len(f0_clean) > 0 else 0.0
    pitch_std  = float(np.std(f0_clean))  if len(f0_clean) > 0 else 0.0

    rms = librosa.feature.rms(y=y)[0]
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    return {
        "pitch_mean_hz"    : pitch_mean,
        "pitch_std_hz"     : pitch_std,
        "energy_mean"      : float(np.mean(rms)),
        "energy_std"       : float(np.std(rms)),
        "speech_rate_proxy": float(librosa.feature.zero_crossing_rate(y)[0].mean()),
        "mfcc_means"       : mfccs.mean(axis=1).tolist(),
    }


def classify_emotion_heuristic(features: dict) -> dict:
    """
    Classificação heurística baseada em prosódia (fallback sem modelo treinado).
    Heurísticas calibradas com base nos padrões do RAVDESS/CREMA-D:
    - Pitch baixo + energia baixa → tristeza/depressão
    - Pitch alto + energia alta  → ansiedade/medo
    - Pitch variável + energia alta → raiva
    """
    p_mean = features["pitch_mean_hz"]
    p_std  = features["pitch_std_hz"]
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
        "predicted_emotion" : emotion,
        "alert"             : emotion in EMOTIONS_OF_CONCERN,
        "method"            : "heuristic",
        "features"          : features,
    }


def analyze_audio_emotion(audio_path: str, model_path: str | Path | None = None) -> dict:
    """
    Pipeline completo de análise emocional de áudio.

    1. Tenta usar o modelo sklearn treinado no Notebook 02 (binary classifier).
    2. Se modelo não disponível, usa heurísticas de prosódia.

    Args:
        audio_path : Caminho para o arquivo de áudio (.wav, .mp3).
        model_path : Caminho alternativo para o pkl. Usa padrão se None.

    Returns:
        Dict com predicted_emotion, alert, probability, method.
    """
    model_data = _load_model(model_path)

    if model_data is not None:
        # Usar modelo treinado (Notebook 02)
        features_vec = extract_features_vector(audio_path)
        if features_vec is not None:
            clf  = model_data["model"]
            pred = int(clf.predict([features_vec])[0])
            prob = float(clf.predict_proba([features_vec])[0][1]) if hasattr(clf, "predict_proba") else 0.5

            return {
                "predicted_emotion" : "risco" if pred == 1 else "normal",
                "alert"             : pred == 1,
                "probability"       : round(prob, 3),
                "method"            : "sklearn_binary",
            }

    # Fallback: heurísticas
    logger.debug("Usando heurísticas de prosódia para %s", Path(audio_path).name)
    features = extract_prosody_features(audio_path)
    return classify_emotion_heuristic(features)
