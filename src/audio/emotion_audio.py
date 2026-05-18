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
import traceback
import os
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
        logger.debug(f"✓ Modelo em cache: {path.name}")
        return _model_cache[key]

    if not path.exists():
        logger.warning(f"⚠️  Modelo não encontrado em {path}. Usando heurísticas.")
        return None

    try:
        logger.info(f"🤖 Carregando modelo de emoção: {path.name}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        _model_cache[key] = data
        logger.info(f"✅ Modelo de emoção vocal carregado: {path.name}")
        return data
    except Exception as e:
        logger.error(f"❌ Falha ao carregar modelo de emoção: {e}")
        logger.debug(traceback.format_exc())
        return None


def extract_features_vector(audio_path: str, duration: float = 3.0) -> np.ndarray | None:
    """
    Extrai vetor de 42 features de prosódia para o classificador sklearn.
    Mesmo pipeline do Notebook 02:
      MFCCs (13 média + 13 desvio) + RMS (média + desvio) + ZCR + Centroid + Chroma(12)
    """
    try:
        logger.debug(f"🎵 Carregando áudio para features: {Path(audio_path).name}")
        y, sr = librosa.load(audio_path, duration=duration, res_type="kaiser_fast")
        logger.debug(f"   Sample rate: {sr}, Amostras: {len(y)}")
        
        if len(y) < sr * 0.5:
            logger.warning(f"   ⚠️  Áudio muito curto ({len(y)} amostras)")
            return None

        mfccs  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        rms    = librosa.feature.rms(y=y)[0]
        zcr    = librosa.feature.zero_crossing_rate(y)[0].mean()
        sc     = librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12).mean(axis=1)

        features = np.concatenate([
            mfccs.mean(axis=1), mfccs.std(axis=1),
            [rms.mean(), rms.std(), zcr, sc],
            chroma,
        ])
        logger.debug(f"   ✓ Features extraídas: {len(features)} dimensões")
        return features
    except Exception as e:
        logger.error(f"❌ Falha na extração de features: {e}")
        logger.debug(traceback.format_exc())
        return None


def extract_prosody_features(audio_path: str) -> dict:
    """
    Extrai características prosódicas interpretáveis (pitch, energia, ZCR, MFCCs).
    Retorna dict para uso nas heurísticas e logs.
    """
    try:
        logger.debug(f"🎙️  Extraindo prosódia: {Path(audio_path).name}")
        y, sr = librosa.load(audio_path, sr=None)
        
        f0, _, _ = librosa.pyin(y, fmin=50, fmax=400, sr=sr)
        f0_clean  = f0[~np.isnan(f0)]
        pitch_mean = float(np.mean(f0_clean)) if len(f0_clean) > 0 else 0.0
        pitch_std  = float(np.std(f0_clean))  if len(f0_clean) > 0 else 0.0

        rms = librosa.feature.rms(y=y)[0]
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        prosody = {
            "pitch_mean_hz"    : pitch_mean,
            "pitch_std_hz"     : pitch_std,
            "energy_mean"      : float(np.mean(rms)),
            "energy_std"       : float(np.std(rms)),
            "speech_rate_proxy": float(librosa.feature.zero_crossing_rate(y)[0].mean()),
            "mfcc_means"       : mfccs.mean(axis=1).tolist(),
        }
        logger.debug(f"   ✓ Pitch: {pitch_mean:.1f}±{pitch_std:.1f} Hz, Energia: {prosody['energy_mean']:.3f}")
        return prosody
    except Exception as e:
        logger.error(f"❌ Erro ao extrair prosódia: {e}")
        logger.debug(traceback.format_exc())
        return {}


def classify_emotion_heuristic(features: dict) -> dict:
    """
    Classificação heurística baseada em prosódia (fallback sem modelo treinado).
    Heurísticas calibradas com base nos padrões do RAVDESS/CREMA-D:
    - Pitch baixo + energia baixa → tristeza/depressão
    - Pitch alto + energia alta  → ansiedade/medo
    - Pitch variável + energia alta → raiva
    """
    try:
        p_mean = features.get("pitch_mean_hz", 150)
        p_std  = features.get("pitch_std_hz", 0)
        e_mean = features.get("energy_mean", 0.05)

        if p_mean < 150 and e_mean < 0.05:
            emotion = "sad"
            logger.debug(f"   🟡 Heurística: DEPRESSÃO (pitch={p_mean:.1f}, energia={e_mean:.3f})")
        elif p_mean > 250 and e_mean > 0.1:
            emotion = "fearful"
            logger.debug(f"   🔴 Heurística: ANSIEDADE (pitch={p_mean:.1f}, energia={e_mean:.3f})")
        elif p_std > 50 and e_mean > 0.08:
            emotion = "angry"
            logger.debug(f"   🟠 Heurística: RAIVA (pitch_std={p_std:.1f}, energia={e_mean:.3f})")
        else:
            emotion = "neutral"
            logger.debug(f"   🟢 Heurística: NORMAL (pitch={p_mean:.1f}, energia={e_mean:.3f})")

        return {
            "predicted_emotion" : emotion,
            "alert"             : emotion in EMOTIONS_OF_CONCERN,
            "method"            : "heuristic",
            "features"          : features,
        }
    except Exception as e:
        logger.error(f"❌ Erro na classificação heurística: {e}")
        return {
            "predicted_emotion": "neutral",
            "alert": False,
            "method": "heuristic_error",
            "error": str(e)
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
    try:
        logger.info(f"🎵 Analisando emoção no áudio: {Path(audio_path).name}")
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Áudio não encontrado: {audio_path}")
        
        file_size = os.path.getsize(audio_path) / 1024
        logger.debug(f"   Tamanho: {file_size:.1f} KB")
        
        model_data = _load_model(model_path)

        if model_data is not None:
            # Usar modelo treinado (Notebook 02)
            logger.info("   → Usando modelo sklearn (binary classifier)")
            features_vec = extract_features_vector(audio_path)
            if features_vec is not None:
                clf  = model_data["model"]
                pred = int(clf.predict([features_vec])[0])
                prob = float(clf.predict_proba([features_vec])[0][1]) if hasattr(clf, "predict_proba") else 0.5

                result = {
                    "predicted_emotion" : "risco" if pred == 1 else "normal",
                    "alert"             : pred == 1,
                    "probability"       : round(prob, 3),
                    "method"            : "sklearn_binary",
                }
                logger.info(f"   ✅ Resultado: {result['predicted_emotion'].upper()} (confiança: {prob:.1%})")
                return result
            else:
                logger.warning("   ⚠️  Falha ao extrair features. Usando fallback heurístico.")

        # Fallback: heurísticas
        logger.info("   → Usando heurísticas de prosódia (fallback)")
        features = extract_prosody_features(audio_path)
        result = classify_emotion_heuristic(features)
        logger.info(f"   ✅ Resultado (heurística): {result['predicted_emotion'].upper()}")
        return result
    
    except FileNotFoundError as e:
        logger.error(f"❌ [AUDIO_EMOTION] Arquivo não encontrado: {e}")
        return {
            "predicted_emotion": "neutral",
            "alert": False,
            "method": "error",
            "error": str(e)
        }
    except Exception as e:
        logger.error(f"❌ [AUDIO_EMOTION] Erro na análise emocional: {e}")
        logger.debug(traceback.format_exc())
        return {
            "predicted_emotion": "neutral",
            "alert": False,
            "method": "error",
            "error": str(e)
        }
