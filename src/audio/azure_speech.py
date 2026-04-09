"""
Transcrição e análise de fala via Azure Speech-to-Text.

Objetivo coberto:
- Utilizar serviços em nuvem para ampliar capacidade de processamento
"""

import os
import azure.cognitiveservices.speech as speechsdk


def get_speech_config() -> speechsdk.SpeechConfig:
    key = os.environ["AZURE_SPEECH_KEY"]
    region = os.environ["AZURE_SPEECH_REGION"]
    config = speechsdk.SpeechConfig(subscription=key, region=region)
    config.speech_recognition_language = "pt-BR"
    return config


def transcribe_audio_azure(audio_path: str) -> dict:
    """
    Transcreve arquivo de áudio usando Azure Speech-to-Text.

    Returns:
        Dict com text, reason e detalhes de confiança.
    """
    speech_config = get_speech_config()
    audio_config = speechsdk.audio.AudioConfig(filename=audio_path)
    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config,
    )

    result = recognizer.recognize_once_async().get()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return {"text": result.text, "status": "success", "confidence": None}
    elif result.reason == speechsdk.ResultReason.NoMatch:
        return {"text": "", "status": "no_match", "confidence": None}
    else:
        cancellation = result.cancellation_details
        return {
            "text": "",
            "status": "error",
            "error": cancellation.error_details,
        }


def transcribe_continuous_azure(audio_path: str) -> list[dict]:
    """
    Transcrição contínua com timestamps por segmento.
    Ideal para consultas longas.

    Returns:
        Lista de segmentos com offset, duration e text.
    """
    speech_config = get_speech_config()
    speech_config.request_word_level_timestamps()
    audio_config = speechsdk.audio.AudioConfig(filename=audio_path)
    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config,
    )

    segments = []
    done = False

    def on_recognized(evt):
        segments.append({
            "text": evt.result.text,
            "offset_s": evt.result.offset / 10_000_000,
            "duration_s": evt.result.duration / 10_000_000,
        })

    def on_session_stopped(evt):
        nonlocal done
        done = True

    recognizer.recognized.connect(on_recognized)
    recognizer.session_stopped.connect(on_session_stopped)
    recognizer.start_continuous_recognition()

    import time
    while not done:
        time.sleep(0.5)

    recognizer.stop_continuous_recognition()
    return segments
