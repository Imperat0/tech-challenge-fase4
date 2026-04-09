"""
Transcrição automática de áudio usando Whisper e SpeechRecognition.
Aula 04 - Transcrição automática de áudio e conversão de fala em texto.
"""

import os
import whisper
import speech_recognition as sr
from pathlib import Path


def transcribe_with_whisper(audio_path: str, model_size: str = "base") -> dict:
    """
    Transcreve áudio usando OpenAI Whisper (local, sem API key).

    Args:
        audio_path: Caminho para o arquivo de áudio (.mp3, .wav, .m4a).
        model_size: Tamanho do modelo Whisper (tiny, base, small, medium, large).

    Returns:
        Dict com text, language e segments com timestamps.
    """
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, language="pt")
    return {
        "text": result["text"].strip(),
        "language": result.get("language", "pt"),
        "segments": [
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip(),
            }
            for seg in result.get("segments", [])
        ],
    }


def transcribe_with_speech_recognition(audio_path: str) -> str:
    """
    Transcreve áudio usando Google Speech Recognition (requer internet).
    Fallback quando Whisper não está disponível.
    """
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio, language="pt-BR")
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        raise RuntimeError(f"Erro na API Google Speech: {e}") from e


def extract_audio_from_video(video_path: str, output_dir: str | None = None) -> str:
    """
    Extrai trilha de áudio de um vídeo usando MoviePy.

    Returns:
        Caminho para o arquivo .wav gerado.
    """
    from moviepy.editor import VideoFileClip

    video = VideoFileClip(video_path)
    out_dir = Path(output_dir) if output_dir else Path(video_path).parent
    output_path = out_dir / (Path(video_path).stem + "_audio.wav")
    video.audio.write_audiofile(str(output_path), verbose=False, logger=None)
    video.close()
    return str(output_path)


def save_transcript(transcript: dict, output_path: str) -> None:
    """Salva transcrição em arquivo .json."""
    import json
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)
