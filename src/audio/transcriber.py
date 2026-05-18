"""
Transcrição automática de áudio usando Whisper e SpeechRecognition.
Aula 04 - Transcrição automática de áudio e conversão de fala em texto.
"""

import os
import logging
import traceback
from pathlib import Path

import whisper
import speech_recognition as sr

logger = logging.getLogger(__name__)


def transcribe_with_whisper(audio_path: str, model_size: str = "base") -> dict:
    """
    Transcreve áudio usando OpenAI Whisper (local, sem API key).

    Args:
        audio_path: Caminho para o arquivo de áudio (.mp3, .wav, .m4a).
        model_size: Tamanho do modelo Whisper (tiny, base, small, medium, large).

    Returns:
        Dict com text, language e segments com timestamps.
        
    Raises:
        FileNotFoundError: Se arquivo de áudio não existe
        RuntimeError: Se Whisper falhar na transcrição
    """
    try:
        logger.info(f"🎵 Carregando áudio: {audio_path}")
        
        # Valida arquivo
        if not os.path.exists(audio_path):
            logger.error(f"❌ Arquivo não encontrado: {audio_path}")
            raise FileNotFoundError(f"Áudio não encontrado: {audio_path}")
        
        file_size = os.path.getsize(audio_path)
        logger.info(f"📊 Tamanho do arquivo: {file_size / (1024*1024):.2f} MB")
        
        # Garanta que o modelo escolhido seja leve para rodar na CPU gratuita
        logger.info(f"🤖 Carregando modelo Whisper ({model_size})...")
        model = whisper.load_model("tiny")
        logger.info("✅ Modelo Whisper carregado")
        
        logger.info("🔄 Iniciando transcrição...")
        result = model.transcribe(audio_path, language="pt")
        logger.info("✅ Transcrição concluída")
        
        transcript_text = result["text"].strip()
        language = result.get("language", "pt")
        
        logger.info(f"🌐 Idioma detectado: {language}")
        logger.info(f"📝 Tamanho do texto: {len(transcript_text)} caracteres")
        logger.debug(f"Texto transcrito (primeiros 200 chars): {transcript_text[:200]}...")
        
        return {
            "text": transcript_text,
            "language": language,
            "segments": [
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"].strip(),
                }
                for seg in result.get("segments", [])
            ],
        }
    
    except FileNotFoundError as e:
        logger.error(f"❌ [WHISPER] Arquivo não encontrado: {e}")
        raise
    except RuntimeError as e:
        logger.error(f"❌ [WHISPER] Erro runtime: {e}")
        logger.debug(traceback.format_exc())
        raise RuntimeError(f"Falha no Whisper: {str(e)}") from e
    except Exception as e:
        logger.error(f"❌ [WHISPER] Erro inesperado: {e}")
        logger.debug(traceback.format_exc())
        raise RuntimeError(f"Erro ao transcrever áudio: {str(e)}") from e


def transcribe_with_speech_recognition(audio_path: str) -> str:
    """
    Transcreve áudio usando Google Speech Recognition (requer internet).
    Fallback quando Whisper não está disponível.
    """
    try:
        logger.info(f"🎤 Iniciando Google Speech Recognition: {audio_path}")
        
        if not os.path.exists(audio_path):
            logger.error(f"❌ Arquivo não encontrado: {audio_path}")
            raise FileNotFoundError(f"Áudio não encontrado: {audio_path}")
        
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            logger.info("📖 Lendo arquivo de áudio...")
            audio = recognizer.record(source)
        
        logger.info("🌐 Enviando para Google Cloud Speech...")
        result = recognizer.recognize_google(audio, language="pt-BR")
        logger.info(f"✅ Transcrição Google concluída: {len(result)} caracteres")
        return result
    
    except sr.UnknownValueError as e:
        logger.error(f"❌ [Google] Áudio não compreendido: {e}")
        return ""
    except sr.RequestError as e:
        logger.error(f"❌ [Google] Erro na API: {e}")
        raise RuntimeError(f"Erro na API Google Speech: {e}") from e
    except Exception as e:
        logger.error(f"❌ [Google] Erro inesperado: {e}")
        logger.debug(traceback.format_exc())
        raise


def extract_audio_from_video(video_path: str, output_dir: str | None = None) -> str:
    """
    Extrai trilha de áudio de um vídeo usando MoviePy.

    Returns:
        Caminho para o arquivo .wav gerado.
        
    Raises:
        FileNotFoundError: Se vídeo não existe
        RuntimeError: Se falhar ao extrair áudio
    """
    try:
        logger.info(f"🎬 Extraindo áudio do vídeo: {video_path}")
        
        if not os.path.exists(video_path):
            logger.error(f"❌ Vídeo não encontrado: {video_path}")
            raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")
        
        from moviepy.editor import VideoFileClip
        
        logger.info("🔄 Carregando vídeo...")
        video = VideoFileClip(video_path)
        video_duration = video.duration
        logger.info(f"📊 Duração do vídeo: {video_duration:.2f} segundos")
        
        out_dir = Path(output_dir) if output_dir else Path(video_path).parent
        output_path = out_dir / (Path(video_path).stem + "_audio.wav")
        
        logger.info(f"💾 Salvando áudio em: {output_path}")
        video.audio.write_audiofile(str(output_path), verbose=False, logger=None)
        video.close()
        
        logger.info(f"✅ Áudio extraído com sucesso")
        return str(output_path)
    
    except FileNotFoundError as e:
        logger.error(f"❌ [MoviePy] Arquivo não encontrado: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ [MoviePy] Erro ao extrair áudio: {e}")
        logger.debug(traceback.format_exc())
        raise RuntimeError(f"Falha ao extrair áudio do vídeo: {str(e)}") from e


def save_transcript(transcript: dict, output_path: str) -> None:
    """Salva transcrição em arquivo .json com logging."""
    try:
        import json
        logger.info(f"💾 Salvando transcrição em: {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(transcript, f, ensure_ascii=False, indent=2)
        logger.info("✅ Transcrição salva com sucesso")
    except Exception as e:
        logger.error(f"❌ Erro ao salvar transcrição: {e}")
        raise
