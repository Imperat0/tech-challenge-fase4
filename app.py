import logging
import traceback
import gradio as gr
from src.utils.logger import setup_logger, log_exception

# Setup logger centralizado
logger = setup_logger(__name__)

# ── VERIFICAÇÃO DE DEPENDÊNCIAS ─────────────────────────────────────────
logger.info("=" * 70)
logger.info("📦 VERIFICANDO DEPENDÊNCIAS")

def check_dependency(name: str, import_statement: str = None) -> bool:
    """Verifica se uma dependência está disponível."""
    try:
        import_stmt = import_statement or f"import {name}"
        exec(import_stmt)
        logger.info(f"✅ {name:30} → Disponível")
        return True
    except ImportError as e:
        logger.warning(f"⚠️  {name:30} → NÃO DISPONÍVEL: {e}")
        return False
    except Exception as e:
        logger.warning(f"⚠️  {name:30} → ERRO: {e}")
        return False

# Verificar dependências críticas
deps = {
    "cv2": "import cv2",
    "deepface": "import deepface",
    "mediapipe": "import mediapipe",
    "ultralytics": "from ultralytics import YOLO",
    "moviepy": "from moviepy.editor import VideoFileClip",
    "whisper": "import whisper",
    "librosa": "import librosa",
    "tensorflow": "import tensorflow",
    "torch": "import torch",
    "transformers": "from transformers import pipeline",
}

available_deps = {}
for dep_name, import_stmt in deps.items():
    available_deps[dep_name] = check_dependency(dep_name, import_stmt)

logger.info("=" * 70)

# ── IMPORTAR PIPELINE ───────────────────────────────────────────────────
from src.pipeline.orchestrator import run_pipeline

def executar_pipeline(video, texto_clinico, id_paciente):
    """Executa pipeline com logging detalhado de erros."""
    try:
        logger.info("=" * 60)
        logger.info("🚀 INICIANDO PIPELINE - Entrada do usuário")
        
        if video is None:
            logger.warning("⚠️ Vídeo não fornecido")
            return "⚠️ Por favor, selecione um vídeo de teste."
        
        logger.info(f"📹 Vídeo recebido: tipo={type(video)}, tamanho={len(video) if hasattr(video, '__len__') else 'unknown'}")
        
        # CORREÇÃO: Transforma string vazia ou espaços em None para ativar o fallback do Whisper
        texto_filtrado = texto_clinico.strip() if texto_clinico and texto_clinico.strip() else None
        id_filtrado = id_paciente.strip() if id_paciente and id_paciente.strip() else "PAC-HF-TEST"
        
        logger.info(f"📝 Texto clínico: {'fornecido' if texto_filtrado else 'não fornecido'}")
        logger.info(f"👤 ID do paciente: {id_filtrado}")
        
        # Executa o seu orquestrador multimodal
        logger.info("🔄 Chamando orquestrador...")
        resultado = run_pipeline(
            video_path=video,
            clinical_text=texto_filtrado,
            patient_id=id_filtrado
        )
        
        logger.info("✅ Pipeline executado com sucesso")
        logger.info(f"📊 Resultado: {resultado}")
        
        # Formata o retorno de forma legível para o gr.Textbox
        if isinstance(resultado, dict):
            risco = resultado.get("overall_risk", "NÃO DETECTADO").upper()
            atencao = "SIM" if resultado.get("requires_immediate_attention") else "NÃO"
            msg = f"📊 ANÁLISE CONCLUÍDA\n\n🔴 Nível de Risco Geral: {risco}\n🚨 Requer Atenção Imediata? {atencao}"
            logger.info(f"💾 Mensagem retornada: {msg}")
            return msg
        
        logger.info(f"📋 Resultado em formato string: {str(resultado)}")
        return str(resultado)
        
    except FileNotFoundError as e:
        error_msg = log_exception(logger, e, "ARQUIVO_NAO_ENCONTRADO")
        return f"❌ Arquivo não encontrado:\n{error_msg}\n\n📍 Verifique se o vídeo foi carregado corretamente."
    
    except ImportError as e:
        error_msg = log_exception(logger, e, "IMPORTACAO")
        msg = f"❌ Erro de dependência:\n{error_msg}\n\n🔧 Dependências disponíveis:\n"
        msg += "\n".join([f"  {'✅' if v else '❌'} {k}" for k, v in available_deps.items()])
        return msg
    
    except ValueError as e:
        error_msg = log_exception(logger, e, "VALOR_INVALIDO")
        return f"❌ Valor inválido nos parâmetros:\n{error_msg}"
    
    except RuntimeError as e:
        error_msg = log_exception(logger, e, "ERRO_EXECUCAO")
        return f"❌ Erro durante execução (API/Modelo):\n{error_msg}\n\n💡 Possível causa: API Key faltando ou limite de rate reached."
    
    except Exception as e:
        error_msg = log_exception(logger, e, "ERRO_DESCONHECIDO")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        return f"❌ Erro interno durante a execução do pipeline:\n\n{error_msg}\n\n📋 Stack trace está nos logs."

# Monta o design da página web que vai aparecer no Hugging Face
with gr.Blocks(title="Monitoramento Multimodal") as demo:
    gr.Markdown("# 🎙️ Sistema Multimodal de Monitoramento - Saúde da Mulher")
    gr.Markdown("Interface de validação do pipeline do Tech Challenge (Fase 4) - FIAP.")
    
    with gr.Row():
        with gr.Column():
            input_video = gr.Video(label="Vídeo Clínico (Upload)")
            input_text = gr.Textbox(label="Laudo / Texto Clínico (Opcional)", placeholder="Ex: Paciente gestante, 32 semanas... (Deixe em branco para usar o áudio do vídeo)")
            input_id = gr.Textbox(label="ID do Paciente", placeholder="Ex: PAC-001")
            btn = gr.Button("Executar Análise Multimodal", variant="primary")
        
        with gr.Column():
            output_text = gr.Textbox(label="Resultado / Score de Risco", lines=6, interactive=False)
            
    btn.click(fn=executar_pipeline, inputs=[input_video, input_text, input_id], outputs=output_text)

if __name__ == "__main__":
    logger.info("🌐 Iniciando Gradio Interface")
    logger.info(f"LOG_LEVEL configurado: {logger.level}")
    demo.launch()