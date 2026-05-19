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
    """Executa pipeline com logging detalhado de erros e múltiplos retornos para a UI."""
    try:
        logger.info("=" * 60)
        logger.info("🚀 INICIANDO PIPELINE - Entrada do usuário")
        
        if video is None:
            logger.warning("⚠️ Vídeo não fornecido")
            erro_html = "<div style='color: #ef4444; padding: 10px; border: 1px solid #ef4444; border-radius: 8px;'>⚠️ Por favor, selecione um vídeo de teste.</div>"
            return erro_html, "", {"status": "erro", "detalhe": "Vídeo não fornecido"}
        
        logger.info(f"📹 Vídeo recebido: tipo={type(video)}, tamanho={len(video) if hasattr(video, '__len__') else 'unknown'}")
        
        # Transforma string vazia ou espaços em None para ativar o fallback do Whisper
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
        
        # 1. Monta um Card HTML para o Risco
        risco = resultado.get("overall_risk", "NÃO DETECTADO") if isinstance(resultado, dict) else "NÃO DETECTADO"
        risco = str(risco).upper()
        atencao = resultado.get("requires_immediate_attention", False) if isinstance(resultado, dict) else False
        
        cor_alerta = "#ef4444" if atencao or risco == "ALTO" else "#eab308" if risco == "MÉDIO" else "#22c55e"
        
        html_score = f"""
        <div style="padding: 20px; border-radius: 8px; background-color: {cor_alerta}20; border: 2px solid {cor_alerta}; text-align: center;">
            <h2 style="margin: 0; color: {cor_alerta};">Risco Geral: {risco}</h2>
            <p style="margin: 5px 0 0 0; font-size: 16px;"><strong>Atenção Imediata:</strong> {'🚨 SIM' if atencao else '✅ NÃO'}</p>
        </div>
        """

        # 2. Monta o Markdown de Detalhes
        if isinstance(resultado, dict):
            detalhes_md = f"""
### 🧠 Análise por Modalidade
* **Análise Facial (Visão):** {resultado.get("facial_analysis", "Dados não disponíveis")}
* **Transcrição/Sentimento (Áudio):** {resultado.get("audio_analysis", "Dados não disponíveis")}
* **Análise Clínica (Texto):** {resultado.get("text_analysis", "Dados não disponíveis")}
            """
        else:
            detalhes_md = f"O pipeline retornou um formato inesperado:\n\n{str(resultado)}"
        
        # Retorna na ordem exata dos outputs definidos no gr.Blocks
        return html_score, detalhes_md, resultado if isinstance(resultado, dict) else {"resultado_bruto": str(resultado)}
        
    except FileNotFoundError as e:
        error_msg = log_exception(logger, e, "ARQUIVO_NAO_ENCONTRADO")
        erro_html = f"<div style='color: #ef4444; padding: 10px; border: 1px solid #ef4444; border-radius: 8px;'>❌ Arquivo não encontrado:<br>{error_msg}<br><br>📍 Verifique se o vídeo foi carregado corretamente.</div>"
        return erro_html, "", {"status": "erro", "detalhe": "FileNotFoundError"}
    
    except ImportError as e:
        error_msg = log_exception(logger, e, "IMPORTACAO")
        deps_html = "<br>".join([f"  {'✅' if v else '❌'} {k}" for k, v in available_deps.items()])
        erro_html = f"<div style='color: #ef4444; padding: 10px; border: 1px solid #ef4444; border-radius: 8px;'>❌ Erro de dependência:<br>{error_msg}<br><br>🔧 Dependências disponíveis:<br>{deps_html}</div>"
        return erro_html, "", {"status": "erro", "detalhe": "ImportError"}
    
    except ValueError as e:
        error_msg = log_exception(logger, e, "VALOR_INVALIDO")
        erro_html = f"<div style='color: #ef4444; padding: 10px; border: 1px solid #ef4444; border-radius: 8px;'>❌ Valor inválido nos parâmetros:<br>{error_msg}</div>"
        return erro_html, "", {"status": "erro", "detalhe": "ValueError"}
    
    except RuntimeError as e:
        error_msg = log_exception(logger, e, "ERRO_EXECUCAO")
        erro_html = f"<div style='color: #ef4444; padding: 10px; border: 1px solid #ef4444; border-radius: 8px;'>❌ Erro durante execução (API/Modelo):<br>{error_msg}<br><br>💡 Possível causa: API Key faltando ou limite de rate limit atingido.</div>"
        return erro_html, "", {"status": "erro", "detalhe": "RuntimeError"}
    
    except Exception as e:
        error_msg = log_exception(logger, e, "ERRO_DESCONHECIDO")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        erro_html = f"<div style='color: #ef4444; padding: 10px; border: 1px solid #ef4444; border-radius: 8px;'>❌ Erro interno durante a execução do pipeline:<br>{error_msg}<br><br>📋 Verifique os logs do terminal para o stack trace completo.</div>"
        return erro_html, "", {"status": "erro", "detalhe": str(e)}

# Monta o design da página web que vai aparecer no Hugging Face
with gr.Blocks(title="Monitoramento Multimodal", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ Sistema Multimodal de Monitoramento - Saúde da Mulher")
    gr.Markdown("Interface de validação do pipeline do Tech Challenge (Fase 4) - FIAP.")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_video = gr.Video(label="Vídeo Clínico (Upload)")
            input_text = gr.Textbox(label="Laudo / Texto Clínico (Opcional)", placeholder="Ex: Paciente gestante, 32 semanas... (Deixe em branco para usar o áudio do vídeo)")
            input_id = gr.Textbox(label="ID do Paciente", placeholder="Ex: PAC-001")
            btn = gr.Button("Executar Análise Multimodal", variant="primary")
        
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Resultado do Score")
            
            # Card principal de risco
            output_score = gr.HTML()
            
            # Acordeão para detalhes amigáveis
            with gr.Accordion("Ver Detalhes da Análise", open=True):
                output_details = gr.Markdown()
                
            # Acordeão para o JSON bruto (ótimo para validação técnica)
            with gr.Accordion("Payload Bruto (JSON)", open=False):
                output_json = gr.JSON()
            
    # Conecta o botão à função, passando os inputs e esperando 3 outputs
    btn.click(
        fn=executar_pipeline, 
        inputs=[input_video, input_text, input_id], 
        outputs=[output_score, output_details, output_json]
    )

if __name__ == "__main__":
    logger.info("🌐 Iniciando Gradio Interface")
    logger.info(f"LOG_LEVEL configurado: {logger.level}")
    demo.launch()