import gradio as gr
import os
# Importa o orquestrador principal do seu projeto
from src.pipeline.orchestrator import run_pipeline

def process_multimodal_analysis(video_file, clinical_text, patient_id):
    # Validação simples dos campos obrigatórios
    if not video_file:
        return "⚠️ Por favor, faça o upload de um vídeo para análise."
        
    if not patient_id or not patient_id.strip():
        patient_id = "ANONIMO"

    # CORREÇÃO: Se o texto estiver vazio, envia None para ativar o fallback do Whisper
    text_param = clinical_text.strip() if clinical_text and clinical_text.strip() else None

    try:
        # Executa o pipeline do seu Tech Challenge
        report = run_pipeline(
            video_path=video_file,
            clinical_text=text_param,
            patient_id=patient_id
        )
        
        # Retorna o dicionário diretamente para o gr.JSON
        return report
        
    except Exception as e:
        return {"erro": f"Falha ao processar o pipeline: {str(e)}"}

# Construção da Interface Gradio utilizando Blocks para melhor estilização
with gr.Blocks(title="Tech Challenge Fase 4 — Monitoramento Multimodal") as demo:
    gr.Markdown("# 🎙️ Sistema Multimodal de Monitoramento — Saúde da Mulher")
    gr.Markdown("Triagem integrada de vídeo (expressão/pose), áudio (prosódia) e texto clínico via IA e Azure.")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📥 Dados de Entrada")
            input_video = gr.Video(label="Upload do Vídeo Clínico/Consulta")
            input_text = gr.Textbox(label="Laudo Clínico / Notas de Texto (Opcional)", lines=4, placeholder="Digite as observações ou deixe vazio para usar a transcrição do áudio...")
            input_id = gr.Textbox(label="ID da Paciente", placeholder="Ex: PAC-001")
            
            btn_submit = gr.Button("🚀 Iniciar Análise Multimodal", variant="primary")
            
        with gr.Column():
            gr.Markdown("### 📊 Relatório e Score de Risco")
            output_report = gr.JSON(label="Resultado do Score de Risco")

    # Mapeamento do clique do botão para a função de processamento
    btn_submit.click(
        fn=process_multimodal_analysis,
        inputs=[input_video, input_text, input_id],
        outputs=output_report
    )

# Inicializa o app exposto na porta padrão (7860) exigida pelo Hugging Face Spaces
if __name__ == "__main__":
    demo.launch()