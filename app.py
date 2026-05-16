import gradio as gr
from src.pipeline.orchestrator import run_pipeline

def executar_pipeline(video, texto_clinico, id_paciente):
    if video is None:
        return "⚠️ Por favor, selecione um vídeo de teste."
    
    # CORREÇÃO: Transforma string vazia ou espaços em None para ativar o fallback do Whisper
    texto_filtrado = texto_clinico.strip() if texto_clinico and texto_clinico.strip() else None
    id_filtrado = id_paciente.strip() if id_paciente and id_paciente.strip() else "PAC-HF-TEST"
    
    try:
        # Executa o seu orquestrador multimodal
        resultado = run_pipeline(
            video_path=video,
            clinical_text=texto_filtrado,
            patient_id=id_filtrado
        )
        
        # Formata o retorno de forma legível para o gr.Textbox
        if isinstance(resultado, dict):
            risco = resultado.get("overall_risk", "NÃO DETECTADO").upper()
            atencao = "SIM" if resultado.get("requires_immediate_attention") else "NÃO"
            return f"📊 ANÁLISE CONCLUÍDA\n\n🔴 Nível de Risco Geral: {risco}\n🚨 Requer Atenção Imediata? {atencao}"
        
        return str(resultado)
        
    except Exception as e:
        # Se faltar alguma API Key ou pacote de sistema, o erro aparecerá aqui em vez de travar a tela
        return f"❌ Erro interno durante a execução do pipeline:\n\n{str(e)}"

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
    demo.launch()