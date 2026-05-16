import gradio as gr
from src.pipeline.orchestrator import run_pipeline

def executar_pipeline(video, texto_clinico, id_paciente):
    if video is None:
        return "Por favor, selecione um vídeo de teste."
    
    # Executa a lógica que você desenvolveu na Fase 4
    resultado = run_pipeline(
        video_path=video,
        clinical_text=texto_clinico,
        patient_id=id_paciente if id_paciente else "PAC-HF-TEST"
    )
    
    # Retorna o resultado formatado para exibir na tela do Hugging Face
    return str(resultado.get("overall_risk", resultado))

# Monta o design da página web que vai aparecer no Hugging Face
with gr.Blocks() as demo:
    gr.Markdown("# 🎙️ Sistema Multimodal de Monitoramento - Saúde da Mulher")
    gr.Markdown("Interface de validação do pipeline do Tech Challenge (Fase 4) - FIAP.")
    
    with gr.Row():
        with gr.Column():
            input_video = gr.Video(label="Vídeo Clínico (Upload)")
            input_text = gr.Textbox(label="Laudo / Texto Clínico", placeholder="Ex: Paciente gestante, 32 semanas...")
            input_id = gr.Textbox(label="ID do Paciente", placeholder="Ex: PAC-001")
            btn = gr.Button("Executar Análise Multimodal")
        
        with gr.Column():
            output_text = gr.Textbox(label="Resultado / Score de Risco", interactive=False)
            
    btn.click(fn=executar_pipeline, inputs=[input_video, input_text, input_id], outputs=output_text)

if __name__ == "__main__":
    demo.launch()