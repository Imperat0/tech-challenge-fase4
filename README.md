# Tech Challenge — Fase 4
**PosTech FIAP | IA para Devs**

Sistema multimodal de monitoramento contínuo para saúde da mulher, integrando análise de vídeo, áudio e texto com serviços Azure Cognitive Services.

---

## Objetivos de Análise

| # | Objetivo | Modalidades |
|---|---|---|
| 1 | Detectar precocemente riscos em saúde materna e ginecológica | Texto (laudos) + Áudio (consultas) + Vídeo (cirurgias) |
| 2 | Identificar sinais de violência doméstica ou abuso | Vídeo (pose + emoção) + Áudio (prosódia) + Texto (transcrição) |
| 3 | Utilizar serviços em nuvem (Azure) para processamento especializado | Azure Speech + Azure Language + GPT-4o |

## Funcionalidades de Processamento

- **Análise de Vídeo**: YOLOv8 (instrumentos cirúrgicos) + DeepFace (emoção facial) + MediaPipe (pose/linguagem corporal)
- **Processamento de Áudio**: Whisper / Azure Speech-to-Text + análise de prosódia (pitch, energia, ritmo)
- **Detecção de Anomalias**: Score de risco multimodal com alerta em tempo real para equipe médica

---

## Estrutura do Projeto

```
tech-challenge-fase4/
├── data/
│   ├── videos/          # Vídeos clínicos (raw / processed / samples)
│   ├── audios/          # Gravações e transcrições
│   ├── datasets/        # Maternal Risk, RAVDESS, EndoVis
│   └── annotations/     # Labels YOLO
├── models/yolov8/       # Pesos e configs do modelo customizado
├── src/
│   ├── video/           # face_emotion, pose_detection, yolo_detector, violence_screening
│   ├── audio/           # transcriber, azure_speech, emotion_audio
│   ├── text/            # gpt_analysis, azure_language, report_generator
│   ├── cloud/           # alerts
│   └── pipeline/        # orchestrator (ponto de entrada principal)
├── notebooks/           # EDA e experimentos
├── reports/             # Relatórios gerados automaticamente
└── tests/               # Testes unitários e de integração
```

---

## Instalação

```bash
# 1. Clonar o repositório
git clone <repo-url>
cd tech-challenge-fase4

# 2. Criar ambiente virtual
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Configurar variáveis de ambiente
cp .env.example .env
# Edite o .env com suas chaves Azure e OpenAI
```

---

## Uso

```bash
# Pipeline completo com vídeo
python -m src.pipeline.orchestrator data/videos/samples/consulta.mp4

# Ou via Python
from src.pipeline.orchestrator import run_pipeline

report = run_pipeline(
    video_path="data/videos/samples/consulta.mp4",
    clinical_text="Paciente gestante, 32 semanas, PA 140/90...",
    patient_id="PAC-001",
)
print(report["overall_risk"])
```

---

## Datasets Utilizados

| Dataset | Objetivo | Link |
|---|---|---|
| Maternal Health Risk | Saúde materna | Kaggle: `mariaadyas/maternal-health-risk-data` |
| RAVDESS | Emoção vocal | Zenodo (gratuito) |
| CREMA-D | Emoção vocal | Kaggle / GitHub |
| EndoVis 2017 | YOLOv8 cirúrgico | MICCAI Challenge |
| RWF-2000 | Triagem violência | GitHub |

---

## Testes

```bash
pytest tests/ -v
```

---

## Entregáveis

- [x] Repositório Git com código-fonte completo
- [ ] Relatório técnico (`reports/relatorio_tecnico.pdf`)
- [ ] Vídeo demonstração (YouTube/Vimeo — até 15 min)
