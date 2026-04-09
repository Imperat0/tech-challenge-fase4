# Relatório Técnico — Tech Challenge Fase 4
**PosTech FIAP | IA para Devs**  
**Tema:** Sistema Multimodal de Monitoramento Contínuo para Saúde da Mulher

---

## 1. Introdução

### 1.1 Contexto e Motivação
Com a IA integrada aos processos médicos especializados em saúde da mulher, este sistema
monitora continuamente pacientes por meio de dados multimodais (áudio, vídeo e texto)
para identificar sinais precoces de risco específicos da saúde e segurança feminina.

### 1.2 Objetivos de Análise Escolhidos
1. **Detectar precocemente riscos em saúde materna e ginecológica**
2. **Identificar sinais de violência doméstica ou abuso**
3. **Utilizar serviços em nuvem (Azure) para ampliar capacidade de processamento**

### 1.3 Funcionalidades de Processamento Implementadas
- Analisar vídeos de consultas e cirurgias ginecológicas
- Processar gravações de voz, detectando sintomas relacionados à fala
- Detectar anomalias em sinais vitais e evolução clínica
- Integrar com Azure Cognitive Services

---

## 2. Arquitetura do Sistema

### 2.1 Fluxo Multimodal

```
Entrada Vídeo  →  [YOLOv8] → Detecção instrumentos cirúrgicos
               →  [DeepFace] → Análise emoções faciais          ─┐
               →  [MediaPipe] → Postura / linguagem corporal      │
                                                                   │
Entrada Áudio  →  [Whisper/SR] → Transcrição                      ├─→ Fusão → Score de Risco
               →  [Classificador] → Emoção vocal (RAVDESS)        │         → GPT-4o
                                                                   │         → Relatório JSON
Entrada Texto  →  [GPT-4o] → Análise laudo maternal               │         → Alerta Equipe
               →  [Azure Language] → NLP / Sentiment             ─┘
```

### 2.2 Tecnologias Utilizadas

| Componente | Tecnologia | Referência de Aula |
|---|---|---|
| Reconhecimento facial | `face_recognition`, `DeepFace` | Aula 01-02 |
| Detecção de pose | `MediaPipe` | Aula 03 |
| Detecção de objetos | `YOLOv8` (`ultralytics`) | Aula 03 |
| Transcrição de áudio | `SpeechRecognition`, `MoviePy` | Aula 04 |
| Classificação de emoção | `scikit-learn`, `librosa` | Aula 05 |
| Análise de texto | `GPT-4o`, Azure Language | Aulas OpenAI |
| Serviços de nuvem | Azure Speech, Azure Language | Matéria 2 |
| Orquestração | Python pipeline (`src/pipeline/`) | — |

---

## 3. Datasets Utilizados

### 3.1 Maternal Health Risk Dataset
- **Fonte:** Kaggle (`mariaadyas/maternal-health-risk-data`)
- **Tamanho:** 1.014 registros clínicos de gestantes
- **Features:** Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate, RiskLevel
- **Aplicação:** EDA e extração de limiares clínicos para o sistema de alertas

**Resultados EDA (Notebook 01):**
- Pressão Sistólica: variável com maior correlação com risco (r = ___)
- Random Forest baseline: acurácia = ___, F1 = ___
- Limiares exportados: PA ≥ 140 mmHg, BS ≥ 7.8 mmol/L, FC > 100 bpm

### 3.2 RAVDESS + CREMA-D (Emoção Vocal)
- **Fonte:** Zenodo (RAVDESS) + Kaggle (CREMA-D)
- **Tamanho:** 2.880 + 7.442 arquivos WAV = 10.322 total
- **Classes:** ANG, DIS, FEA, HAP, NEU, SAD
- **Aplicação:** Treinamento do classificador binário de risco vocal

**Resultados Treinamento (Notebook 02):**
- Melhor modelo: ___ (SVM / Random Forest / Naive Bayes)
- F1-weighted: ___
- AUC-ROC binário (risco vs sem risco): ___
- Recall de `fearful`: ___ | `sad`: ___

### 3.3 m2caiSeg (Instrumentos Cirúrgicos)
- **Fonte:** Dataset de laparoscopia com máscaras de segmentação
- **Tamanho:** 245 imagens de treino, 62 de teste
- **Conversão:** máscaras PNG → labels YOLO (.txt) — script `convert_m2cai_to_yolo.py`
- **Aplicação:** Fine-tuning YOLOv8 para detecção de instrumentos

**Resultados Fine-tuning YOLOv8 (Notebook 03):**
- mAP@50: ___
- Precision: ___ | Recall: ___
- Épocas até convergência: ___

---

## 4. Modelos Especializados

### 4.1 YOLOv8 — Detecção de Instrumentos Cirúrgicos
- **Modelo base:** `yolov8n.pt` (pré-treinado COCO, transfer learning)
- **Classe detectada:** `surgical_instrument` (1 classe)
- **Augmentation:** mosaic, flipud, fliplr, HSV
- **Localização:** `models/yolov8/weights/best.pt`

### 4.2 Classificador de Emoção Vocal
- **Features:** 42 dimensões (MFCCs × 2 + RMS + ZCR + Centroid + Chroma)
- **Modelo:** SVM RBF com `class_weight='balanced'`
- **Tarefa:** Binária (risco / sem risco)
- **Localização:** `data/datasets/violence_audio/emotion_classifier_binary.pkl`

### 4.3 Pipeline de Texto (GPT-4o)
- **Prompts especializados:** saúde materna + indicadores de violência
- **Output:** JSON estruturado com `risk_level`, `findings`, `recommendations`
- **Fallback:** análise por palavras-chave clínicas (sem API key)

---

## 5. Resultados e Exemplos de Anomalias Detectadas

### 5.1 Caso Demonstração — Notebook 04

**Entrada:**
- Vídeo: `facial_recognition_activities.mp4`
- Áudio: `consulta_sample.mp3`
- Texto: Laudo simulado com pré-eclâmpsia + sinais de violência

**Resultados do Pipeline:**

| Modalidade | Achado | Alerta |
|---|---|---|
| Vídeo (DeepFace) | ___ frames com emoção de risco | ___ |
| Vídeo (MediaPipe) | ___ posturas defensivas | ___ |
| Áudio | Emoção vocal: ___ (prob: ___) | ___ |
| Texto (GPT-4o) | Risco: critical — PA 158/102, RCIU | ✅ |
| **Global** | **Score: _/5 — Risco: ___** | **___** |

### 5.2 Anomalias Salvas
- `reports/anomalies_examples/dashboard_demo.png`

---

## 6. Integração Azure Cognitive Services

### 6.1 Azure Speech-to-Text
- Transcrição contínua de consultas médicas em português
- Identificação de hesitações e pausas longas
- Implementação: `src/audio/azure_speech.py`

### 6.2 Azure Language (Text Analytics for Health)
- Extração de entidades de saúde (diagnósticos, medicamentos, sintomas)
- Análise de sentimento de transcrições
- Implementação: `src/text/azure_language.py`

### 6.3 Fluxo de Alerta
```
Anomalia detectada → build_report() → score ≥ high
→ send_alert() → webhook / email → equipe médica
```

---

## 7. Estrutura do Repositório

```
tech-challenge-fase4/
├── src/
│   ├── video/          # DeepFace, MediaPipe, YOLOv8, fusão
│   ├── audio/          # Whisper, Azure Speech, classificador vocal
│   ├── text/           # GPT-4o, Azure Language, gerador de relatórios
│   ├── cloud/          # Sistema de alertas
│   └── pipeline/       # Orquestrador multimodal
├── notebooks/          # 01 EDA Maternal | 02 Emoção Vocal | 03 YOLO | 04 Pipeline
├── data/datasets/      # maternal_risk, violence_audio, surgical_tools
├── models/yolov8/      # Pesos treinados
├── tests/              # 13 testes (todos passando)
└── reports/            # Relatórios JSON + exemplos de anomalias
```

---

## 8. Conclusão

_(preencher após executar os notebooks e obter os resultados finais)_

O sistema demonstra capacidade de:
1. Detectar precocemente indicadores de risco gestacional via análise multimodal
2. Identificar padrões vocais e posturais associados a situações de vulnerabilidade
3. Integrar serviços Azure para processamento escalável e seguro
4. Gerar alertas automáticos para a equipe médica com relatório JSON estruturado

---

## Referências

- Ahmed, M. et al. (2020). IoT-based Risk Assessment for Maternal Health.
- RAVDESS: Livingstone, S.R. & Russo, F.A. (2018). Zenodo.
- CREMA-D: Cao, H. et al. (2014). IEEE TAFFC.
- m2caiSeg: MICCAI Workflow Recognition Challenge 2016.
- Jocher, G. et al. (2023). Ultralytics YOLOv8.
- OpenAI. (2024). GPT-4o Technical Report.
