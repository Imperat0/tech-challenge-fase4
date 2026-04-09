"""
Análise de laudos e relatórios médicos via GPT-4o.
Aulas OpenAI - Integração com API da OpenAI.

Objetivos cobertos:
- Detectar precocemente riscos em saúde materna (análise de laudos)
- Geração de relatórios automáticos especializados
"""

import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

SYSTEM_PROMPT_MATERNAL = """
Você é um assistente especializado em saúde materna e ginecológica.
Analise o texto clínico fornecido e identifique:
1. Indicadores de risco gestacional (pressão arterial, glicemia, edema)
2. Sinais de depressão pós-parto ou ansiedade
3. Anomalias em exames ou prescrições hormonais
4. Qualquer sinal que requeira atenção imediata da equipe médica

Responda em JSON com os campos: risk_level (low/medium/high/critical),
findings (lista de achados), recommendations (lista de recomendações), alert (bool).
"""

SYSTEM_PROMPT_VIOLENCE = """
Você é um assistente especializado em identificação de sinais de violência doméstica
em contexto clínico. Analise a transcrição de consulta fornecida e identifique:
1. Inconsistências na narrativa do paciente
2. Hesitações ou contradições ao descrever lesões
3. Padrões de linguagem associados a trauma ou medo
4. Qualquer sinal verbal indicativo de abuso

Responda em JSON com os campos: violence_indicators (lista), confidence (0-1),
alert (bool), recommended_action (string).
"""


def analyze_medical_text(text: str, analysis_type: str = "maternal") -> dict:
    """
    Analisa texto clínico usando GPT-4o.

    Args:
        text: Laudo, prescrição ou transcrição de consulta.
        analysis_type: "maternal" ou "violence".

    Returns:
        Dict com análise estruturada em JSON.
    """
    system_prompt = (
        SYSTEM_PROMPT_MATERNAL if analysis_type == "maternal" else SYSTEM_PROMPT_VIOLENCE
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    import json
    return json.loads(response.choices[0].message.content)


def generate_clinical_report(multimodal_results: dict) -> str:
    """
    Gera relatório clínico consolidado a partir dos resultados multimodais.

    Args:
        multimodal_results: Dict com resultados de vídeo, áudio e texto.

    Returns:
        Relatório em texto formatado.
    """
    prompt = f"""
    Com base nos seguintes dados de monitoramento multimodal de uma paciente:

    ANÁLISE DE VÍDEO:
    {multimodal_results.get('video', 'Não disponível')}

    ANÁLISE DE ÁUDIO:
    {multimodal_results.get('audio', 'Não disponível')}

    ANÁLISE DE TEXTO CLÍNICO:
    {multimodal_results.get('text', 'Não disponível')}

    Gere um relatório clínico conciso (máximo 300 palavras) destacando:
    1. Principais achados
    2. Nível de risco geral
    3. Ações recomendadas para a equipe médica
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Você é um assistente clínico especializado em saúde da mulher."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content
