"""
Análise de linguagem natural via Azure Language Service.

Objetivo coberto:
- Utilizar serviços em nuvem para ampliar capacidade de processamento
- Análise de sentimento e extração de entidades clínicas
"""

import os
from azure.ai.language.conversations import ConversationAnalysisClient
from azure.core.credentials import AzureKeyCredential


def get_language_client():
    endpoint = os.environ["AZURE_LANGUAGE_ENDPOINT"]
    key = os.environ["AZURE_LANGUAGE_KEY"]
    return ConversationAnalysisClient(endpoint, AzureKeyCredential(key))


def analyze_sentiment_azure(text: str) -> dict:
    """
    Analisa sentimento de transcrição médica via Azure Language.
    Retorna: positive, negative, neutral + score de confiança.
    """
    import requests

    endpoint = os.environ["AZURE_LANGUAGE_ENDPOINT"]
    key = os.environ["AZURE_LANGUAGE_KEY"]

    url = f"{endpoint}/language/:analyze-text?api-version=2023-04-01"
    headers = {"Ocp-Apim-Subscription-Key": key, "Content-Type": "application/json"}
    body = {
        "kind": "SentimentAnalysis",
        "analysisInput": {
            "documents": [{"id": "1", "language": "pt", "text": text}]
        },
    }

    response = requests.post(url, headers=headers, json=body)
    response.raise_for_status()
    result = response.json()

    doc = result["results"]["documents"][0]
    sentiment = doc["sentiment"]
    scores = doc["confidenceScores"]

    return {
        "sentiment": sentiment,
        "scores": scores,
        "alert": sentiment in ("negative",) and scores.get("negative", 0) > 0.7,
    }


def extract_health_entities(text: str) -> list[dict]:
    """
    Extrai entidades de saúde do texto (diagnósticos, medicamentos, sintomas).
    Usa Azure Text Analytics for Health.
    """
    import requests

    endpoint = os.environ["AZURE_LANGUAGE_ENDPOINT"]
    key = os.environ["AZURE_LANGUAGE_KEY"]

    url = f"{endpoint}/language/analyze-text/jobs?api-version=2023-04-01"
    headers = {"Ocp-Apim-Subscription-Key": key, "Content-Type": "application/json"}
    body = {
        "displayName": "health_extraction",
        "analysisInput": {
            "documents": [{"id": "1", "language": "pt", "text": text}]
        },
        "tasks": [{"kind": "Healthcare"}],
    }

    response = requests.post(url, headers=headers, json=body)
    response.raise_for_status()

    # Healthcare é assíncrono — retorna URL de polling
    operation_url = response.headers.get("operation-location", "")
    return {"status": "submitted", "operation_url": operation_url}
