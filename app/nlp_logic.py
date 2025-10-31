import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

MODEL_NAME = "CyberPeace-Institute/SecureBERT-NER"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
ner_pipeline = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
topic_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def extract_entities(text):
    """Extracts cybersecurity entities using SecureBERT."""
    entities = ner_pipeline(text)
    df = pd.DataFrame(entities).rename(columns={"word": "Entity", "entity_group": "Type"})
    df["Score"] = df["score"].round(3)
    return df[["Entity", "Type", "Score"]]

def analyze_sentiment(text):
    """Performs sentiment analysis on the input text."""
    result = sentiment_pipeline(text[:512])[0]
    return {"label": result["label"], "score": round(result["score"], 3)}

def topic_modeling(text):
    """Performs zero-shot topic classification."""
    candidate_labels = ["malware", "phishing", "APT", "vulnerability", "data breach", "threat actor"]
    result = topic_pipeline(text[:512], candidate_labels)
    return {"topic": result["labels"][0], "confidence": round(result["scores"][0], 3)}
