import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gradio as gr
from datetime import datetime
import pandas as pd

#---Emotion classification model---#
emotion_model_name = "SamLowe/roberta-base-go_emotions"
emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)
emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_name)

#---Sentiment polarity model----#
sentiment_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)

def sentiment_analysis(text):
    if not text.strip():
        return pd.DataFrame([{
            "Label": "No input text provided",
            "Score": 0.0,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }])

    inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    label = sentiment_model.config.id2label[torch.argmax(probs).item()]
    score = torch.max(probs).item()

    return pd.DataFrame([{
        "Label": label,
        "Score": round(score, 3),
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }])

def cti_classification(text):
    """
    Cybersecurity-aware classification replacing generic emotion labels.
    Deduplicates labels even if keywords appear multiple times.
    """
    if not text.strip():
        return pd.DataFrame([{
            "Label": "No input text provided",
            "Score": 0.0,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }])

    # Keyword-to-label mapping
    keywords = {
        "phishing": "Phishing",
        "malware": "Malware",
        "ransomware": "Malware",
        "cve": "Vulnerability",
        "exploit": "Exploit",
        "incident": "Security Alert",
        "breach": "Security Alert",
        "attack": "Attack"
    }

    text_lower = text.lower()
    detected_labels = set()  # Use a set to deduplicate
    for word, label in keywords.items():
        if word in text_lower:
            detected_labels.add(label)  # Only unique labels

    if not detected_labels:
        detected_labels.add("Informational")  # Default label

    return pd.DataFrame([{
        "Label": label,
        "Score": 1.0 if label != "Informational" else 0.5,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    } for label in detected_labels])

with gr.Blocks(title="CTI Text Classifier + Sentiment Detector") as demo:
    gr.Markdown("## CTI Text Classifier + Sentiment Detector")
    gr.Markdown("Enter cybersecurity-related text and analyze both **Sentiment** and **CTI classification** (with timestamps).")

    with gr.Tab(" Input"):
        input_text = gr.Textbox(
            lines=6,
            label="Enter text",
            placeholder="e.g., A phishing campaign distributing the Emotet loader..."
        )
        analyze_btn = gr.Button("Analyze")

    with gr.Tab("Results"):
        gr.Markdown("###  Sentiment Analysis")
        sentiment_output = gr.Dataframe(headers=["Label", "Score", "Timestamp"], interactive=False)

        gr.Markdown("### Cybersecurity Classification")
        cti_output = gr.Dataframe(headers=["Label", "Score", "Timestamp"], interactive=False)

    # Button triggers both analyses
    analyze_btn.click(
        fn=lambda text: (sentiment_analysis(text), cti_classification(text)),
        inputs=input_text,
        outputs=[sentiment_output, cti_output]
    )

# Launch the app
demo.launch()

'''  Sample Input :- A phishing attack was detected targeting multiple users.
This phishing campaign uses ransomware malware to exploit CVE-2025-12345 vulnerabilities.
The attack has caused a security incident and potential breach.
Malware analysis shows the same ransomware is being reused in another attack.
'''