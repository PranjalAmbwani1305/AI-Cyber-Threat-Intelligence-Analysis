import pandas as pd
import io
import nltk
import networkx as nx
from pyvis.network import Network
from transformers import pipeline
import pdfplumber

# Ensure NLTK sentence tokenizer is ready
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# -------------------------
# TEXT EXTRACTION
# -------------------------
def extract_text(file):
    """Extract text from PDF, CSV, or TXT file."""
    if file.name.endswith(".csv"):
        try:
            df = pd.read_csv(file)
            text = " ".join(df.astype(str).fillna("").values.flatten())
        except Exception:
            text = ""
    elif file.name.endswith(".pdf"):
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    elif file.name.endswith(".txt"):
        text = file.read().decode("utf-8", errors="ignore")
    else:
        text = ""
    return text.strip()

# -------------------------
# NLP PIPELINES
# -------------------------
def load_nlp_models():
    """Load NER and Sentiment models."""
    ner = pipeline("ner", grouped_entities=True)
    sentiment = pipeline("sentiment-analysis")
    return ner, sentiment

def analyze_text(text, ner, sentiment):
    """Run NLP pipelines on extracted text."""
    if not text.strip():
        raise ValueError("Empty text content.")
    
    sentences = nltk.sent_tokenize(text)
    entities = []
    for sent in sentences[:50]:
        entities.extend(ner(sent))

    sentiments = sentiment(sentences[:50])
    sentiment_df = pd.DataFrame(sentiments)
    sentiment_summary = sentiment_df["label"].value_counts().reset_index()
    sentiment_summary.columns = ["Sentiment", "Count"]

    return entities, sentiment_summary

# -------------------------
# KNOWLEDGE GRAPH BUILDER
# -------------------------
def build_knowledge_graph(entities):
    """Create a CTI-oriented knowledge graph using PyVis."""
    G = nx.Graph()

    for ent in entities:
        group = ent.get("entity_group", "Unknown")
        word = ent.get("word", "")
        if not word.strip():
            continue

        # Different node colors for CTI entity types
        color_map = {
            "ORG": "#00b4d8",
            "PERSON": "#ffb703",
            "LOC": "#8ecae6",
            "MALWARE": "#d62828",
            "CVE": "#90be6d",
            "TOOL": "#219ebc",
            "Unknown": "#adb5bd"
        }
        color = color_map.get(group, "#adb5bd")

        G.add_node(word, label=word, color=color)
        G.add_node(group, label=group, color="#023047")
        G.add_edge(group, word)

    net = Network(height="600px", width="100%", bgcolor="#0e1117", font_color="white")
    net.from_nx(G)
    net.toggle_physics(True)
    return net.generate_html()
