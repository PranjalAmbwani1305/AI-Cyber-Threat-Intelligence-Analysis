import pandas as pd
import io
import nltk
import networkx as nx
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import streamlit as st

# Ensure sentence tokenizer is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# --- Cached model load for fast startup ---
@st.cache_resource
def load_ner_model():
    MODEL_NAME = "CyberPeace-Institute/SecureBERT-NER"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
    ner_pipeline = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple"
    )
    return ner_pipeline

# --- Text extraction ---
def extract_text(file):
    if file.name.endswith(".csv"):
        try:
            df = pd.read_csv(file)
            text = " ".join(df.astype(str).fillna("").values.flatten())
        except Exception:
            text = ""
    elif file.name.endswith(".txt"):
        text = file.read().decode("utf-8", errors="ignore")
    elif file.name.endswith(".pdf"):
        import PyPDF2
        reader = PyPDF2.PdfReader(file)
        text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    else:
        text = ""
    return text

def split_into_sentences(text):
    return nltk.sent_tokenize(text)

# --- Entity extraction ---
def extract_entities(text):
    ner_pipeline = load_ner_model()
    sentences = split_into_sentences(text)
    entities = []
    for sent in sentences[:50]:  # limit for speed
        entities.extend(ner_pipeline(sent))
    df = pd.DataFrame(entities)
    if not df.empty:
        df = df.rename(columns={"word": "Entity", "entity_group": "Type", "score": "Score"})
        df["Score"] = df["Score"].round(3)
        df = df[["Entity", "Type", "Score"]]
    return df

# --- Knowledge Graph builder ---
def build_cti_graph(entities, labels):
    G = nx.Graph()
    for e, l in zip(entities, labels):
        G.add_node(e, label=l)
    for i in range(len(entities) - 1):
        G.add_edge(entities[i], entities[i + 1])
    return G

# --- Plot graph ---
import matplotlib.pyplot as plt
def plot_cti_graph(G):
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, k=0.3, seed=42)
    node_labels = nx.get_node_attributes(G, "label")

    nx.draw_networkx_nodes(G, pos, node_color="#007ACC", alpha=0.8, node_size=900)
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="white")
    nx.draw_networkx_edges(G, pos, edge_color="#AAAAAA", width=1.0, alpha=0.6)

    plt.axis("off")
    plt.title("Cyber Threat Intelligence Knowledge Graph", fontsize=12)
    return plt.gcf()
