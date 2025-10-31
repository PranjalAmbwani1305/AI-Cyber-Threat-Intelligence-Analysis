import re
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import igraph as ig
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import streamlit as st

# ------------------ INIT CONFIG ------------------
@st.cache_resource
def load_ner_model():
    MODEL_NAME = "CyberPeace-Institute/SecureBERT-NER"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
    ner = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    return ner

ner_pipeline = load_ner_model()
embedder = SentenceTransformer("all-MiniLM-L6-v2")

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# ------------------ TEXT PROCESSING ------------------
def extract_pdf_text(file_path):
    """Reads text from uploaded file (PDF or CSV)."""
    if file_path.name.endswith(".csv"):
        df = pd.read_csv(file_path)
        text_columns = [c for c in df.columns if df[c].dtype == "object"]
        return " ".join(df[text_columns].fillna("").astype(str).values.flatten())
    elif file_path.name.endswith(".pdf"):
        from PyPDF2 import PdfReader
        reader = PdfReader(file_path)
        return " ".join([page.extract_text() or "" for page in reader.pages])
    else:
        return ""

def split_into_sentences(text):
    """Splits text into sentences."""
    text = re.sub(r"\s+", " ", text.strip())
    return nltk.sent_tokenize(text)

# ------------------ NLP FUNCTIONS ------------------
def perform_clustering(sentences, n_clusters=5):
    """Groups sentences into semantic clusters."""
    if not sentences:
        return [], [], {}
    embeddings = embedder.encode(sentences)
    kmeans = KMeans(n_clusters=min(n_clusters, len(sentences)), random_state=42)
    labels = kmeans.fit_predict(embeddings)
    topic_map = {i: [] for i in range(max(labels) + 1)}
    for s, l in zip(sentences, labels):
        topic_map[l].append(s)
    return embeddings, labels, topic_map

def build_cti_graph(entities_df, labels=None):
    """Builds an iGraph knowledge graph from entity pairs."""
    if entities_df.empty:
        return ig.Graph()

    entities = entities_df["Entity"].tolist()
    types = entities_df["Type"].tolist() if "Type" in entities_df else ["Unknown"] * len(entities)

    G = ig.Graph(directed=True)
    G.add_vertices(list(set(entities)))
    color_map = {
        "ACT": "#1f78b4", "TOOL": "#33a02c", "IDTY": "#ff7f00", "APT": "#e31a1c",
        "CVE": "#ffff99", "IP": "#a6cee3", "URL": "#b2df8a", "DOMAIN": "#fdbf6f",
        "HASH": "#fb9a99", "FILE": "#cab2d6", "Unknown": "#cccccc"
    }
    G.vs["color"] = [color_map.get(t, "#cccccc") for t in types[:len(G.vs)]]
    G.vs["label"] = G.vs["name"]

    edges, labels = [], []
    for i in range(len(entities) - 1):
        src, dst = entities[i], entities[i + 1]
        if src != dst:
            edges.append((src, dst))
            labels.append("related_to")
    if edges:
        G.add_edges(edges)
        G.es["label"] = labels
        G.es["color"] = "gray"
    return G

# ------------------ MAIN PROCESSOR ------------------
def process_cti_data(file):
    """Extracts entities, clusters text, and builds CTI knowledge graph."""
    text = extract_pdf_text(file)
    sentences = split_into_sentences(text)
    chunks = [text[i:i + 500] for i in range(0, len(text), 500)]
    results = [res for chunk in chunks for res in ner_pipeline(chunk)]
    if not results:
        return {"entities": pd.DataFrame(), "sentences": sentences, "graph": ig.Graph()}

    entities_df = pd.DataFrame(results).rename(columns={"word": "Entity", "entity_group": "Type"})
    entities_df["Score"] = entities_df["score"].round(3)

    _, labels, topic_map = perform_clustering(sentences)
    graph = build_cti_graph(entities_df)
    return {"entities": entities_df, "sentences": sentences, "graph": graph, "cluster_labels": labels, "topic_map": topic_map}
