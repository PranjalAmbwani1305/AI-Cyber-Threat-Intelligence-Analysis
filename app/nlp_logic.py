import re
import nltk
import pandas as pd
import igraph as ig
from transformers import pipeline
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import streamlit as st

# ------------------ CACHED MODEL LOAD ------------------
@st.cache_resource
def load_models():
    ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return ner, embedder

ner_pipeline, embedder = load_models()

# ------------------ NLTK CHECK ------------------
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# ------------------ TEXT EXTRACTION ------------------
def extract_pdf_text(file_path):
    """Reads structured/unstructured CTI data."""
    if file_path.name.endswith(".csv"):
        df = pd.read_csv(file_path)
        text_columns = [c for c in df.columns if df[c].dtype == "object"]
        return " ".join(df[text_columns].fillna("").astype(str).values.flatten())
    elif file_path.name.endswith(".pdf"):
        from PyPDF2 import PdfReader
        reader = PdfReader(file_path)
        return " ".join([page.extract_text() or "" for page in reader.pages])
    elif file_path.name.endswith(".txt"):
        return file_path.read().decode("utf-8")
    return ""

# ------------------ SENTENCE SPLITTING ------------------
def split_into_sentences(text):
    """Splits long text into meaningful sentences."""
    text = re.sub(r"\s+", " ", text.strip())
    return nltk.sent_tokenize(text)

# ------------------ CLUSTERING ------------------
def perform_clustering(sentences, n_clusters=5):
    """Performs semantic clustering on CTI sentences."""
    if not sentences:
        return [], [], {}
    embeddings = embedder.encode(sentences, show_progress_bar=False)
    n_clusters = min(n_clusters, len(sentences))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    topic_map = {i: [] for i in range(n_clusters)}
    for s, l in zip(sentences, labels):
        topic_map[l].append(s)
    return embeddings, labels, topic_map

# ------------------ GRAPH BUILDING ------------------
def build_cti_graph(entities_df, labels=None):
    """Creates an interactive CTI knowledge graph."""
    if entities_df.empty:
        return ig.Graph()

    entities = entities_df["Entity"].tolist()
    types = entities_df.get("Type", ["Unknown"] * len(entities))

    G = ig.Graph(directed=True)
    G.add_vertices(list(set(entities)))

    color_map = {
        "ORG": "#33a02c", "PERSON": "#1f78b4", "LOC": "#ff7f00",
        "PRODUCT": "#e31a1c", "MISC": "#6a3d9a", "Unknown": "#aaaaaa"
    }
    G.vs["color"] = [color_map.get(t, "#cccccc") for t in types[:len(G.vs)]]
    G.vs["label"] = G.vs["name"]

    edges, labels = [], []
    for i in range(0, len(entities) - 1, 2):
        src, dst = entities[i], entities[i + 1]
        if src != dst:
            edges.append((src, dst))
            labels.append("related_to")
    if edges:
        G.add_edges(edges)
        G.es["label"] = labels
        G.es["color"] = "gray"
    return G

# ------------------ MAIN PROCESS ------------------
def process_cti_data(file):
    """Main CTI NLP logic pipeline."""
    text = extract_pdf_text(file)
    sentences = split_into_sentences(text)

    # Lightweight NER
    chunk_size = 1000
    text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    results = []
    for chunk in text_chunks[:5]:  # limit for speed
        results += ner_pipeline(chunk)

    if not results:
        return {"entities": pd.DataFrame(), "sentences": sentences, "graph": ig.Graph()}

    entities_df = pd.DataFrame(results).rename(columns={"word": "Entity", "entity_group": "Type"})
    entities_df["Score"] = entities_df["score"].round(3)

    _, labels, topic_map = perform_clustering(sentences)
    graph = build_cti_graph(entities_df)
    return {"entities": entities_df, "sentences": sentences, "graph": graph, "cluster_labels": labels, "topic_map": topic_map}
