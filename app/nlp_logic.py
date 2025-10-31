"""
CTI (Cyber Threat Intelligence) PDF Processor
---------------------------------------------
Extracts text from CTI reports, performs NER using SecureBERT,
builds a knowledge graph, and clusters sentences semantically.

Dependencies:
    pip install torch transformers sentence-transformers nltk PyPDF2 igraph scikit-learn pandas matplotlib
"""

import warnings
import re
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import igraph as ig
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN

# ---------------- SETUP ----------------
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

MODEL_NAME = "CyberPeace-Institute/SecureBERT-NER"
NER_INITIALIZED = False
EMBEDDING_INITIALIZED = False

# ---------------- LOAD MODELS ----------------
print("[INFO] Loading models...")

# Load NER model
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
    ner_pipeline = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple"
    )
    NER_INITIALIZED = True
    print("[OK] SecureBERT-NER loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load NER model: {e}")

# Load sentence embedding model
try:
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    EMBEDDING_INITIALIZED = True
    print("[OK] SentenceTransformer loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load SentenceTransformer: {e}")

# Ensure NLTK tokenizer available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True, download_dir="/tmp/nltk_data")
    nltk.data.path.append("/tmp/nltk_data")

# ---------------- UTILITIES ----------------
def extract_pdf_text(pdf_file):
    """Extract text from a PDF file."""
    text = ""
    try:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        print(f"[ERROR] Reading PDF failed: {e}")
    return text.strip()

def split_into_sentences(text):
    """Split text into clean sentences."""
    if not text or not isinstance(text, str):
        return []
    clean_text = re.sub(r"\s+", " ", text)
    sentences = nltk.sent_tokenize(clean_text)
    return [s.strip() for s in sentences if s.strip()]

def chunk_text(text, max_length=512, overlap=50):
    """Split text into overlapping token chunks for NER."""
    if not NER_INITIALIZED or not text:
        return []
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = [
        tokenizer.decode(tokens[i:i + max_length])
        for i in range(0, len(tokens), max_length - overlap)
    ]
    return chunks

# ---------------- KNOWLEDGE GRAPH ----------------
def build_cti_graph(entities, labels):
    """Build a directed CTI knowledge graph based on entity sequences."""
    G = ig.Graph(directed=True)
    if not entities:
        return G

    G.add_vertices(len(entities))
    G.vs["name"] = entities
    G.vs["label"] = entities
    G.vs["type"] = labels

    color_map = {
        "ACT": "#1f78b4",
        "TOOL": "#33a02c",
        "IDTY": "#ff7f00",
        "APT": "#e31a1c",
        "MALWARE": "#fb9a99",
        "IP": "#fdbf6f",
        "DOMAIN": "#b2df8a",
        "CVE": "#ffff99",
        "URL": "#ff7f00"
    }
    G.vs["color"] = [color_map.get(l, "#a6cee3") for l in labels]

    edges, relations = [], []
    for i in range(len(entities) - 1):
        l1, l2 = labels[i], labels[i + 1]
        relation = "related_to"
        if l1 == "IDTY" and l2 == "ACT":
            relation = "performs_ttp"
        elif l1 == "ACT" and l2 == "TOOL":
            relation = "uses_tool"
        elif l1 == "APT" and l2 == "MALWARE":
            relation = "uses_malware"
        elif l1 == "MALWARE" and l2 in ["IP", "DOMAIN"]:
            relation = "connects_to"
        elif l1 == "VULID" and l2 in ["OS", "TOOL"]:
            relation = "affects"
        edges.append((i, i + 1))
        relations.append(relation)

    G.add_edges(edges)
    G.es["label"] = relations
    return G

# ---------------- CLUSTERING ----------------
def perform_clustering(sentences):
    """Perform semantic clustering using SentenceTransformer + DBSCAN."""
    if not sentences or not EMBEDDING_INITIALIZED:
        return None, None, {}

    embeddings = embedding_model.encode(sentences)
    dbscan = DBSCAN(eps=1.0, min_samples=2)
    labels = dbscan.fit_predict(embeddings)

    topic_map = {
        cid: f"Topic {cid}" if cid != -1 else "Outliers"
        for cid in set(labels)
    }
    return embeddings, labels, topic_map

# ---------------- MAIN PROCESSOR ----------------
def process_cti_pdf(file_path):
    """
    Process a PDF CTI report:
    1. Extract text
    2. Perform NER
    3. Build CTI knowledge graph
    4. Optionally cluster sentences
    """
    print(f"[INFO] Processing {file_path}...")
    text = extract_pdf_text(file_path)
    sentences = split_into_sentences(text)
    chunks = chunk_text(text)

    ner_results = []
    if NER_INITIALIZED and chunks:
        for chunk in chunks:
            try:
                ner_results.extend(ner_pipeline(chunk))
            except Exception as e:
                print(f"[WARN] NER chunk failed: {e}")
                continue

    if ner_results:
        df = pd.DataFrame(ner_results).rename(
            columns={"word": "Entity", "entity_group": "Type"}
        )
        df["Score"] = df["score"].round(3)
        G = build_cti_graph(df["Entity"].tolist(), df["Type"].tolist())
    else:
        df = pd.DataFrame(columns=["Entity", "Type", "Score"])
        G = ig.Graph(directed=True)

    # Optional clustering
    _, cluster_labels, topic_map = perform_clustering(sentences)

    print(f"[DONE] Extracted {len(df)} entities, {len(sentences)} sentences.")
    return {
        "entities": df,
        "graph": G,
        "sentences": sentences,
        "cluster_labels": cluster_labels,
        "topic_map": topic_map
    }

# ---------------- DEMO ----------------
if __name__ == "__main__":
    # Example usage: replace with your file path
    pdf_path = "sample_cti_report.pdf"
    result = process_cti_pdf(pdf_path)

    print("\n=== Extracted Entities ===")
    print(result["entities"].head())

    print("\n=== Graph Summary ===")
    print(result["graph"].summary())
