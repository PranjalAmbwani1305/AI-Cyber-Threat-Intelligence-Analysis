# app/nlp_logic.py
import pandas as pd
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
import warnings
import spacy
from spacy import displacy

warnings.filterwarnings("ignore", category=FutureWarning)

# Try loading a spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None


# -------------------- MOCK NER PIPELINE --------------------
def mock_ner_pipeline(text):
    """Simulated named entity recognition output."""
    return [
        {'word': 'APT29', 'entity_group': 'THREAT_ACTOR', 'score': 0.99},
        {'word': 'TrickBot', 'entity_group': 'MALWARE', 'score': 0.97},
        {'word': '192.168.1.1', 'entity_group': 'IP', 'score': 0.98},
        {'word': 'LockBit', 'entity_group': 'RANSOMWARE', 'score': 0.96},
        {'word': 'CVE-2024-4228', 'entity_group': 'VULID', 'score': 0.99},
    ]


# -------------------- TEXT EXTRACTION HELPERS --------------------
def extract_pdf_text(pdf_file):
    """Mock PDF text extraction."""
    return (
        "APT29 used spear phishing to deliver TrickBot malware. "
        "TrickBot connected to 192.168.1.1. LockBit ransomware was also involved. "
        "CVE-2024-4228 was exploited during the intrusion."
    )


def split_into_sentences(text):
    """Split text into sentences using spaCy."""
    if not nlp:
        return text.split(". ")
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def clean_and_normalize_entities(df):
    """Normalize mock NER output."""
    if df.empty:
        return pd.DataFrame(columns=["Entity", "Type", "Score"])
    df = df.rename(columns={'word': 'Entity', 'entity_group': 'Type', 'score': 'Score'})
    df['Score'] = df['Score'].round(3)
    df = df.drop_duplicates(subset='Entity')
    return df.reset_index(drop=True)


# -------------------- GRAPH CREATION --------------------
def build_cti_graph(df_entities, labels=None):
    """Build simple CTI knowledge graph."""
    if df_entities.empty:
        return ig.Graph(directed=True)

    G = ig.Graph(directed=True)
    G.add_vertices(df_entities['Entity'].tolist())
    G.vs['type'] = df_entities['Type'].tolist()
    G.vs['color'] = [
        {"THREAT_ACTOR": "red", "MALWARE": "green", "IP": "orange", "RANSOMWARE": "blue"}.get(t, "gray")
        for t in G.vs['type']
    ]

    edges = [(i, i + 1) for i in range(len(G.vs) - 1)]
    G.add_edges(edges)
    G.es['label'] = ['related_to'] * len(edges)
    return G


# -------------------- STRUCTURED DATA --------------------
def process_structured_data(file_obj):
    """Process structured CTI dataset (CSV/XLSX)."""
    df = pd.read_csv(file_obj)
    df['combined_text'] = df.astype(str).agg(' '.join, axis=1)
    text = " ".join(df['combined_text'].tolist())

    # Mock entity extraction
    ner_results = mock_ner_pipeline(text)
    entities_df = clean_and_normalize_entities(pd.DataFrame(ner_results))
    sentences = split_into_sentences(text)
    graph = build_cti_graph(entities_df, sentences)

    return {
        "entities": entities_df,
        "sentences": sentences,
        "graph": graph
    }


# -------------------- UNSTRUCTURED DATA --------------------
def process_cti_pdf(pdf_file):
    """Process unstructured CTI reports (PDF)."""
    text = extract_pdf_text(pdf_file)
    sentences = split_into_sentences(text)
    results = mock_ner_pipeline(text)
    entities_df = clean_and_normalize_entities(pd.DataFrame(results))
    graph = build_cti_graph(entities_df, sentences)
    return {
        "text": text,
        "entities": entities_df,
        "sentences": sentences,
        "graph": graph
    }


# -------------------- CLUSTERING --------------------
def perform_clustering(sentences):
    """Simulate clustering output."""
    if not sentences:
        return sentences, [], {}
    cluster_labels = np.random.randint(0, 3, len(sentences))
    topic_map = {0: "Malware", 1: "Actor Behavior", 2: "Vulnerability"}
    return sentences, cluster_labels, topic_map
