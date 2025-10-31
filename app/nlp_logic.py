# nlp_logic.py
"""
NLP + CTI logic module.
Exports:
  - load_models()
  - extract_text_from_file(file)
  - split_into_sentences(text)
  - run_ner(text_or_chunks, ner_pipeline)
  - sentiment_analysis(sentences, sentiment_pipeline)
  - topic_modeling(sentences, n_topics=6)
  - perform_clustering(sentences, embedding_model=None)
  - build_cti_graph(entities, types)
  - plot_cti_graph_matplotlib(G, figsize=(10,8))
  - simple_summarize(text, n_sentences=3)

This module attempts to load heavy models but provides safe fallbacks.
"""

import os
import re
import math
import warnings
from typing import List, Tuple, Dict, Any

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# NLP infra
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# Lightweight alternatives
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.pairwise import cosine_similarity

# PDF reading
from PyPDF2 import PdfReader

# NLTK sentence tokenizer
import nltk
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    try:
        nltk.download("punkt", quiet=True)
    except Exception:
        pass
from nltk.tokenize import sent_tokenize

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# -------------------------
# Model loading / helpers
# -------------------------
def load_models(model_name_ner: str = "CyberPeace-Institute/SecureBERT-NER"):
    """
    Try to load NER model + sentiment + embedding. Return dictionary of objects.
    If heavy models fail, return None for them and the app will use fallbacks.
    """
    models = {
        "ner_tokenizer": None,
        "ner_model": None,
        "ner_pipeline": None,
        "sentiment_pipeline": None,
        "embedding_model": None
    }

    # Load NER
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_ner)
        model = AutoModelForTokenClassification.from_pretrained(model_name_ner)
        ner_pipe = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple"
        )
        models["ner_tokenizer"] = tokenizer
        models["ner_model"] = model
        models["ner_pipeline"] = ner_pipe
    except Exception as e:
        # don't crash; set None and use fallback
        print(f"[nlp_logic] NER model load failed: {e}. NER will fallback to simple heuristics.")

    # sentiment
    try:
        sentiment = pipeline("sentiment-analysis")
        models["sentiment_pipeline"] = sentiment
    except Exception:
        print("[nlp_logic] Sentiment model failed to load. Using keyword heuristic fallback.")

    # Embeddings
    if SentenceTransformer is not None:
        try:
            emb = SentenceTransformer("all-MiniLM-L6-v2")
            models["embedding_model"] = emb
        except Exception:
            print("[nlp_logic] SentenceTransformer failed to load; clustering will fallback to TF-IDF.")
    else:
        print("[nlp_logic] sentence-transformers package not available; clustering will fallback.")
    return models


# -------------------------
# Text extraction
# -------------------------
def extract_text_from_file(uploaded_file) -> str:
    """
    Accepts a Python file-like object (Streamlit upload or similar).
    Supports: CSV, TXT, PDF, XLSX.
    Returns concatenated textual content as a string.
    """
    filename = getattr(uploaded_file, "name", None) or str(uploaded_file)
    if not filename:
        raise ValueError("Missing file name")

    filename = filename.lower()
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(uploaded_file, dtype=str, encoding="utf-8", errors="ignore")
            # Combine all string columns into text
            textcols = [c for c in df.columns if df[c].dtype == "object"]
            if not textcols:
                # fallback: flatten everything
                content = " ".join(df.fillna("").astype(str).values.flatten())
            else:
                content = "\n".join(df[textcols].astype(str).agg(" ".join, axis=1).tolist())
            return content
        elif filename.endswith(".xlsx") or filename.endswith(".xls"):
            df = pd.read_excel(uploaded_file)
            textcols = [c for c in df.columns if df[c].dtype == "object"]
            if not textcols:
                return " ".join(df.fillna("").astype(str).values.flatten())
            return "\n".join(df[textcols].astype(str).agg(" ".join, axis=1).tolist())
        elif filename.endswith(".txt"):
            raw = uploaded_file.read()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="ignore")
            return raw
        elif filename.endswith(".pdf"):
            # Use PyPDF2 which is light
            reader = PdfReader(uploaded_file)
            pages = []
            for p in reader.pages:
                txt = p.extract_text()
                if txt:
                    pages.append(txt)
            return "\n".join(pages)
        else:
            # try to read as text
            raw = uploaded_file.read()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="ignore")
            return raw
    except Exception as e:
        raise ValueError(f"Failed to extract text: {e}")


# -------------------------
# Utilities
# -------------------------
def split_into_sentences(text: str) -> List[str]:
    if not text or not isinstance(text, str):
        return []
    # normalize whitespace
    txt = re.sub(r"\s+", " ", text).strip()
    try:
        sents = sent_tokenize(txt)
    except Exception:
        # fallback: naive split on punctuation
        sents = re.split(r'(?<=[.!?])\s+', txt)
    sents = [s.strip() for s in sents if s.strip()]
    return sents


# -------------------------
# NER
# -------------------------
def run_ner_on_text(text: str, ner_pipeline_obj) -> List[Dict[str, Any]]:
    """
    Run NER pipeline on text. If pipeline is None, fallback: simple keyword/IP/CVE regex matching.
    Returns list of {'word','entity_group','score'} style dicts to be consistent.
    """
    if not text:
        return []

    if ner_pipeline_obj is not None:
        # chunk long text into ~500-token chunks (pipeline will accept strings)
        # simple split by sentences so we keep boundaries
        sents = split_into_sentences(text)
        results = []
        # process in groups of ~50 sentences to limit tokenization cost
        group_size = 50
        for i in range(0, len(sents), group_size):
            chunk = " ".join(sents[i : i + group_size])
            try:
                out = ner_pipeline_obj(chunk)
                # pipeline already provides aggregation; ensure uniform fields
                for item in out:
                    results.append({
                        "word": item.get("word") or item.get("entity") or item.get("text"),
                        "entity_group": item.get("entity_group") or item.get("entity"),
                        "score": float(item.get("score", 0.0))
                    })
            except Exception:
                continue
        return results
    else:
        # Fallback NER heuristics for CTI: IPs, CVEs, domains, simple known tokens
        results = []
        # IP regex
        ip_re = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
        cve_re = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE)
        domain_re = re.compile(r"\b([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b")
        # known threat actor / malware keywords list minimal
        malware_list = ["TrickBot", "QakBot", "LockBit", "Emotet", "Cobalt Strike", "Ryuk"]
        actors = ["APT29", "Lazarus Group", "FIN7", "WIZARD SPIDER", "Mustang Panda"]

        for m in ip_re.findall(text):
            results.append({"word": m, "entity_group": "IP", "score": 0.8})
        for m in set([x.upper() for x in cve_re.findall(text)]):
            results.append({"word": m, "entity_group": "CVE", "score": 0.9})
        # domains - filter out true IP-like numbers
        for m in domain_re.findall(text):
            if re.match(r"^\d", m):  # starts with digit -> not domain
                continue
            results.append({"word": m, "entity_group": "DOMAIN", "score": 0.7})
        for mw in malware_list:
            if mw.lower() in text.lower():
                results.append({"word": mw, "entity_group": "MALWARE", "score": 0.85})
        for a in actors:
            if a.lower() in text.lower():
                results.append({"word": a, "entity_group": "THREAT_ACTOR", "score": 0.85})
        # dedupe keeping highest score
        seen = {}
        final = []
        for r in results:
            w = r["word"]
            if w not in seen or r["score"] > seen[w]["score"]:
                seen[w] = r
        final = list(seen.values())
        return final


# -------------------------
# Sentiment (fallback)
# -------------------------
def sentiment_analysis(sentences: List[str], sentiment_pipeline_obj=None, limit: int = 200) -> pd.DataFrame:
    """
    Returns a DataFrame with columns ['label','score','text'] for up to limit sentences.
    """
    sents = sentences[:limit]
    rows = []
    if sentiment_pipeline_obj is not None:
        try:
            out = sentiment_pipeline_obj(sents)
            for sent, res in zip(sents, out):
                rows.append({"label": res.get("label"), "score": float(res.get("score", 0.0)), "text": sent})
        except Exception:
            sentiment_pipeline_obj = None

    if sentiment_pipeline_obj is None:
        # simple heuristic: presence of 'malware', 'attack' => NEGATIVE
        for sent in sents:
            l = "NEUTRAL"
            score = 0.5
            t = sent.lower()
            if any(k in t for k in ["exploit", "malware", "ransom", "compromise", "breach", "attack"]):
                l = "NEGATIVE"
                score = 0.9
            elif any(k in t for k in ["observed", "detected", "suspicious"]):
                l = "MIXED"
                score = 0.7
            else:
                l = "NEUTRAL"
                score = 0.5
            rows.append({"label": l, "score": score, "text": sent})
    return pd.DataFrame(rows)


# -------------------------
# Topic modeling (TF-IDF + NMF)
# -------------------------
def topic_modeling(sentences: List[str], n_topics: int = 6) -> Tuple[Dict[int, List[str]], List[int]]:
    """
    Returns (topic_keywords_map, topic_assignment_list)
      - topic_keywords_map: {topic_id: [top_keywords]}
      - topic_assignment_list: list with same length as sentences -> topic_id
    Uses TF-IDF + NMF for lightweight topic extraction.
    """
    if not sentences:
        return {}, []

    # Limit size for speed
    max_docs = 1000
    docs = sentences[:max_docs]

    vectorizer = TfidfVectorizer(stop_words="english", max_features=3000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(docs)
    # handle degenerate case
    n_topics = min(n_topics, X.shape[0], 20)
    if n_topics <= 0:
        return {}, [-1] * len(docs)

    nmf = NMF(n_components=n_topics, random_state=42, init="nndsvda", max_iter=300)
    W = nmf.fit_transform(X)
    H = nmf.components_

    feature_names = vectorizer.get_feature_names_out()
    topic_keywords = {}
    for topic_idx, comp in enumerate(H):
        top_idx = np.argsort(comp)[-8:][::-1]
        top_terms = [feature_names[i] for i in top_idx if comp[i] > 0]
        topic_keywords[topic_idx] = top_terms

    topic_assignment = np.argmax(W, axis=1).tolist()
    # pad assignment to full sentences length using -1 for truncated parts
    full_assignment = topic_assignment + [-1] * max(0, len(sentences) - len(docs))
    return topic_keywords, full_assignment


# -------------------------
# Clustering (embeddings -> DBSCAN/KMeans)
# -------------------------
def perform_clustering(sentences: List[str], embedding_model=None, eps=0.9, min_samples=2) -> Tuple[Any, List[int]]:
    """
    Return (embeddings, cluster_labels)
    If embedding_model is None fallback to TF-IDF vectors.
    """
    if not sentences:
        return None, []

    docs = sentences[:2000]  # keep bounded for memory

    if embedding_model is not None:
        try:
            emb = embedding_model.encode(docs, convert_to_numpy=True, show_progress_bar=False)
        except Exception:
            emb = None
    else:
        emb = None

    if emb is None:
        # TF-IDF fallback
        v = TfidfVectorizer(stop_words="english", max_features=2000)
        emb = v.fit_transform(docs).toarray()

    # DBSCAN expects eps tuned; KMeans fallback
    try:
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(emb)
        labels = clustering.labels_.tolist()
    except Exception:
        # fallback KMeans with 6 clusters
        k = min(6, len(docs))
        clustering = KMeans(n_clusters=k, random_state=42).fit(emb)
        labels = clustering.labels_.tolist()
    # pad
    labels_full = labels + [-1] * max(0, len(sentences) - len(docs))
    return emb, labels_full


# -------------------------
# Knowledge graph: build + plotting (networkx + matplotlib)
# -------------------------
def build_cti_graph(entities: List[str], types: List[str]) -> nx.DiGraph:
    """
    Build a directed graph from entity sequences with CTI-aware relation rules.
    Entities and types are parallel lists (extracted in document order).
    """
    G = nx.DiGraph()
    if not entities or not types:
        return G

    # Map unique names -> type (prefer first occurrence)
    name2type = {}
    for e, t in zip(entities, types):
        e_clean = str(e).strip()
        if not e_clean:
            continue
        if e_clean not in name2type:
            name2type[e_clean] = t

    # Add nodes with type attribute
    for name, t in name2type.items():
        G.add_node(name, ctype=t)

    # Add edges using simple rule-based adjacency relations
    cleaned = [str(x).strip() for x in entities]
    for i in range(len(cleaned) - 1):
        n1, t1 = cleaned[i], types[i]
        n2, t2 = cleaned[i + 1], types[i + 1]
        if not n1 or not n2 or n1 not in G.nodes or n2 not in G.nodes:
            continue

        relation = "related_to"
        if t1 in ["THREAT_ACTOR", "APT"] and t2 in ["MALWARE", "RANSOMWARE"]:
            relation = "uses"
        elif t1 in ["MALWARE", "RANSOMWARE"] and t2 in ["IP", "DOMAIN", "URL"]:
            relation = "connects_to"
        elif t1 in ["IP", "DOMAIN"] and t2 in ["MALWARE"]:
            relation = "hosts"
        elif t1 in ["ACT"] and t2 in ["MALWARE", "TOOL"]:
            relation = "implements"
        elif t1 in ["CVE", "VULID"] and t2 in ["OS", "TOOL"]:
            relation = "affects"

        # accumulate edge attributes (counts)
        if G.has_edge(n1, n2):
            G[n1][n2]["weight"] += 1
            # append relation if new
            if relation not in G[n1][n2]["relation"]:
                G[n1][n2]["relation"].append(relation)
        else:
            G.add_edge(n1, n2, weight=1, relation=[relation])
    return G


def plot_cti_graph_matplotlib(G: nx.DiGraph, figsize=(10, 8)):
    """
    Returns a matplotlib figure for the directed CTI graph with clear styles:
     - color by node type
     - label edges with relations
     - use spring_layout but try to force readable spacing
    """
    if G is None or len(G.nodes) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No graph to display", ha="center", va="center")
        ax.axis("off")
        return fig

    # node colors map
    type_color = {
        "THREAT_ACTOR": "#e31a1c",
        "APT": "#e31a1c",
        "MALWARE": "#33a02c",
        "RANSOMWARE": "#1f78b4",
        "IP": "#fdbf6f",
        "DOMAIN": "#b2df8a",
        "URL": "#b2df8a",
        "CVE": "#cab2d6",
        "VULID": "#cab2d6",
        "ACT": "#ff7f00",
        "IDTY": "#ffd92f",
        "FILE": "#fb9a99",
    }

    node_colors = []
    node_sizes = []
    for n in G.nodes:
        t = G.nodes[n].get("ctype", "MISC")
        node_colors.append(type_color.get(t, "#a6cee3"))
        # size scaled by degree (min size 300)
        deg = max(1, G.degree(n))
        node_sizes.append(300 + deg * 80)

    # layout: use spring (force) with seed for reproducibility; may adjust k by size
    try:
        pos = nx.spring_layout(G, k=0.7, seed=42, iterations=200)
    except Exception:
        pos = nx.spring_layout(G, seed=42)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#0f1113")  # dark background
    # draw nodes and labels with white text
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.95, linewidths=1, edgecolors="black")
    nx.draw_networkx_labels(G, pos, font_size=9, font_color="white")

    # draw edges and relation labels
    widths = [max(1.0, G[u][v].get("weight", 1)) for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=widths, edge_color="#7f8c8d", arrowsize=20, arrowstyle="-|>")
    # compose edge label from relations list
    edge_label_map = {}
    for u, v in G.edges():
        rels = G[u][v].get("relation", [])
        edge_label_map[(u, v)] = ", ".join(rels[:2])
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_label_map, font_color="#c0c0c0", font_size=8)

    ax.set_title("CTI Knowledge Graph (1-hop relationships shown)", color="white", pad=12)
    ax.set_axis_off()
    plt.tight_layout()
    return fig


# -------------------------
# Simple extractive summarizer
# -------------------------
def simple_summarize(text: str, n_sentences: int = 3) -> str:
    """
    A quick extractive summarizer: TF-IDF scoring of sentences, return top n sentences in original order.
    Works as a fallback if no summarization model available.
    """
    sents = split_into_sentences(text)
    if not sents:
        return ""
    if len(sents) <= n_sentences:
        return " ".join(sents)

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(sents)
    # score = sum of tf-idf terms per sentence
    scores = X.sum(axis=1).A1
    # pick top N by score
    top_idx = np.argsort(scores)[-n_sentences:]
    top_idx_sorted = sorted(top_idx.tolist())
    summary = " ".join([sents[i] for i in top_idx_sorted])
    return summary
