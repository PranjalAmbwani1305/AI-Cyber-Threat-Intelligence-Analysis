# app/nlp_logic.py
import os
import re
import nltk
import warnings
import pandas as pd
import igraph as ig
import matplotlib.pyplot as plt
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------- NLTK: robust punkt/punkt_tab handling ----------------
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    try:
        nltk.download("punkt_tab", quiet=True)
    except:
        nltk.download("punkt", quiet=True)

# ---------------- Cached model loaders (fast repeated runs) -------------
# Note: Streamlit uses st.cache_resource; for plain import usage we do simple lazy load.
_NER_PIPELINE = None
_EMBEDDER = None

def get_ner_pipeline(model_name="CyberPeace-Institute/SecureBERT-NER"):
    global _NER_PIPELINE
    if _NER_PIPELINE is None:
        # Use the secureBERT model if available — fallback to smaller NER if load fails
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForTokenClassification.from_pretrained(model_name)
            _NER_PIPELINE = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
        except Exception:
            # fallback to a widely available lightweight NER (fast)
            _NER_PIPELINE = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    return _NER_PIPELINE

def get_embedder(model_name="all-MiniLM-L6-v2"):
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer(model_name)
    return _EMBEDDER

# ---------------- Text extractors ----------------
def extract_text_from_uploaded(uploaded_file):
    """
    Accepts a file-like (Streamlit upload) and returns combined text for NLP.
    Supports: .csv, .xlsx, .txt, .pdf
    """
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        try:
            df = pd.read_csv(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding="utf-8", errors="ignore")
        text_cols = [c for c in df.columns if df[c].dtype == "object"]
        if not text_cols:
            # fallback: combine all columns to string
            return " ".join(df.astype(str).values.flatten())
        return " \n".join(df[text_cols].fillna("").astype(str).agg(" ".join, axis=1).tolist())
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(uploaded_file)
        text_cols = [c for c in df.columns if df[c].dtype == "object"]
        return " \n".join(df[text_cols].fillna("").astype(str).agg(" ".join, axis=1).tolist())
    elif name.endswith(".txt"):
        uploaded_file.seek(0)
        try:
            return uploaded_file.read().decode("utf-8", errors="ignore")
        except Exception:
            uploaded_file.seek(0)
            return uploaded_file.read().decode("latin-1", errors="ignore")
    elif name.endswith(".pdf"):
        try:
            from PyPDF2 import PdfReader
            uploaded_file.seek(0)
            reader = PdfReader(uploaded_file)
            pages = []
            for p in reader.pages:
                pages.append(p.extract_text() or "")
            return "\n".join(pages)
        except Exception:
            # graceful fallback message
            return ""
    else:
        return ""

# ---------------- Sentence splitting ----------------
def split_into_sentences(text):
    if not text or not isinstance(text, str):
        return []
    # normalize whitespace
    txt = re.sub(r"\s+", " ", text).strip()
    try:
        return nltk.sent_tokenize(txt)
    except LookupError:
        # try to redownload if missing
        try:
            nltk.download("punkt_tab", quiet=True)
        except:
            nltk.download("punkt", quiet=True)
        return nltk.sent_tokenize(txt)

# ---------------- NER extraction ----------------
def extract_entities_from_text(text, ner_pipeline=None, max_chunks=10, chunk_size=1500):
    """
    Runs NER over text in chunked mode (to avoid tokenizer length errors).
    Returns pandas DataFrame with columns: Entity, Type, Score
    """
    if ner_pipeline is None:
        ner_pipeline = get_ner_pipeline()

    if not text or not isinstance(text, str) or text.strip() == "":
        return pd.DataFrame(columns=["Entity", "Type", "Score"])

    # chunk text into overlapping slices to keep context but limit length
    text_len = len(text)
    if text_len <= chunk_size:
        chunks = [text]
    else:
        chunks = []
        step = chunk_size // 2
        for i in range(0, min(text_len, chunk_size * max_chunks), step):
            if i + chunk_size > text_len:
                chunks.append(text[i:text_len])
                break
            chunks.append(text[i:i + chunk_size])
        if not chunks:
            chunks = [text[:chunk_size]]

    results = []
    for chunk in chunks:
        try:
            ents = ner_pipeline(chunk)
            for e in ents:
                # normalize keys across pipeline versions
                word = e.get("word") or e.get("entity") or e.get("text") or ""
                group = e.get("entity_group") or e.get("entity") or e.get("label") or "MISC"
                score = float(e.get("score", e.get("confidence", 0.0)))
                results.append({"Entity": str(word).strip(), "Type": str(group).strip(), "Score": round(score, 4)})
        except Exception:
            continue

    if not results:
        return pd.DataFrame(columns=["Entity", "Type", "Score"])

    df = pd.DataFrame(results)
    # deduplicate by best score (keep highest score per entity text)
    df = df.sort_values("Score", ascending=False).drop_duplicates(subset=["Entity"], keep="first").reset_index(drop=True)
    return df

# ---------------- Clustering ----------------
def perform_clustering(sentences, n_clusters=5):
    """
    Semantic clustering with SentenceTransformer + KMeans.
    Returns embeddings, labels, topic_map (top sentences per cluster)
    """
    embedder = get_embedder()
    if not sentences:
        return [], [], {}

    # limit sentences to reasonable number for speed
    MAX_SENTENCES = 500
    sample = sentences[:MAX_SENTENCES]
    embeddings = embedder.encode(sample, show_progress_bar=False)
    n_clusters = min(max(2, n_clusters), len(sample))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    topic_map = {}
    for cid in range(n_clusters):
        topic_map[cid] = [s for s, l in zip(sample, labels) if l == cid][:5]
    return embeddings, list(labels), topic_map

# ---------------- Knowledge graph builder & visualizer ----------------
def build_cti_graph(entities_df):
    """
    Build igraph directed graph from entities dataframe (Entity, Type).
    Returns igraph.Graph
    """
    if entities_df is None or entities_df.empty:
        return ig.Graph(directed=True)

    # Keep order of first occurrence
    unique_entities = []
    type_map = {}
    for _, row in entities_df.iterrows():
        ent = str(row["Entity"]).strip()
        tp = str(row.get("Type", "MISC")).strip()
        if ent and ent not in unique_entities:
            unique_entities.append(ent)
            type_map[ent] = tp

    G = ig.Graph(directed=True)
    G.add_vertices(unique_entities)
    G.vs["name"] = unique_entities
    G.vs["label"] = unique_entities
    G.vs["type"] = [type_map.get(n, "MISC") for n in unique_entities]

    # color map (expandable)
    color_map = {
        "THREAT_ACTOR": "#e31a1c", "APT": "#e31a1c",
        "MALWARE": "#33a02c", "RANSOMWARE": "#1f78b4",
        "IP": "#fdbf6f", "DOMAIN": "#b2df8a", "URL": "#ff7f00",
        "VULID": "#cab2d6", "CVE": "#cab2d6",
        "ACT": "#fb9a99", "IDTY": "#ffd92f",
        "TOOL": "#6a3d9a", "FILE": "#fb9a99", "MISC": "#cccccc"
    }
    G.vs["color"] = [color_map.get(tp, "#cccccc") for tp in G.vs["type"]]

    # infer edges from consecutive appearances (simple heuristic)
    edges = []
    edge_labels = []
    for i in range(len(unique_entities) - 1):
        src = unique_entities[i]
        dst = unique_entities[i + 1]
        if src != dst:
            edges.append((src, dst))
            edge_labels.append("related_to")
    if edges:
        G.add_edges(edges)
        G.es["label"] = edge_labels
        G.es["color"] = "gray"
    return G

def visualize_graph_matplotlib(graph, figsize=(10, 7)):
    """Return matplotlib fig for the igraph graph (readable layout + labels)."""
    if graph is None or len(graph.vs) == 0:
        return None
    layout = graph.layout("fruchterman_reingold") if len(graph.vs) <= 150 else graph.layout("kk")
    fig, ax = plt.subplots(figsize=figsize)
    ig.plot(
        graph,
        target=ax,
        layout=layout,
        vertex_label=graph.vs["label"],
        vertex_size=30,
        vertex_color=graph.vs["color"],
        edge_color="gray",
        bbox=(900, 600),
        margin=50
    )
    ax.set_title("Cyber Threat Knowledge Graph", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig

# ---------------- Top-level processor ----------------
def process_cti_file(uploaded_file, clustering_k=5):
    """
    Full pipeline: extracts text, sentences, entities, clustering, graph.
    Returns dict with keys: text, sentences, entities_df, cluster_labels, topic_map, graph
    """
    text = extract_text_from_uploaded(uploaded_file)
    sentences = split_into_sentences(text)

    ner = get_ner_pipeline()
    entities_df = extract_entities_from_text(text, ner_pipeline=ner, max_chunks=12, chunk_size=1500)

    embeddings, cluster_labels, topic_map = perform_clustering(sentences, n_clusters=clustering_k)
    graph = build_cti_graph(entities_df)

    return {
        "text": text,
        "sentences": sentences,
        "entities": entities_df,
        "cluster_labels": cluster_labels,
        "topic_map": topic_map,
        "graph": graph
    }
