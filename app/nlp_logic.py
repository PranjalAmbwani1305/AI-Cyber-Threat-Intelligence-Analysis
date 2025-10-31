import pandas as pd
import igraph as ig
import nltk
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Optional for first run
nltk.download('punkt', quiet=True)

# -----------------------------
# GLOBAL MODEL INITIALIZATION
# -----------------------------
NER_MODEL_NAME = "CyberPeace-Institute/SecureBERT-NER"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

ner_tokenizer = None
ner_model = None
ner_pipeline = None
sentence_model = None

# -----------------------------
# 1. LOAD MODELS
# -----------------------------
def load_models():
    """Initialize NLP and NER models (lazy loading)."""
    global ner_tokenizer, ner_model, ner_pipeline, sentence_model

    if ner_pipeline is None:
        try:
            ner_tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_NAME)
            ner_model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_NAME)
            ner_pipeline = pipeline(
                "token-classification",
                model=ner_model,
                tokenizer=ner_tokenizer,
                aggregation_strategy="simple"
            )
            print("✅ SecureBERT-NER loaded successfully.")
        except Exception as e:
            print(f"❌ Failed to load NER model: {e}")

    if sentence_model is None:
        try:
            sentence_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            print("✅ SentenceTransformer loaded successfully.")
        except Exception as e:
            print(f"❌ Failed to load SentenceTransformer: {e}")


# -----------------------------
# 2. ENTITY EXTRACTION
# -----------------------------
def extract_entities(text):
    """Extract entities from text using SecureBERT-NER."""
    if ner_pipeline is None:
        load_models()

    if not text or len(text.strip()) == 0:
        return pd.DataFrame(columns=["Entity", "Type", "Score"])

    try:
        ner_results = ner_pipeline(text[:10000])  # limit for performance
        df = pd.DataFrame(ner_results)
        if not df.empty:
            df.rename(columns={"word": "Entity", "entity_group": "Type", "score": "Score"}, inplace=True)
            df["Score"] = df["Score"].round(3)
            df = df[["Entity", "Type", "Score"]]
        return df
    except Exception as e:
        print(f"NER extraction error: {e}")
        return pd.DataFrame(columns=["Entity", "Type", "Score"])


# -----------------------------
# 3. CLUSTERING LOGIC
# -----------------------------
def perform_clustering(sentences, num_clusters=5):
    """Clusters semantically similar sentences."""
    if sentence_model is None:
        load_models()

    if not sentences or len(sentences) < 2:
        return [], [], {}

    try:
        embeddings = sentence_model.encode(sentences)
        kmeans = KMeans(n_clusters=min(num_clusters, len(sentences)//2 or 1), random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)

        topic_map = {}
        for i, label in enumerate(cluster_labels):
            topic_map.setdefault(label, []).append(sentences[i])

        return embeddings, cluster_labels, topic_map
    except Exception as e:
        print(f"Clustering error: {e}")
        return [], [], {}


# -----------------------------
# 4. KNOWLEDGE GRAPH CONSTRUCTION
# -----------------------------
def build_cti_graph(entities_df, labels):
    """
    Build a semantically enriched Cyber Threat Intelligence Knowledge Graph.
    Each entity node has a CTI-aware color and connection meaning.
    """
    if entities_df is None or entities_df.empty:
        return ig.Graph()

    entities = [str(e).strip() for e in entities_df["Entity"].tolist()]
    unique_entities = list(dict.fromkeys(entities))
    node_labels = [labels[entities.index(e)] for e in unique_entities]

    G = ig.Graph(directed=True)
    G.add_vertices(len(unique_entities))
    G.vs["name"] = unique_entities
    G.vs["node_type"] = node_labels

    # --- Color and sizing ---
    color_map = {
        "ORG": "#1f77b4", "PERSON": "#2ca02c", "GPE": "#ff7f0e", "PRODUCT": "#9467bd",
        "EVENT": "#e377c2", "DATE": "#8c564b", "LOC": "#bcbd22", "NORP": "#7f7f7f",
        "CVE": "#d62728", "MALWARE": "#17becf", "TOOL": "#aec7e8", "VULNERABILITY": "#ff9896"
    }
    G.vs["color"] = [color_map.get(t, "#d3d3d3") for t in node_labels]
    G.vs["size"] = [28 if t in ["ORG", "CVE", "MALWARE"] else 18 for t in node_labels]

    # --- Relationship patterns ---
    edges = []
    relations = []

    for i in range(len(unique_entities) - 1):
        e1, e2 = unique_entities[i], unique_entities[i + 1]
        l1, l2 = node_labels[i], node_labels[i + 1]

        if l1 == "ORG" and l2 in ["MALWARE", "PRODUCT"]:
            rel = "develops"
        elif l1 == "ORG" and l2 == "CVE":
            rel = "responsible_for"
        elif l1 == "MALWARE" and l2 == "CVE":
            rel = "exploits"
        elif l1 == "MALWARE" and l2 == "GPE":
            rel = "targets_region"
        elif l1 == "ORG" and l2 == "GPE":
            rel = "operates_in"
        elif l1 == "PERSON" and l2 == "ORG":
            rel = "affiliated_with"
        elif l1 == "CVE" and l2 == "PRODUCT":
            rel = "affects"
        elif l1 == "MALWARE" and l2 == "TOOL":
            rel = "uses_tool"
        elif l1 == "ORG" and l2 == "EVENT":
            rel = "involved_in"
        else:
            rel = "related_to"

        edges.append((i, i + 1))
        relations.append(rel)

    G.add_edges(edges)
    G.es["label"] = relations
    G.es["color"] = ["#A0A0A0" if r == "related_to" else "#555555" for r in relations]
    G.es["width"] = [2 if r != "related_to" else 1 for r in relations]

    return G


# -----------------------------
# 5. MAIN PIPELINE ENTRY POINT
# -----------------------------
def process_cti_pdf(file_obj):
    """
    Process a structured CTI dataset (CSV or PDF text)
    Extracts entities, clusters sentences, and builds a graph.
    """
    try:
        if hasattr(file_obj, "read"):
            text = file_obj.read().decode("utf-8", errors="ignore")
        else:
            with open(file_obj, "r", encoding="utf-8") as f:
                text = f.read()
    except Exception:
        text = str(file_obj)

    # Split text into sentences
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        sentences = [text]

    entities_df = extract_entities(text)
    if entities_df.empty:
        return {"entities": pd.DataFrame(), "sentences": sentences, "graph": ig.Graph()}

    labels = entities_df["Type"].tolist()
    graph = build_cti_graph(entities_df, labels)
    embeddings, cluster_labels, topic_map = perform_clustering(sentences)

    return {
        "entities": entities_df,
        "sentences": sentences,
        "graph": graph,
        "cluster_labels": cluster_labels,
        "topic_map": topic_map
    }
