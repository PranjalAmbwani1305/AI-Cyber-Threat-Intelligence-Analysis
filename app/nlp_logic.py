import nltk
import pandas as pd
import igraph as ig
import matplotlib.pyplot as plt
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from PyPDF2 import PdfReader
import re
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- DOWNLOAD NLTK RESOURCES ---
nltk.download("punkt", quiet=True)

# --- LOAD MODEL ---
MODEL_NAME = "CyberPeace-Institute/SecureBERT-NER"
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
    ner_pipeline = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple"
    )
    MODEL_LOADED = True
except Exception as e:
    print(f"⚠️ Model load failed: {e}")
    MODEL_LOADED = False


# -------------------------------------------------------------------
# 1️⃣ UTILITIES
# -------------------------------------------------------------------

def extract_pdf_text(file_obj):
    """Extract readable text from PDF."""
    try:
        reader = PdfReader(file_obj)
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"


def split_into_sentences(text: str):
    """Split text into sentences."""
    try:
        sentences = nltk.sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]
    except Exception:
        return [text]


def clean_text(text):
    """Normalize and clean extracted text."""
    text = re.sub(r"\s+", " ", str(text))
    text = text.replace("–", "-").replace("•", "")
    return text.strip()


# -------------------------------------------------------------------
# 2️⃣ NLP CORE
# -------------------------------------------------------------------

def process_cti_pdf(file_obj):
    """
    Handles both structured and unstructured CTI input.
    Returns a dictionary with entities, sentences, and clustering info.
    """
    if not MODEL_LOADED:
        return {"error": "NER model not loaded."}

    text = ""
    if hasattr(file_obj, "name") and file_obj.name.endswith(".pdf"):
        text = extract_pdf_text(file_obj)
    elif hasattr(file_obj, "read"):  # Streamlit uploaded CSV/XLSX
        text = file_obj.read().decode("utf-8", errors="ignore")
    else:
        try:
            df = pd.read_csv(file_obj)
            text_cols = [c for c in df.columns if df[c].dtype == "object"]
            text = " ".join(df[text_cols].fillna("").astype(str).values.flatten())
        except Exception:
            text = str(file_obj)

    text = clean_text(text)
    sentences = split_into_sentences(text)

    entities, labels = [], []
    for sent in sentences:
        try:
            results = ner_pipeline(sent)
            for r in results:
                entities.append(r["word"])
                labels.append(r["entity_group"])
        except Exception:
            continue

    entities_df = pd.DataFrame({"Entity": entities, "Type": labels})
    return {
        "text": text,
        "sentences": sentences,
        "entities": entities_df
    }


# -------------------------------------------------------------------
# 3️⃣ CLUSTERING
# -------------------------------------------------------------------

def perform_clustering(sentences):
    """Groups semantically similar CTI sentences."""
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans

    if not sentences:
        return [], [], {}

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sentences)
    n_clusters = min(5, len(sentences))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    topic_map = {
        i: " | ".join([sentences[j][:80] for j in range(len(sentences)) if labels[j] == i][:2])
        for i in range(n_clusters)
    }

    return embeddings, labels, topic_map


# -------------------------------------------------------------------
# 4️⃣ KNOWLEDGE GRAPH
# -------------------------------------------------------------------

def build_cti_graph(entities_df):
    """Create a visually enhanced CTI knowledge graph using iGraph."""
    if entities_df.empty:
        return ig.Graph()

    entities = entities_df["Entity"].tolist()
    labels = entities_df["Type"].tolist()

    G = ig.Graph(directed=True)
    unique_entities = list(dict.fromkeys(entities))
    G.add_vertices(unique_entities)

    color_palette = {
        "APT": "#ff4b4b", "ACT": "#007bff", "TOOL": "#28a745", "IDTY": "#ffc107",
        "VULID": "#6610f2", "IP": "#20c997", "URL": "#6f42c1", "DOMAIN": "#fd7e14",
        "HASH": "#17a2b8", "FILE": "#e83e8c", "MISC": "#adb5bd"
    }
    node_colors = [color_palette.get(lbl, "#a6a6a6") for lbl in labels[:len(unique_entities)]]
    G.vs["name"] = unique_entities
    G.vs["color"] = node_colors
    G.vs["label"] = G.vs["name"]

    edges = []
    for i in range(len(entities) - 1):
        if entities[i] != entities[i + 1]:
            edges.append((entities[i], entities[i + 1]))

    valid_edges = [(a, b) for a, b in edges if a in G.vs["name"] and b in G.vs["name"]]
    G.add_edges(valid_edges)

    G.es["color"] = "gray"
    G.es["label"] = "related_to"
    return G


def plot_knowledge_graph(G):
    """Return a Matplotlib figure of the knowledge graph."""
    if G.vcount() == 0:
        return None

    layout = G.layout("fr")
    fig, ax = plt.subplots(figsize=(10, 7))
    ig.plot(
        G,
        target=ax,
        layout=layout,
        vertex_size=25,
        vertex_label=G.vs["label"],
        vertex_color=G.vs["color"],
        edge_arrow_size=0.4,
        edge_color="gray"
    )
    ax.set_title("Cyber Threat Knowledge Graph", fontsize=14)
    return fig
