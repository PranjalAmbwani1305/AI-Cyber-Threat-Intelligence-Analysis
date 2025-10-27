import warnings
import nltk
import re
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

# ---------------- LOAD MODELS ----------------
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
    print("NER model loaded successfully.")
except Exception as e:
    print(f"Failed to load NER model: {e}")

try:
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    print(f"Failed to load sentence transformer: {e}")

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True, download_dir="/tmp/nltk_data")
    nltk.data.path.append("/tmp/nltk_data")

# ---------------- UTILITIES ----------------
def extract_pdf_text(pdf_file):
    """Extract text from a PDF file."""
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def split_into_sentences(text):
    """Split a block of text into sentences."""
    if not text or not isinstance(text, str):
        return []
    sentences = nltk.sent_tokenize(re.sub(r"\n+", " ", text))
    return [s.strip() for s in sentences if s.strip()]

def chunk_text(text, max_length=512, overlap=50):
    """Split text into overlapping chunks for NER."""
    if not NER_INITIALIZED or not text:
        return []
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return [
        tokenizer.decode(tokens[i:i + max_length])
        for i in range(0, len(tokens), max_length - overlap)
    ]

# ---------------- KNOWLEDGE GRAPH ----------------
def build_cti_graph(entities, labels):
    """Build a simple CTI knowledge graph."""
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
    """Perform semantic clustering on sentences."""
    if not sentences or "embedding_model" not in globals():
        return None, None, {}
    embeddings = embedding_model.encode(sentences)
    dbscan = DBSCAN(eps=1.0, min_samples=2)
    labels = dbscan.fit_predict(embeddings)
    topic_map = {cid: f"Topic {cid}" if cid != -1 else "Outliers"
                 for cid in set(labels)}
    return embeddings, labels, topic_map

# ---------------- MAIN PROCESSOR ----------------
def process_cti_pdf(file):
    """
    Process a PDF CTI report:
    - Extract text
    - Perform NER
    - Build knowledge graph
    """
    text = extract_pdf_text(file)
    sentences = split_into_sentences(text)

    chunks = chunk_text(text)
    results = []
    if NER_INITIALIZED and chunks:
        for chunk in chunks:
            try:
                results.extend(ner_pipeline(chunk))
            except Exception:
                continue

    if results:
        df = pd.DataFrame(results).rename(
            columns={"word": "Entity", "entity_group": "Type"}
        )
        df["Score"] = df["score"].round(3)
        G = build_cti_graph(df["Entity"].tolist(), df["Type"].tolist())
    else:
        df = pd.DataFrame(columns=["Entity", "Type", "Score"])
        G = None

    return df[["Entity", "Type", "Score"]], G, sentences
