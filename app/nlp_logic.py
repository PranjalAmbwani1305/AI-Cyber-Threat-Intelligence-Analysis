import warnings
import nltk
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import igraph as ig
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------- GLOBAL MODELS ----------------
MODEL_NAME = "CyberPeace-Institute/SecureBERT-NER"
NER_INITIALIZED = False
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
    ner_pipeline = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    NER_INITIALIZED = True
    print("✅ NER model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load NER model: {e}")

try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"❌ Failed to load embedding model: {e}")

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# ---------------- CORE UTILITIES ----------------
def extract_pdf_text(pdf_path):
    """Extract text from PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + " \n"
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"

def chunk_text(text, max_length=512, overlap=50):
    """Split long text into overlapping token chunks."""
    if not NER_INITIALIZED:
        return ["Model not loaded."]
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = [tokenizer.decode(tokens[i:i + max_length])
              for i in range(0, len(tokens), max_length - overlap)]
    return chunks

def split_into_sentences(text):
    sentences = nltk.sent_tokenize(re.sub(r'\n+', ' ', text))
    return [s.strip() for s in sentences if s.strip()]

# ---------------- KNOWLEDGE GRAPH ----------------
def build_cti_graph(entities, labels):
    """Builds directed CTI knowledge graph."""
    G = ig.Graph(directed=True)
    if not entities:
        return G
    G.add_vertices(len(entities))
    G.vs["name"] = entities
    G.vs["label"] = entities
    G.vs["type"] = labels
    color_map = {
        'ACT': '#1f78b4', 'TOOL': '#33a02c', 'IDTY': '#ff7f00',
        'APT': '#e31a1c', 'MALWARE': '#fb9a99', 'IP': '#fdbf6f',
        'DOMAIN': '#b2df8a', 'CVE': '#ffff99', 'URL': '#ff7f00'
    }
    G.vs["color"] = [color_map.get(l, '#a6cee3') for l in labels]
    edges, relations = [], []
    for i in range(len(entities) - 1):
        e1, e2 = entities[i], entities[i + 1]
        l1, l2 = labels[i], labels[i + 1]
        relation = "related_to"
        if l1 == "IDTY" and l2 == "ACT": relation = "performs_ttp"
        elif l1 == "ACT" and l2 == "TOOL": relation = "uses_tool"
        elif l1 == "APT" and l2 == "MALWARE": relation = "uses_malware"
        elif l1 == "MALWARE" and l2 in ["IP", "DOMAIN"]: relation = "connects_to"
        elif l1 == "VULID" and l2 in ["OS", "TOOL"]: relation = "affects"
        edges.append((i, i + 1))
        relations.append(relation)
    G.add_edges(edges)
    G.es["label"] = relations
    return G

def visualize_subgraph(G, entity):
    """Plot 1-hop subgraph for entity."""
    if G is None or entity not in G.vs["name"]:
        return None, f"Entity '{entity}' not found."
    vid = G.vs.find(name=entity).index
    neigh = G.neighbors(vid, mode="all")
    sub = G.induced_subgraph(list(set([vid] + neigh)))
    layout = sub.layout("kamada_kawai")
    fig, ax = plt.subplots(figsize=(10, 8))
    ig.plot(sub, target=ax, layout=layout, vertex_label=sub.vs["name"],
            vertex_color=sub.vs["color"], edge_label=sub.es["label"])
    ax.set_title(f"1-Hop Knowledge Graph for '{entity}'")
    return fig, f"Subgraph with {sub.vcount()} nodes."

# ---------------- CLUSTERING ----------------
def perform_clustering(sentences):
    if not sentences:
        return None, None, "No text to cluster."
    embeddings = embedding_model.encode(sentences)
    dbscan = DBSCAN(eps=1.0, min_samples=2)
    labels = dbscan.fit_predict(embeddings)
    topic_map = {cid: f"Topic {cid}" if cid != -1 else "Outliers"
                 for cid in set(labels)}
    return embeddings, labels, topic_map

def plot_clusters(embeddings, labels, topic_map):
    if embeddings is None:
        return None
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)
    fig, ax = plt.subplots(figsize=(10, 8))
    unique = sorted(set(labels))
    for k in unique:
        mask = labels == k
        ax.scatter(reduced[mask, 0], reduced[mask, 1],
                   label=topic_map.get(k, "Unknown"), alpha=0.7)
    ax.legend()
    ax.set_title("Semantic Topic Clusters")
    return fig

# ---------------- MAIN PROCESSOR ----------------
def process_cti_pdf(file_path):
    text = extract_pdf_text(file_path)
    sentences = split_into_sentences(text)
    chunks = chunk_text(text)
    results = [r for c in chunks for r in ner_pipeline(c)]
    if not results:
        return pd.DataFrame(), None, sentences
    df = pd.DataFrame(results).rename(columns={'word': 'Entity', 'entity_group': 'Type'})
    df['Score'] = df['score'].round(3)
    G = build_cti_graph(df['Entity'].tolist(), df['Type'].tolist())
    return df[['Entity', 'Type', 'Score']], G, sentences
