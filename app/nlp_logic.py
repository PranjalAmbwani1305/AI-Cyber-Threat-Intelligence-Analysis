# nlp_logic.py
import pandas as pd
import matplotlib.pyplot as plt
import igraph as ig
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from PyPDF2 import PdfReader
import warnings

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

MODEL_NAME = "CyberPeace-Institute/SecureBERT-NER"


# --- LOAD MODEL ---
def load_model():
    """Load SecureBERT-NER model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
    ner_pipeline = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple"
    )
    return tokenizer, ner_pipeline


# --- EXTRACT TEXT ---
def extract_text(uploaded_file):
    """Extract text content from CSV, PDF, or TXT files."""
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        text_columns = [col for col in df.columns if df[col].dtype == "object"]
        text = " ".join(df[text_columns].fillna("").astype(str).values.flatten())

    elif filename.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        text = " ".join([page.extract_text() or "" for page in reader.pages])

    elif filename.endswith(".txt"):
        text = uploaded_file.read().decode("utf-8", errors="ignore")

    else:
        text = ""

    return text


# --- CHUNK TEXT ---
def chunk_text(text, tokenizer, max_length=512, overlap=50):
    """Tokenize text into overlapping chunks."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), max_length - overlap):
        chunk = tokens[i:i + max_length]
        chunks.append(tokenizer.decode(chunk))
    return chunks


# --- BUILD CTI KNOWLEDGE GRAPH ---
def build_cti_graph(entities, labels):
    """Construct an iGraph-based CTI knowledge graph."""
    name_to_label = {}
    vertex_names = []
    for ent, lab in zip(entities, labels):
        clean_ent = ent.strip()
        if clean_ent and clean_ent not in name_to_label:
            name_to_label[clean_ent] = lab
            vertex_names.append(clean_ent)

    G = ig.Graph(directed=True)
    G.add_vertices(len(vertex_names))
    G.vs["name"] = vertex_names
    G.vs["type"] = [name_to_label[name] for name in vertex_names]

    # Color mapping for CTI entity types
    color_map = {
        'ACT': '#1f78b4', 'TOOL': '#33a02c', 'IDTY': '#ff7f00', 'TIME': '#cab2d6',
        'APT': '#e31a1c', 'VULID': '#ffff99', 'IP': '#fdbf6f',
        'URL': '#ff7f00', 'DOMAIN': '#b2df8a', 'FILE': '#fb9a99', 'HASH': '#a6cee3',
        'CVE': '#ffff99', 'OS': '#cab2d6', 'PROTOCOL': '#fdbf6f'
    }
    G.vs["color"] = [color_map.get(lab, '#a6cee3') for lab in G.vs["type"]]

    # Build simple directed edges
    edges = []
    relations = []
    for i in range(len(vertex_names) - 1):
        source = i
        target = i + 1
        edges.append((source, target))
        relations.append("related_to")

    G.add_edges(edges)
    G.es["label"] = relations
    G.es["color"] = "gray"

    return G


# --- PLOT GRAPH ---
def plot_cti_graph(G):
    """Visualize CTI graph using matplotlib."""
    layout = G.layout("fr")
    fig, ax = plt.subplots(figsize=(10, 8))

    ig.plot(
        G,
        target=ax,
        layout=layout,
        vertex_color=G.vs["color"],
        vertex_label=G.vs["name"],
        edge_color="gray",
        vertex_size=25,
        vertex_label_size=9,
        edge_arrow_size=0.3
    )

    ax.set_title("Cyber Threat Intelligence Knowledge Graph", fontsize=14)
    return fig
