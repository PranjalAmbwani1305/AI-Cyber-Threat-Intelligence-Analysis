import pandas as pd
import igraph as ig
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import PyPDF2

# -------------------- MODEL LOADING --------------------
def load_model():
    MODEL_NAME = "CyberPeace-Institute/SecureBERT-NER"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
    ner_pipeline = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple"
    )
    return tokenizer, ner_pipeline

# -------------------- TEXT EXTRACTION --------------------
def extract_text(file):
    """Extract text from CSV, TXT, or PDF files."""
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
        return " ".join(df.astype(str).fillna("").values.flatten())
    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8", errors="ignore")
    elif file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        text = " ".join(page.extract_text() for page in reader.pages if page.extract_text())
        return text
    return ""

# -------------------- CHUNKING --------------------
def chunk_text(text, tokenizer, max_length=512, overlap=50):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), max_length - overlap):
        chunk = tokens[i:i + max_length]
        chunks.append(tokenizer.decode(chunk))
    return chunks

# -------------------- BUILD CYBER GRAPH --------------------
def build_cti_graph(entities, labels):
    """Build a domain-specific CTI graph with realistic cyber relations."""
    G = ig.Graph(directed=True)
    node_map = {}

    for ent, lbl in zip(entities, labels):
        clean_ent = ent.strip()
        if clean_ent and clean_ent not in node_map:
            node_map[clean_ent] = lbl

    G.add_vertices(list(node_map.keys()))
    G.vs["type"] = [node_map[n] for n in G.vs["name"]]

    # Define cyber threat relationships (real-world inspired)
    relations = {
        "APT": ["TOOL", "DOMAIN", "IP", "VULID"],
        "TOOL": ["CVE", "FILE", "DOMAIN"],
        "CVE": ["OS", "VULID"],
        "DOMAIN": ["IP", "URL"],
        "IP": ["URL", "FILE"],
        "FILE": ["HASH"],
        "HASH": ["VULID"],
    }

    edges = []
    for src, src_type in zip(G.vs["name"], G.vs["type"]):
        for dst, dst_type in zip(G.vs["name"], G.vs["type"]):
            if dst != src and src_type in relations and dst_type in relations[src_type]:
                edges.append((src, dst))

    G.add_edges(edges)
    G.es["label"] = ["related_to"] * len(edges)

    # Visual appearance
    color_map = {
        "APT": "#e31a1c", "TOOL": "#33a02c", "CVE": "#ff7f00", "VULID": "#cab2d6",
        "DOMAIN": "#1f78b4", "IP": "#a6cee3", "URL": "#b2df8a", "FILE": "#fb9a99",
        "HASH": "#fdbf6f", "OS": "#cab2d6", "PROTOCOL": "#ffff99"
    }
    G.vs["color"] = [color_map.get(t, "#9e9e9e") for t in G.vs["type"]]

    return G

# -------------------- MATPLOTLIB PLOT --------------------
def plot_cti_graph(G):
    layout = G.layout("fruchterman_reingold")
    fig, ax = plt.subplots(figsize=(12, 10))
    ig.plot(
        G,
        target=ax,
        layout=layout,
        vertex_color=G.vs["color"],
        vertex_label=G.vs["name"],
        vertex_label_size=8,
        vertex_size=25,
        edge_arrow_size=0.4,
        edge_color="gray",
        margin=60
    )
    ax.set_title("Cyber Threat Intelligence Knowledge Graph", fontsize=14, pad=20)
    ax.axis("off")
    return fig
