import io
import re
import networkx as nx
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, pipeline
from PyPDF2 import PdfReader


# -------------------------------
# LOAD TRANSFORMER MODEL
# -------------------------------
def load_model():
    """
    Loads an NLP model and tokenizer for cyber entity recognition.
    """
    model_name = "dslim/bert-base-NER"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ner_pipeline = pipeline("ner", model=model_name, tokenizer=tokenizer, aggregation_strategy="simple")
    return tokenizer, ner_pipeline


# -------------------------------
# TEXT EXTRACTION FROM FILE
# -------------------------------
def extract_text(uploaded_file):
    """
    Extracts text from PDF or plain text file.
    """
    text = ""

    if uploaded_file.name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            text += page.extract_text() or ""
    elif uploaded_file.name.endswith(".txt"):
        text = uploaded_file.read().decode("utf-8", errors="ignore")
    elif uploaded_file.name.endswith(".csv"):
        text = uploaded_file.read().decode("utf-8", errors="ignore")
    else:
        text = ""

    return text.strip()


# -------------------------------
# TEXT CHUNKING
# -------------------------------
def chunk_text(text, tokenizer, max_tokens=400):
    """
    Splits long text into smaller chunks for model processing.
    """
    tokens = tokenizer.tokenize(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokenizer.convert_tokens_to_string(tokens[i:i + max_tokens])
        chunks.append(chunk)
    return chunks


# -------------------------------
# KNOWLEDGE GRAPH BUILDING
# -------------------------------
def build_cti_graph(entities, types):
    """
    Builds a simple knowledge graph linking related cyber entities.
    """
    G = nx.DiGraph()

    # Add nodes with types
    for entity, etype in zip(entities, types):
        if entity.strip():
            G.add_node(entity, type=etype)

    # Add heuristic relationships
    for i in range(len(entities) - 1):
        src, dst = entities[i], entities[i + 1]
        if src != dst:
            G.add_edge(src, dst, relation="related_to")

    return G


# -------------------------------
# GRAPH VISUALIZATION
# -------------------------------
def plot_cti_graph(G):
    """
    Plots the CTI knowledge graph with colors by entity type.
    """
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G, k=0.5, seed=42)

    # Node colors by type
    color_map = []
    for node in G.nodes(data=True):
        ntype = node[1].get("type", "")
        if "IP" in ntype:
            color_map.append("lightblue")
        elif "Malware" in ntype:
            color_map.append("salmon")
        elif "Domain" in ntype:
            color_map.append("lightgreen")
        else:
            color_map.append("gray")

    nx.draw(G, pos, with_labels=True, node_color=color_map,
            node_size=2000, font_size=8, font_weight="bold", edge_color="gray",
            arrowsize=15, connectionstyle="arc3,rad=0.1")

    # Edge labels
    edge_labels = nx.get_edge_attributes(G, 'relation')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    plt.title("Cyber Threat Intelligence Knowledge Graph", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return plt
