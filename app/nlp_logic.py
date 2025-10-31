import pandas as pd
import networkx as nx
from pyvis.network import Network
import PyPDF2
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

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

# -------------------- BUILD STRUCTURED CTI GRAPH --------------------
def build_structured_cti_graph():
    """
    Builds a predefined Cyber Threat Intelligence flow like:
    Firewall → Destination IP → Protocol → Alert
    """
    G = nx.DiGraph()

    # Nodes
    entities = {
        "Firewall": "Security Device",
        "Source IP": "Network Node",
        "Destination IP": "Network Node",
        "Protocol": "Network Protocol",
        "Alert": "Threat Event",
        "User": "Human Actor"
    }

    for node, node_type in entities.items():
        G.add_node(node, type=node_type)

    # Relationships
    relationships = [
        ("Firewall", "Destination IP", "blocks"),
        ("Firewall", "Source IP", "monitors"),
        ("Source IP", "User", "triggers"),
        ("Destination IP", "Protocol", "uses"),
        ("Protocol", "Alert", "uses"),
        ("User", "Alert", "triggers"),
    ]

    for src, dst, rel in relationships:
        G.add_edge(src, dst, label=rel)

    return G

# -------------------- PLOT GRAPH USING PYVIS --------------------
def plot_cti_graph_pyvis(G, output_html="cti_graph.html"):
    net = Network(height="700px", width="100%", directed=True, bgcolor="#0e1117", font_color="white")
    net.repulsion(node_distance=180, spring_length=200)

    color_map = {
        "Security Device": "#ff7f0e",
        "Network Node": "#1f77b4",
        "Network Protocol": "#2ca02c",
        "Threat Event": "#d62728",
        "Human Actor": "#9467bd",
    }

    for node, data in G.nodes(data=True):
        node_type = data.get("type", "Unknown")
        net.add_node(node, label=node, color=color_map.get(node_type, "#7f7f7f"), shape="box")

    for src, dst, data in G.edges(data=True):
        label = data.get("label", "")
        net.add_edge(src, dst, label=label, color="gray")

    net.save_graph(output_html)
    return output_html
