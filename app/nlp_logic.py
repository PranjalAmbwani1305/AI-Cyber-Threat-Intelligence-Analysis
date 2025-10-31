# nlp_logic.py
import pandas as pd
import matplotlib.pyplot as plt
import igraph as ig
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from PyPDF2 import PdfReader
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

MODEL_NAME = "CyberPeace-Institute/SecureBERT-NER"

def load_model():
    """Load the SecureBERT-NER model for CTI entity recognition."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
    ner_pipeline = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple"
    )
    return tokenizer, ner_pipeline


def extract_pdf_text(uploaded_file):
    """Extract text from PDF file."""
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def chunk_text(text, tokenizer, max_length=512, overlap=50):
    """Split long text into overlapping chunks for BERT processing."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), max_length - overlap):
        chunk = tokens[i:i + max_length]
        chunks.append(tokenizer.decode(chunk))
    return chunks


def build_cti_graph(entities, labels):
    """Construct a cyber threat intelligence knowledge graph."""
    name_to_label = {}
    vertex_names = []

    for ent, lab in zip(entities, labels):
        clean_ent = ent.strip().replace('\n', ' ')
        if clean_ent and clean_ent not in name_to_label:
            name_to_label[clean_ent] = lab
            vertex_names.append(clean_ent)

    G = ig.Graph(directed=True)
    G.add_vertices(vertex_names)
    G.vs["name"] = vertex_names
    G.vs["type"] = [name_to_label[name] for name in vertex_names]
    G.vs["label"] = vertex_names

    color_map = {
        'APT': '#e31a1c', 'ACT': '#1f78b4', 'TOOL': '#33a02c', 'MALWARE': '#6a3d9a',
        'IP': '#fdbf6f', 'DOMAIN': '#ff7f00', 'URL': '#ff7f00',
        'CVE': '#ffff99', 'HASH': '#a6cee3', 'FILE': '#fb9a99',
        'VULID': '#cab2d6', 'OS': '#b2df8a', 'PROTOCOL': '#fdbf6f',
        'IDTY': '#ff7f00', 'TIME': '#cab2d6', 'MISC': '#999999'
    }
    G.vs["color"] = [color_map.get(lab, '#a6cee3') for lab in G.vs["type"]]

    def infer_relation(t1, t2):
        """Define relations between entity types."""
        if t1 == "APT" and t2 == "TOOL": return "uses_tool"
        if t1 == "APT" and t2 == "CVE": return "exploits_vulnerability"
        if t1 == "MALWARE" and t2 in ["IP", "DOMAIN", "URL"]: return "connects_to"
        if t1 == "MALWARE" and t2 == "FILE": return "drops_file"
        if t1 == "TOOL" and t2 == "PROTOCOL": return "communicates_via"
        if t1 == "CVE" and t2 == "OS": return "affects_system"
        if t1 == "APT" and t2 == "ACT": return "performs_attack"
        if t1 == "ACT" and t2 == "IDTY": return "targets_identity"
        return "related_to"

    edges, relations = [], []
    for i in range(len(entities) - 1):
        e1, e2 = entities[i].strip(), entities[i + 1].strip()
        t1, t2 = labels[i], labels[i + 1]
        if e1 in G.vs["name"] and e2 in G.vs["name"]:
            id1, id2 = G.vs.find(name=e1).index, G.vs.find(name=e2).index
            relation = infer_relation(t1, t2)
            edges.append((id1, id2))
            relations.append(relation)

    G.add_edges(edges)
    G.es["label"] = relations
    G.es["color"] = "gray"
    return G


def draw_subgraph(G, entity_name):
    """Generate 1-hop subgraph visualization."""
    center_idx = G.vs.find(name=entity_name).index
    neighbors = G.neighbors(center_idx, mode="all")
    subgraph = G.induced_subgraph([center_idx] + neighbors)
    layout = subgraph.layout("kamada_kawai")

    fig, ax = plt.subplots(figsize=(8, 6))
    ig.plot(
        subgraph,
        target=ax,
        layout=layout,
        vertex_color=subgraph.vs["color"],
        vertex_label=subgraph.vs["name"],
        edge_label=subgraph.es["label"],
        vertex_size=25,
        edge_color="gray",
        bbox=(800, 600),
        margin=50
    )
    plt.title(f"1-Hop Subgraph: {entity_name}", fontsize=14)
    return fig
