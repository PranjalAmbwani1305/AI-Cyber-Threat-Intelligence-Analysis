import pandas as pd
import nltk
import networkx as nx
import matplotlib.pyplot as plt
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from PyPDF2 import PdfReader
import io
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Ensure NLTK tokenizer is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
try:
    nltk.data.find("tokenizers/punkt_tab/english")
except LookupError:
    nltk.download("punkt_tab")

# --- Load Cybersecurity NER model (SecureBERT) ---
MODEL_NAME = "CyberPeace-Institute/SecureBERT-NER"

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
    ner_pipeline = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple"
    )
    return tokenizer, ner_pipeline


# --- File text extraction ---
def extract_text(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        return " ".join(df.astype(str).fillna("").values.flatten())

    elif uploaded_file.name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        return " ".join(
            [page.extract_text() for page in reader.pages if page.extract_text()]
        )

    elif uploaded_file.name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")

    else:
        return ""


# --- Sentence splitting ---
def split_into_sentences(text):
    return nltk.sent_tokenize(text)


# --- Text chunking for long files ---
def chunk_text(text, tokenizer, max_length=512, overlap=50):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), max_length - overlap):
        chunk = tokens[i:i + max_length]
        chunks.append(tokenizer.decode(chunk))
    return chunks


# --- Knowledge Graph Construction ---
def build_cti_graph(entities, labels):
    G = nx.DiGraph()

    color_map = {
        'ACT': '#1f78b4', 'TOOL': '#33a02c', 'IDTY': '#ff7f00',
        'APT': '#e31a1c', 'VULID': '#ffff99', 'IP': '#fdbf6f',
        'URL': '#ff7f00', 'DOMAIN': '#b2df8a', 'FILE': '#fb9a99',
        'HASH': '#a6cee3', 'CVE': '#ffff99', 'OS': '#cab2d6', 'PROTOCOL': '#fdbf6f'
    }

    for ent, lab in zip(entities, labels):
        clean_ent = str(ent).strip()
        if not clean_ent:
            continue
        G.add_node(clean_ent, color=color_map.get(lab, "#a6cee3"), type=lab)

    for i in range(len(entities) - 1):
        e1, l1 = entities[i], labels[i]
        e2, l2 = entities[i+1], labels[i+1]
        if not e1 or not e2:
            continue

        relation = "related_to"
        if l1 == "IDTY" and l2 == "ACT": relation = "performs"
        elif l1 == "ACT" and l2 == "TOOL": relation = "uses_tool"
        elif l1 == "APT" and l2 == "VULID": relation = "exploits"
        elif l1 == "MALWARE" and l2 in ["IP", "URL", "DOMAIN"]: relation = "communicates_with"
        elif l1 == "VULID" and l2 in ["OS", "TOOL"]: relation = "affects"

        G.add_edge(e1, e2, label=relation)

    return G


# --- Graph Plotting ---
def plot_cti_graph(G):
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    node_colors = [G.nodes[n]["color"] for n in G.nodes]
    nx.draw(
        G, pos, with_labels=True, node_color=node_colors,
        node_size=1500, font_size=8, font_color="black", edge_color="gray", arrows=True
    )

    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
    plt.title("Cyber Threat Intelligence Knowledge Graph", fontsize=14)
    plt.axis("off")
    return plt.gcf()
