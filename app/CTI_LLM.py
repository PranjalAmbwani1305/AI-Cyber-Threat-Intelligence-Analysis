import pdfplumber
import pytesseract
from PIL import Image
from transformers import pipeline
import networkx as nx
import matplotlib.pyplot as plt
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PyPDF2 import PdfReader

model_name = "CyberPeace-Institute/SecureBERT-NER"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

ner_pipeline = pipeline(
    "token-classification",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"   # groups sub-tokens into full entities
)


def extract_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, max_length=512, overlap=50):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), max_length - overlap):
        chunk = tokens[i:i+max_length]
        chunks.append(tokenizer.decode(chunk))
    return chunks

pdf_path = "/content/Innovant CyberSecurity Report.pdf"  # change if needed
pdf_text = extract_pdf_text(pdf_path)

pdf_text

chunks = chunk_text(pdf_text)

results = []
for idx, chunk in enumerate(chunks):
    print(f"Processing chunk {idx+1}/{len(chunks)}...")
    res = ner_pipeline(chunk)
    results.extend(res)

for r in results:  # preview first 50
    print(r)

print(f"\nTotal entities extracted: {len(results)}")

df = pd.DataFrame(results)
df = df[["word", "entity_group", "score"]]

df.head(20)

entities = df["word"].tolist()
labels = df["entity_group"].tolist()

G_cti = None

def build_cti_knowledge_graph(entities, labels):
    G = nx.Graph()

    # Add entity nodes
    for ent, lab in zip(entities, labels):
        G.add_node(ent, label=lab, color="skyblue", size=1500)

    # Connect nodes with relations based on rules
    for i in range(len(entities) - 1):
        e1, l1 = entities[i], labels[i]
        e2, l2 = entities[i+1], labels[i+1]

        relation = None
        if l1 == "THREAT_ACTOR" and l2 == "MALWARE":
            relation = "uses"
        elif l1 == "THREAT_ACTOR" and l2 == "ORG":
            relation = "targets"
        elif l1 == "THREAT_ACTOR" and l2 == "TOOL":
            relation = "uses"
        elif l1 == "MALWARE" and l2 == "FILE":
            relation = "drops"
        elif l1 == "MALWARE" and l2 in ["IP", "DOMAIN"]:
            relation = "connects_to"
        elif l1 == "IP" and l2 == "DOMAIN":
            relation = "resolves_to"
        elif l1 == "FILE" and l2 == "IP":
            relation = "communicates_with"
        else:
            relation = "related_to"

        G.add_edge(e1, e2, relation=relation)

    return G

def query_entity_graph(G, entity_name):
    """
    Show subgraph for a specific entity and its direct neighbors.
    """
    if entity_name not in G:
        print(f"⚠️ Entity '{entity_name}' not found in the graph.")
        return

    neighbors = list(G.neighbors(entity_name))
    sub_nodes = [entity_name] + neighbors
    subgraph = G.subgraph(sub_nodes)

    # Layout
    pos = nx.spring_layout(subgraph, k=0.6, seed=42)

    colors = [subgraph.nodes[n].get("color", "lightblue") for n in subgraph.nodes]
    sizes = [subgraph.nodes[n].get("size", 800) for n in subgraph.nodes]

    plt.figure(figsize=(10, 7))
    nx.draw(
        subgraph, pos,
        with_labels=True,
        node_color=colors,
        node_size=sizes,
        font_size=10,
        font_weight="bold",
        edge_color="gray"
    )

    edge_labels = nx.get_edge_attributes(subgraph, 'relation')
    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_size=9)

    plt.title(f"Knowledge Graph for '{entity_name}'", fontsize=14)
    plt.show()

G = build_cti_knowledge_graph(entities, labels)

print(G)

query_entity_graph(G, "phishing")