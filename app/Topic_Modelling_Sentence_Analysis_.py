import nltk
nltk.download('punkt_tab')



# --- NLP Topic Modelling and Linguistic Analysis Imports ---
import spacy
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px


!pip install transformers torch pdfplumber pytesseract pillow matplotlib PyPDF2 python-igraph cairocffi gradio nltk

import gradio as gr
import pdfplumber
import pytesseract
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import igraph as ig
import matplotlib.pyplot as plt
import pandas as pd
from PyPDF2 import PdfReader
import warnings
import spacy
from bertopic import BERTopic
from spacy import displacy

# --- SUPPRESS WARNINGS ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- MODEL CONFIGURATION ---
MODEL_NAME = "CyberPeace-Institute/SecureBERT-NER"
MODEL_INITIALIZED = False
tokenizer = None
ner_pipeline = None

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# --- LOAD NER MODEL ---
try:
    print("Attempting to load SecureBERT-NER Model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
    ner_pipeline = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple"
    )
    print("Model loaded successfully.")
    MODEL_INITIALIZED = True
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load model. Details: {e}")

# --- PDF TEXT EXTRACTION ---
def extract_pdf_text(pdf_path):
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

# --- TEXT CHUNKING ---
def chunk_text(text, max_length=512, overlap=50):
    if not MODEL_INITIALIZED:
        return ["Model not loaded."]
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), max_length - overlap):
        chunk = tokens[i:i + max_length]
        chunks.append(tokenizer.decode(chunk))
    return chunks

# --- BUILD KNOWLEDGE GRAPH ---
def build_cti_knowledge_graph_igraph(entities, labels):
    name_to_label = {}
    vertex_names = []
    for ent, lab in zip(entities, labels):
        ent = ent.replace('\n', ' ').strip()
        if ent and ent not in name_to_label:
            name_to_label[ent] = lab
            vertex_names.append(ent)

    G = ig.Graph(directed=True)
    G.add_vertices(len(vertex_names))
    G.vs["name"] = vertex_names
    G.vs["node_type"] = [name_to_label[name] for name in G.vs["name"]]
    G.vs["label"] = G.vs["name"]

    color_map = {
        'ACT': '#1f78b4', 'TOOL': '#33a02c', 'IDTY': '#ff7f00', 'TIME': '#cab2d6',
        'MISC': '#a6cee3', 'APT': '#e31a1c', 'VULID': '#ffff99', 'IP': '#fdbf6f',
        'URL': '#ff7f00', 'DOMAIN': '#b2df8a', 'FILE': '#fb9a99', 'HASH': '#a6cee3',
        'CVE': '#ffff99', 'OS': '#cab2d6', 'PROTOCOL': '#fdbf6f'
    }
    G.vs["color"] = [color_map.get(lab, '#a6cee3') for lab in G.vs["node_type"]]

    edges, relations = [], []
    for i in range(len(entities) - 1):
        e1, l1 = entities[i].strip(), labels[i]
        e2, l2 = entities[i+1].strip(), labels[i+1]
        if e1 in G.vs["name"] and e2 in G.vs["name"]:
            id1, id2 = G.vs.find(name=e1).index, G.vs.find(name=e2).index
            relation = "related_to"
            if l1 == "IDTY" and l2 == "ACT": relation = "performs_ttp"
            elif l1 == "ACT" and l2 == "TOOL": relation = "targets_platform"
            elif l1 == "APT" and l2 == "MALWARE": relation = "uses"
            elif l1 == "MALWARE" and l2 in ["IP", "URL", "DOMAIN", "FILE", "HASH"]: relation = "uses_indicator"
            elif l1 == "VULID" and l2 in ["OS", "TOOL"]: relation = "affects"
            edges.append((id1, id2))
            relations.append(relation)

    G.add_edges(edges)
    G.es["label"] = relations
    G.es["color"] = "gray"
    return G

# --- SUBGRAPH VISUALIZATION ---
def query_entity_graph_igraph(G, entity_name):
    if G is None or entity_name is None:
        return None, "Please process a report first."
    if entity_name not in G.vs["name"]:
        return None, f"Entity '{entity_name}' not found."

    center = G.vs.find(name=entity_name).index
    neighbors = G.neighbors(center, mode="all")
    subgraph = G.induced_subgraph(list(set([center] + neighbors)))

    layout = subgraph.layout("kamada_kawai")
    fig, ax = plt.subplots(figsize=(10, 8))
    ig.plot(subgraph, target=ax, layout=layout,
            vertex_label=subgraph.vs["name"],
            vertex_color=subgraph.vs["color"],
            edge_label=subgraph.es["label"],
            edge_color="gray",
            vertex_size=25,
            bbox=(800, 600))
    ax.set_title(f"1-Hop Graph for '{entity_name}'", fontsize=14)
    return fig, f"Mapped {subgraph.vcount()} nodes."

# --- PROCESS PDF FOR NER + GRAPH ---
GLOBAL_GRAPH = None
def process_cti_report(file_obj):
    global GLOBAL_GRAPH
    if file_obj is None:
        return pd.DataFrame(), gr.Dropdown(), None, "", "Please upload a PDF."
    if not MODEL_INITIALIZED:
        return pd.DataFrame(), gr.Dropdown(), None, "", "NER model not loaded."

    text = extract_pdf_text(file_obj.name)
    chunks = chunk_text(text)
    results = [r for chunk in chunks for r in ner_pipeline(chunk)]

    if not results:
        return pd.DataFrame(), gr.Dropdown(), None, "", "No entities found."

    df = pd.DataFrame(results).rename(columns={'word': 'Entity', 'entity_group': 'Type'})
    df['Score'] = df['score'].round(4)
    G = build_cti_knowledge_graph_igraph(df["Entity"].tolist(), df["Type"].tolist())
    GLOBAL_GRAPH = G
    unique_entities = G.vs["name"]
    status = f"Processed {G.vcount()} unique entities."
    return df[['Entity', 'Type', 'Score']], gr.Dropdown(choices=unique_entities), None, "", status

# --- TOPIC MODELING (BERTopic) ---
def topic_modeling_bertopic(text_input):
    texts = [t.strip() for t in text_input.split("\n") if t.strip()]
    if not texts:
        return "Please enter some text."
    topic_model = BERTopic(verbose=False)
    topics, probs = topic_model.fit_transform(texts)
    return topic_model.visualize_barchart(top_n_topics=5)

# --- LINGUISTIC ANALYSIS (spaCy) ---
def linguistic_analysis_spacy(text):
    if not text.strip():
        return [], "<p>Please enter text for analysis.</p>"
    doc = nlp(text)
    pos_tags = [(t.text, t.pos_, t.dep_) for t in doc]
    html = displacy.render(doc, style="dep", page=True)
    return pos_tags, html

# --- GRADIO UI ---
def create_app():
    with gr.Blocks(title="CTI Knowledge Graph + NLP") as app:
        gr.Markdown("# 🧠 Cyber Threat Intelligence Analyzer")
        gr.Markdown("Upload a CTI report or paste text to explore entities, topics, and syntax.")

        with gr.Tab("Knowledge Graph"):
            file_input = gr.File(label="Upload PDF Report", file_types=[".pdf"])
            process_button = gr.Button("Extract Entities")
            entity_table = gr.DataFrame(headers=["Entity", "Type", "Score"])
            entity_dropdown = gr.Dropdown(label="Select Entity", choices=[])
            query_button = gr.Button("Show Subgraph")
            graph_output = gr.Plot(label="Knowledge Graph")
            graph_status = gr.Textbox(label="Status", interactive=False)

            process_button.click(process_cti_report,
                inputs=[file_input],
                outputs=[entity_table, entity_dropdown, graph_output, graph_status, graph_status])
            query_button.click(query_entity_graph_igraph,
                inputs=[gr.State(GLOBAL_GRAPH), entity_dropdown],
                outputs=[graph_output, graph_status])

        with gr.Tab("Topic Modeling"):
            gr.Markdown("### Topic Discovery using BERTopic")
            topic_input = gr.Textbox(label="Enter multiple reports (one per line)", lines=5)
            topic_button = gr.Button("Run Topic Modeling")
            topic_output = gr.Plot(label="Topic Visualization")
            topic_button.click(topic_modeling_bertopic, inputs=[topic_input], outputs=[topic_output])

        with gr.Tab("Linguistic Analysis"):
            gr.Markdown("### POS Tagging & Dependency Parsing using spaCy")
            ling_input = gr.Textbox(label="Enter text", lines=5)
            ling_button = gr.Button("Analyze Syntax")
            pos_output = gr.DataFrame(headers=["Token", "POS", "Dependency"])
            dep_output = gr.HTML(label="Dependency Visualization")
            ling_button.click(linguistic_analysis_spacy, inputs=[ling_input], outputs=[pos_output, dep_output])
    return app

if __name__ == "__main__":
    app = create_app()
    app.launch()
