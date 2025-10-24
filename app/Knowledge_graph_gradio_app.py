import nltk
nltk.download('punkt_tab')

import gradio as gr
import pdfplumber
import pytesseract
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import igraph as ig
import matplotlib.pyplot as plt
import pandas as pd
from PyPDF2 import PdfReader
import warnings

# Suppress Hugging Face warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- GLOBAL MODEL/PIPELINE INITIALIZATION ---
# This is done once when the app starts for efficiency
MODEL_NAME = "CyberPeace-Institute/SecureBERT-NER"
MODEL_INITIALIZED = False
tokenizer = None
ner_pipeline = None

try:
    print("Attempting to load SecureBERT-NER Model...")
    # Using global is not needed here as we assign directly
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
    print(f"CRITICAL ERROR: Failed to load model or tokenizer. CTI functionality will be disabled.")
    print(f"Details: {e}")

# Global variables to store the generated graph and entities
GLOBAL_GRAPH = None
GLOBAL_ENTITIES_DF = None

# --- CORE UTILITY FUNCTIONS ---

def extract_pdf_text(pdf_path):
    """Extracts text from all pages of a PDF."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + " \n"
        return text
    except Exception as e:
        return f"Error reading PDF file: {type(e).__name__}: {str(e)}"

def chunk_text(text, max_length=512, overlap=50):
    """Tokenizes text and creates overlapping chunks for NER processing."""
    if not MODEL_INITIALIZED:
        return ["Model not loaded."]
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), max_length - overlap):
        chunk = tokens[i:i + max_length]
        chunks.append(tokenizer.decode(chunk))
    return chunks

# --- IGRAPH CONSTRUCTION LOGIC ---

def build_cti_knowledge_graph_igraph(entities, labels):
    """Constructs an iGraph graph using custom CTI rules for edge labeling."""
    name_to_original_label = {}
    vertex_names = []
    for ent, lab in zip(entities, labels):
        clean_ent = ent.replace('\n', ' ').strip()
        if clean_ent and clean_ent not in name_to_original_label:
            name_to_original_label[clean_ent] = lab
            vertex_names.append(clean_ent)

    G = ig.Graph(directed=True)
    G.add_vertices(len(vertex_names))
    G.vs["name"] = vertex_names
    G.vs["node_type"] = [name_to_original_label[name] for name in G.vs["name"]]
    G.vs["label"] = G.vs["name"]
    
    color_map = {
        'ACT': '#1f78b4', 'TOOL': '#33a02c', 'IDTY': '#ff7f00', 'TIME': '#cab2d6',
        'MISC': '#a6cee3', 'APT': '#e31a1c', 'VULID': '#ffff99', 'IP': '#fdbf6f',
        'URL': '#ff7f00', 'DOMAIN': '#b2df8a', 'FILE': '#fb9a99', 'HASH': '#a6cee3',
        'CVE': '#ffff99', 'OS': '#cab2d6', 'PROTOCOL': '#fdbf6f'
    }
    G.vs["color"] = [color_map.get(lab, '#a6cee3') for lab in G.vs["node_type"]]
    
    edges_to_add = []
    edge_relations = []
    cleaned_entities = [ent.replace('\n', ' ').strip() for ent in entities]
    
    for i in range(len(cleaned_entities) - 1):
        e1, l1 = cleaned_entities[i], labels[i]
        e2, l2 = cleaned_entities[i+1], labels[i+1]
        
        if not e1 or not e2 or e1 not in G.vs["name"] or e2 not in G.vs["name"]:
            continue
            
        id1 = G.vs.find(name=e1).index
        id2 = G.vs.find(name=e2).index
        
        relation = "related_to" # Default relation
        if l1 == "IDTY" and l2 == "ACT": relation = "performs_ttp"
        elif l1 == "ACT" and l2 == "TOOL": relation = "targets_platform"
        elif l1 == "APT" and l2 == "MALWARE": relation = "uses"
        elif l1 == "MALWARE" and l2 in ["IP", "URL", "DOMAIN", "FILE", "HASH"]: relation = "uses_indicator"
        elif l1 == "VULID" and l2 in ["OS", "TOOL"]: relation = "affects"
        
        edges_to_add.append((id1, id2))
        edge_relations.append(relation)

    G.add_edges(edges_to_add)
    G.es["label"] = edge_relations
    G.es["color"] = "gray"
    return G

# --- VISUALIZATION FUNCTION ---

def query_entity_graph_igraph(G, entity_name):
    """Generates a 1-hop subgraph plot for the selected entity."""
    if G is None or entity_name is None:
        return None, "Please process a report first and select an entity."
        
    clean_name = entity_name.replace('\n', ' ').strip()
    if clean_name not in G.vs["name"]:
        return None, f"Entity '{clean_name}' not found or has no connections."

    try:
        center_vid = G.vs.find(name=clean_name).index
        neighbor_vids = G.neighbors(center_vid, mode="all")
        subgraph = G.induced_subgraph(list(set([center_vid] + neighbor_vids)))

        layout = subgraph.layout("kamada_kawai")
        visual_style = {
            "vertex_label": subgraph.vs["name"], "vertex_color": subgraph.vs["color"],
            "edge_label": subgraph.es["label"], "edge_color": "gray", "vertex_size": 25,
            "vertex_label_size": 10, "edge_label_size": 9, "bbox": (800, 600), "margin": 50
        }

        fig, ax = plt.subplots(figsize=(10, 8))
        ig.plot(subgraph, target=ax, layout=layout, **visual_style)
        ax.set_title(f"Knowledge Graph: 1-Hop Neighbors of '{clean_name}'", fontsize=14)
        return fig, f"Successfully mapped {subgraph.vcount()} connections for '{clean_name}'."
    except Exception as e:
        plt.close('all')
        return None, f"Error generating subgraph: {e}"

# --- GRADIO INTERFACE LOGIC ---

def process_cti_report(file_obj):
    """Main processing function: extracts text, runs NER, and builds the graph."""
    global GLOBAL_GRAPH, GLOBAL_ENTITIES_DF

    if file_obj is None:
        return pd.DataFrame(), gr.Dropdown(), None, "", "Please upload a PDF file."
    
    if not MODEL_INITIALIZED:
        return pd.DataFrame(), gr.Dropdown(), None, "", "CRITICAL: NER Model failed to load."

    text = extract_pdf_text(file_obj.name)
    if text.startswith("Error"):
        return pd.DataFrame(), gr.Dropdown(), None, "", text

    chunks = chunk_text(text)
    results = [res for chunk in chunks for res in ner_pipeline(chunk)]
    
    if not results:
        return pd.DataFrame(), gr.Dropdown(), None, "", "NER returned no entities."

    df = pd.DataFrame(results).rename(columns={'word': 'Entity', 'entity_group': 'Type'})
    df['Score'] = df['score'].round(4)
    df_display = df[['Entity', 'Type', 'Score']].copy()
    
    try:
        G = build_cti_knowledge_graph_igraph(df["Entity"].tolist(), df["Type"].tolist())
        GLOBAL_GRAPH = G
        GLOBAL_ENTITIES_DF = df_display
        unique_entity_names = G.vs["name"]
        status = f"Successfully processed. Extracted {G.vcount()} unique entities."
        return df_display, gr.Dropdown(choices=unique_entity_names, value=None), None, "", status
    except Exception as e:
        return df_display, gr.Dropdown(), None, "", f"Error building graph: {e}"

def update_subgraph_wrapper(entity_name):
    """Wrapper to pass the global graph to the plotting function."""
    return query_entity_graph_igraph(GLOBAL_GRAPH, entity_name)

# --- GRADIO INTERFACE LAYOUT ---
def create_app():
    with gr.Blocks(title="CTI Knowledge Graph Builder") as app:
        gr.Markdown("# Cyber Threat Intelligence (CTI) Knowledge Graph Analyzer")
        gr.Markdown("Upload a CTI report (PDF) to extract entities and visualize relationships.")

        with gr.Row():
            file_input = gr.File(label="Upload CTI Report (PDF)", file_types=[".pdf"])
            process_button = gr.Button("Process Report", variant="primary")
            status_output = gr.Textbox(label="Status", interactive=False)

        gr.Markdown("---")
        gr.Markdown("## Step 1: Extracted Entities (NER Results)")
        entity_table_output = gr.DataFrame(headers=["Entity", "Type", "Score"], label="Extracted Entities")

        gr.Markdown("---")
        gr.Markdown("## Step 2: Visualize Knowledge Graph Subgraph")
        with gr.Row():
            entity_dropdown = gr.Dropdown(label="Select an Entity to Query", choices=[], interactive=True, scale=2)
            query_button = gr.Button("Show Subgraph", scale=1)
        graph_output = gr.Plot(label="1-Hop Knowledge Graph Subgraph", visible=True)
        graph_status = gr.Textbox(label="Graph Status", interactive=False)

        # --- EVENT HANDLERS ---
        process_button.click(
            fn=process_cti_report,
            inputs=[file_input],
            outputs=[entity_table_output, entity_dropdown, graph_output, graph_status, status_output]
        )
        query_button.click(
            fn=update_subgraph_wrapper,
            inputs=[entity_dropdown],
            outputs=[graph_output, graph_status]
        )
        entity_dropdown.select(
            fn=update_subgraph_wrapper,
            inputs=[entity_dropdown],
            outputs=[graph_output, graph_status]
        )
    return app

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    app = create_app()
    app.launch()