import nltk
nltk.download('punkt_tab')

import gradio as gr
from PyPDF2 import PdfReader
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import igraph as ig
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
import nltk
import re

# Suppress warnings for a cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- GLOBAL MODEL/PIPELINE INITIALIZATION ---

# 1. NER Model for Knowledge Graph
MODEL_NAME = "CyberPeace-Institute/SecureBERT-NER"
NER_MODEL_INITIALIZED = False
ner_tokenizer = None
ner_pipeline = None

try:
    print("Attempting to load SecureBERT-NER Model...")
    ner_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ner_model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
    ner_pipeline = pipeline(
        "token-classification",
        model=ner_model,
        tokenizer=ner_tokenizer,
        aggregation_strategy="simple"
    )
    print("NER Model loaded successfully.")
    NER_MODEL_INITIALIZED = True
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load NER model. Knowledge Graph functionality will be disabled.")
    print(f"Details: {e}")

# 2. Sentence Embedding Model for Clustering
try:
    print("Attempting to load Sentence Transformer Model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Sentence Transformer Model loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load Sentence Transformer model. Clustering functionality will be disabled.")
    print(f"Details: {e}")

# 3. NLTK Tokenizer for Sentence Splitting
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK 'punkt' model...")
    nltk.download('punkt')

# --- CORE UTILITY FUNCTIONS ---

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
        return f"Error reading PDF file: {type(e).__name__}: {str(e)}"

def chunk_text(text, max_length=512, overlap=50):
    if not NER_MODEL_INITIALIZED: return ["Model not loaded."]
    tokens = ner_tokenizer.encode(text, add_special_tokens=False)
    chunks = [ner_tokenizer.decode(tokens[i:i + max_length]) for i in range(0, len(tokens), max_length - overlap)]
    return chunks

def split_into_sentences(text):
    sentences = nltk.sent_tokenize(text)
    sentences = [re.sub(r'\n', ' ', s).strip() for s in sentences]
    return [s for s in sentences if s]

# --- KNOWLEDGE GRAPH FUNCTIONS ---

def build_cti_knowledge_graph_igraph(entities, labels):
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
    
    color_map = {'ACT':'#1f78b4','TOOL':'#33a02c','IDTY':'#ff7f00','TIME':'#cab2d6','MISC':'#a6cee3','APT':'#e31a1c','VULID':'#ffff99','IP':'#fdbf6f','URL':'#ff7f00','DOMAIN':'#b2df8a','FILE':'#fb9a99','HASH':'#a6cee3','CVE':'#ffff99','OS':'#cab2d6','PROTOCOL':'#fdbf6f'}
    G.vs["color"] = [color_map.get(lab, '#a6cee3') for lab in G.vs["node_type"]]
    
    edges_to_add = []
    edge_relations = []
    cleaned_entities = [ent.replace('\n', ' ').strip() for ent in entities]
    
    for i in range(len(cleaned_entities) - 1):
        e1, l1 = cleaned_entities[i], labels[i]
        e2, l2 = cleaned_entities[i+1], labels[i+1]
        if not e1 or not e2 or e1 not in G.vs["name"] or e2 not in G.vs["name"]: continue
        
        id1 = G.vs.find(name=e1).index
        id2 = G.vs.find(name=e2).index
        
        relation = "related_to"
        if l1 == "IDTY" and l2 == "ACT": relation = "performs_ttp"
        elif l1 == "ACT" and l2 == "TOOL": relation = "uses_tool"
        elif l1 == "APT" and l2 == "MALWARE": relation = "uses_malware"
        elif l1 == "MALWARE" and l2 in ["IP", "URL", "DOMAIN"]: relation = "communicates_with"
        elif l1 == "VULID" and l2 in ["OS", "TOOL"]: relation = "affects"
        
        edges_to_add.append((id1, id2))
        edge_relations.append(relation)
        
    G.add_edges(edges_to_add)
    G.es["label"] = edge_relations
    return G

def query_entity_graph_igraph(G, entity_name):
    if G is None or entity_name is None: return None, "Graph not generated. Please process a report first."
    clean_name = entity_name.replace('\n', ' ').strip()
    if clean_name not in G.vs["name"]: return None, f"Entity '{clean_name}' not found."
        
    try:
        center_vid = G.vs.find(name=clean_name).index
        neighbor_vids = G.neighbors(center_vid, mode="all")
        subgraph = G.induced_subgraph(list(set([center_vid] + neighbor_vids)))
        if not subgraph.vs: return None, f"Entity '{clean_name}' has no connections to plot."

        layout = subgraph.layout("kamada_kawai")
        visual_style = {"vertex_label": subgraph.vs["name"], "vertex_color": subgraph.vs["color"], "edge_label": subgraph.es["label"], "edge_color": "gray", "vertex_size": 25, "vertex_label_size": 10, "edge_label_size": 9, "bbox": (800, 600), "margin": 50}
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ig.plot(subgraph, target=ax, layout=layout, **visual_style)
        ax.set_title(f"Knowledge Graph: 1-Hop Neighbors of '{clean_name}'")
        return fig, f"Successfully mapped {subgraph.vcount()} connections."
    except Exception as e:
        plt.close('all')
        return None, f"Error generating subgraph: {e}"

# --- SEMANTIC CLUSTERING FUNCTIONS ---

def get_cluster_topic_names(sentences, cluster_assignments):
    """Generates a topic name for each cluster using TF-IDF keywords."""
    clustered_sentences = {i: [] for i in set(cluster_assignments)}
    for sentence, cluster_id in zip(sentences, cluster_assignments):
        clustered_sentences[cluster_id].append(sentence)

    topic_names = {}
    for cluster_id, docs in clustered_sentences.items():
        if cluster_id == -1:
            topic_names[cluster_id] = "Outliers / Miscellaneous"
            continue
        
        try:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=3, ngram_range=(1, 2))
            corpus = [" ".join(docs)]
            vectorizer.fit(corpus)
            feature_names = vectorizer.get_feature_names_out()
            topic_names[cluster_id] = ", ".join(feature_names)
        except ValueError: # This handles the "empty vocabulary" error
            topic_names[cluster_id] = "Short / Common Phrases"
            
    return topic_names

def perform_clustering(sentences):
    """Generates embeddings and clusters them using DBSCAN."""
    if not sentences: return None, None, None, "No sentences to cluster."
    
    embeddings = embedding_model.encode(sentences)
    dbscan = DBSCAN(eps=1.0, min_samples=2)
    dbscan.fit(embeddings)
    cluster_assignments = dbscan.labels_
    topic_names = get_cluster_topic_names(sentences, cluster_assignments)
    return embeddings, cluster_assignments, topic_names, f"Successfully clustered {len(sentences)} sentences."

def create_cluster_plot(embeddings, cluster_assignments, topic_names):
    """Creates a 2D scatter plot of the clusters with topic names."""
    if embeddings is None: return None
    
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    unique_labels = sorted(set(cluster_assignments))
    colors = [plt.cm.viridis(each) for each in np.linspace(0, 1, len(unique_labels))]
    
    for k, col in zip(unique_labels, colors):
        label = topic_names.get(k, "Unknown")
        if k == -1: col = [0, 0, 0, 1]
        
        class_member_mask = (cluster_assignments == k)
        xy = reduced_embeddings[class_member_mask]
        
        ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=14 if k != -1 else 7, label=label)
                
    ax.set_title("Semantic Topic Clusters from PDF Document")
    ax.legend(title="Topics")
    return fig

# --- GRADIO WORKFLOW FUNCTIONS ---

def unified_process_report(file_obj):
    """Single function to process the PDF for both features."""
    initial_df = pd.DataFrame(columns=['Entity', 'Type', 'Score'])
    initial_dropdown = gr.Dropdown(choices=[], value=None)

    if file_obj is None: return initial_df, initial_dropdown, "Please upload a PDF file.", [], None
    if not NER_MODEL_INITIALIZED: return initial_df, initial_dropdown, "CRITICAL: NER Model failed to load.", [], None

    text = extract_pdf_text(file_obj.name)
    if text.startswith("Error"): return initial_df, initial_dropdown, text, [], None

    sentences = split_into_sentences(text)
    chunks = chunk_text(text)
    results = [res for chunk in chunks for res in ner_pipeline(chunk)]

    if not results:
        status = "NER found no entities. Clustering is still available."
        return initial_df, initial_dropdown, status, sentences, None

    df = pd.DataFrame(results).rename(columns={'word': 'Entity', 'entity_group': 'Type'})
    df['Score'] = df['score'].round(4)
    df_display = df[['Entity', 'Type', 'Score']].copy()

    try:
        G = build_cti_knowledge_graph_igraph(df["Entity"].tolist(), df["Type"].tolist())
        unique_entity_names = G.vs["name"]
        final_status = f"Processed. Found {len(sentences)} sentences and {G.vcount()} unique entities."
        return df_display, gr.Dropdown(choices=unique_entity_names, value=None), final_status, sentences, G
    except Exception as e:
        return initial_df, initial_dropdown, f"Error building graph: {e}", sentences, None

def run_clustering_workflow(sentences):
    embeddings, labels, topics, status = perform_clustering(sentences)
    plot = create_cluster_plot(embeddings, labels, topics)
    return plot, status

# --- GRADIO INTERFACE LAYOUT ---

with gr.Blocks(title="CTI Analysis Tool", theme=gr.themes.Soft()) as app:
    gr.Markdown("# Cyber Threat Intelligence (CTI) Analysis Tool")
    gr.Markdown("Upload a CTI report (PDF) to analyze entities and semantic topics.")
    
    sentences_state = gr.State([])
    graph_state = gr.State(None)

    with gr.Row():
        file_input = gr.File(label="Upload CTI Report (PDF)", file_types=[".pdf"])
        process_button = gr.Button("Process Report", variant="primary")
    status_output = gr.Textbox(label="Processing Status", interactive=False)

    with gr.Tabs():
        with gr.TabItem("Knowledge Graph Analyzer"):
            gr.Markdown("### Visualize Entities and Their Relationships")
            entity_table_output = gr.DataFrame(headers=["Entity", "Type", "Score"], label="Extracted Entities")
            with gr.Row():
                entity_dropdown = gr.Dropdown(label="Select an Entity to Query", choices=[], interactive=True, scale=2)
                query_button = gr.Button("Show Subgraph", scale=1)
            graph_output = gr.Plot(label="1-Hop Knowledge Graph Subgraph")
            graph_status = gr.Textbox(label="Graph Status", interactive=False)

        with gr.TabItem("Semantic Topic Clustering"):
            gr.Markdown("### Group Sentences by Semantic Meaning")
            cluster_button = gr.Button("Cluster Sentences", variant="secondary")
            cluster_plot_output = gr.Plot(label="Sentence Cluster Visualization")
            cluster_status = gr.Textbox(label="Clustering Status", interactive=False)

    # --- EVENT HANDLERS ---
    process_button.click(
        fn=unified_process_report,
        inputs=[file_input],
        outputs=[entity_table_output, entity_dropdown, status_output, sentences_state, graph_state]
    )

    query_button.click(
        fn=query_entity_graph_igraph,
        inputs=[graph_state, entity_dropdown],
        outputs=[graph_output, graph_status]
    )
    
    entity_dropdown.select(
        fn=query_entity_graph_igraph,
        inputs=[graph_state, entity_dropdown],
        outputs=[graph_output, graph_status]
    )

    cluster_button.click(
        fn=run_clustering_workflow,
        inputs=[sentences_state],
        outputs=[cluster_plot_output, cluster_status]
    )

app.launch(debug=True)