import streamlit as st
import pandas as pd
import numpy as np
import warnings
import re
import nltk
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime

# External libraries that require installation:
# pip install streamlit transformers PyPDF2 sentence-transformers scikit-learn python-igraph matplotlib pandas numpy torch spacy bertopic plotly
# python -m spacy download en_core_web_sm
try:
    from PyPDF2 import PdfReader
except ImportError:
    st.error("Please install PyPDF2: `pip install PyPDF2`")

try:
    import torch
    from transformers import (
        pipeline,
        AutoTokenizer,
        AutoModelForTokenClassification,
        AutoModelForSequenceClassification
    )
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import DBSCAN
    from sklearn.decomposition import PCA
    from sklearn.feature_extraction.text import TfidfVectorizer
    import igraph as ig
    import spacy
    from spacy import displacy
    import spacy.cli
    from bertopic import BERTopic
    import streamlit.components.v1 as components
except ImportError:
    st.error("A critical NLP/ML library is missing. Please ensure all required packages are installed.")


# --- CONFIGURATION AND CACHE SETUP ---
st.set_page_config(layout="wide", page_title="CTI & NLP Dashboard")

# Suppress warnings for a cleaner Streamlit app
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Download NLTK resources if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- MODEL LOADING (Uses Streamlit's cache for efficiency) ---

@st.cache_resource
def load_cti_models():
    """Loads all heavy models and pipelines, caching them globally."""
    
    # 1. NER Model (SecureBERT-NER for CTI Entities)
    MODEL_NAME = "CyberPeace-Institute/SecureBERT-NER"
    try:
        ner_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        ner_model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
        ner_pipeline = pipeline(
            "token-classification",
            model=ner_model,
            tokenizer=ner_tokenizer,
            aggregation_strategy="simple"
        )
    except Exception as e:
        st.error(f"Failed to load SecureBERT-NER model. CTI Analysis will be disabled. Error: {e}")
        ner_tokenizer, ner_pipeline = None, None

    # 2. Embedding Model (all-MiniLM-L6-v2 for Clustering)
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        st.error(f"Failed to load Sentence Transformer model. Clustering will be disabled. Error: {e}")
        embedding_model = None
        
    # 3. Sentiment Polarity Model
    try:
        sentiment_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
        sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
    except Exception as e:
        st.error(f"Failed to load Sentiment model. Error: {e}")
        sentiment_tokenizer, sentiment_model = None, None

    # 4. SpaCy Model (for Linguistic Analysis)
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        st.error(f"Failed to load spaCy model. Run `python -m spacy download en_core_web_sm`. Error: {e}")
        nlp = None

    return {
        "ner_tokenizer": ner_tokenizer,
        "ner_pipeline": ner_pipeline,
        "embedding_model": embedding_model,
        "sentiment_tokenizer": sentiment_tokenizer,
        "sentiment_model": sentiment_model,
        "nlp_spacy": nlp
    }

MODELS = load_cti_models()
NER_PIPELINE = MODELS["ner_pipeline"]
NER_TOKENIZER = MODELS["ner_tokenizer"]
EMBEDDING_MODEL = MODELS["embedding_model"]
SENTIMENT_TOKENIZER = MODELS["sentiment_tokenizer"]
SENTIMENT_MODEL = MODELS["sentiment_model"]
NLP_SPACY = MODELS["nlp_spacy"]


# --- UTILITY FUNCTIONS ---

def extract_pdf_text(file_buffer):
    """Extracts text from an uploaded Streamlit file buffer."""
    try:
        reader = PdfReader(file_buffer)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + " \n"
        return text
    except Exception as e:
        return f"Error reading PDF file: {type(e).__name__}: {str(e)}"

def chunk_text(text, max_length=512, overlap=50):
    """Tokenizes and chunks text for NER processing."""
    if not NER_TOKENIZER: return ["Model not loaded."]
    tokens = NER_TOKENIZER.encode(text, add_special_tokens=False)
    chunks = [NER_TOKENIZER.decode(tokens[i:i + max_length]) for i in range(0, len(tokens), max_length - overlap)]
    return chunks

def split_into_sentences(text):
    """Splits text into clean sentences."""
    sentences = nltk.sent_tokenize(text)
    sentences = [re.sub(r'\s+', ' ', s).strip() for s in sentences] # Clean whitespace
    return [s for s in sentences if s]

# --- KNOWLEDGE GRAPH FUNCTIONS (Igraph) ---

def build_cti_knowledge_graph_igraph(entities, labels):
    """Constructs an iGraph graph with CTI-specific rules for relationships."""
    name_to_original_label = {}
    vertex_names = []
    
    # 1. Deduplicate and map entities to labels
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
    
    # Define a color map for visualization
    color_map = {
        'ACT':'#1f78b4','TOOL':'#33a02c','IDTY':'#ff7f00','TIME':'#cab2d6',
        'MISC':'#a6cee3','APT':'#e31a1c','VULID':'#ffff99','IP':'#fdbf6f',
        'URL':'#ff7f00','DOMAIN':'#b2df8a','FILE':'#fb9a99','HASH':'#a6cee3',
        'CVE':'#ffff99','OS':'#cab2d6','PROTOCOL':'#fdbf6f'
    }
    G.vs["color"] = [color_map.get(lab, '#a6cee3') for lab in G.vs["node_type"]]
    
    # 2. Define edges based on sequential entity pairs
    edges_to_add = []
    edge_relations = []
    cleaned_entities = [ent.replace('\n', ' ').strip() for ent in entities]
    
    for i in range(len(cleaned_entities) - 1):
        e1, l1 = cleaned_entities[i], labels[i]
        e2, l2 = cleaned_entities[i+1], labels[i+1]
        
        # Ensure entities are valid and present in the graph
        if not e1 or not e2 or e1 not in G.vs["name"] or e2 not in G.vs["name"]: continue
        
        id1 = G.vs.find(name=e1).index
        id2 = G.vs.find(name=e2).index
        
        relation = "related_to"
        # CTI Relationship Rules (combined from original files)
        if l1 == "IDTY" and l2 == "ACT": relation = "performs_ttp"
        elif l1 == "ACT" and l2 == "TOOL": relation = "uses_tool"
        elif l1 == "APT" and l2 == "MALWARE": relation = "uses_malware"
        elif l1 == "MALWARE" and l2 in ["IP", "URL", "DOMAIN", "FILE", "HASH"]: relation = "communicates_with"
        elif l1 == "VULID" and l2 in ["OS", "TOOL"]: relation = "affects"
        
        edges_to_add.append((id1, id2))
        edge_relations.append(relation)
        
    G.add_edges(edges_to_add)
    G.es["label"] = edge_relations
    return G

def query_entity_graph_igraph(G, entity_name):
    """Generates a 1-hop subgraph plot for the selected entity."""
    if G is None or entity_name is None: return None, "Graph not generated or entity not selected."
    clean_name = entity_name.replace('\n', ' ').strip()
    if clean_name not in G.vs["name"]: return None, f"Entity '{clean_name}' not found."
        
    try:
        center_vid = G.vs.find(name=clean_name).index
        neighbor_vids = G.neighbors(center_vid, mode="all")
        subgraph = G.induced_subgraph(list(set([center_vid] + neighbor_vids)))
        if not subgraph.vs: return None, f"Entity '{clean_name}' has no connections to plot."

        layout = subgraph.layout("kamada_kawai")
        visual_style = {
            "vertex_label": subgraph.vs["name"], 
            "vertex_color": subgraph.vs["color"], 
            "edge_label": subgraph.es["label"], 
            "edge_color": "gray", 
            "vertex_size": 25, 
            "vertex_label_size": 10, 
            "edge_label_size": 9, 
            "bbox": (800, 600), 
            "margin": 50
        }
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ig.plot(subgraph, target=ax, layout=layout, **visual_style)
        ax.set_title(f"Knowledge Graph: 1-Hop Neighbors of '{clean_name}'")
        return fig, f"Successfully mapped {subgraph.vcount()} connections."
    except Exception as e:
        plt.close('all')
        return None, f"Error generating subgraph: {e}"

# --- SEMANTIC CLUSTERING / TOPIC MODELING FUNCTIONS ---

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
            # Use max_features=3 and ngram_range up to 2 for better topic names
            vectorizer = TfidfVectorizer(stop_words='english', max_features=3, ngram_range=(1, 2))
            corpus = [" ".join(docs)]
            vectorizer.fit(corpus)
            feature_names = vectorizer.get_feature_names_out()
            topic_names[cluster_id] = ", ".join(feature_names)
        except ValueError:
            topic_names[cluster_id] = "Short / Common Phrases"
            
    return topic_names

def perform_clustering(sentences):
    """Generates embeddings and clusters them using DBSCAN."""
    if not sentences or not EMBEDDING_MODEL: return None, None, None, "No sentences or embedding model loaded."
    
    embeddings = EMBEDDING_MODEL.encode(sentences)
    # DBSCAN parameters might need tuning based on data quality
    dbscan = DBSCAN(eps=1.0, min_samples=2) 
    dbscan.fit(embeddings)
    cluster_assignments = dbscan.labels_
    topic_names = get_cluster_topic_names(sentences, cluster_assignments)
    return embeddings, cluster_assignments, topic_names, f"Clustered {len(sentences)} sentences into {len(set(cluster_assignments)) - (1 if -1 in cluster_assignments else 0)} topics."

def create_cluster_plot(embeddings, cluster_assignments, topic_names):
    """Creates a 2D scatter plot of the clusters using PCA."""
    if embeddings is None: return None
    
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    unique_labels = sorted(set(cluster_assignments))
    # Use a color map for better visual distinction
    colors = [plt.cm.viridis(each) for each in np.linspace(0, 1, len(unique_labels))]
    
    for k, col in zip(unique_labels, colors):
        label = topic_names.get(k, "Unknown")
        if k == -1: col = [0.2, 0.2, 0.2, 0.5] # Darker color for outliers
        
        class_member_mask = (cluster_assignments == k)
        xy = reduced_embeddings[class_member_mask]
        
        ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=10 if k != -1 else 5, label=label)
                
    ax.set_title("Semantic Topic Clusters (PCA Reduction)")
    ax.legend(title="Topics", loc='best')
    return fig

def topic_modeling_bertopic(text_input):
    """Performs BERTopic on the input text and visualizes results."""
    texts = [t.strip() for t in text_input.split("\n") if t.strip()]
    if len(texts) < 5:
        return st.warning("Please enter at least 5 distinct text segments (one per line) for effective topic modeling.")
    
    with st.spinner("Running BERTopic model..."):
        try:
            # Use a pre-calculated model for speed/reliability
            topic_model = BERTopic(verbose=False, min_topic_size=2)
            topics, probs = topic_model.fit_transform(texts)
            # Visualize the top 5 topics as a bar chart
            fig = topic_model.visualize_barchart(top_n_topics=5)
            # Convert plotly figure to st.plotly_chart compatible format
            st.plotly_chart(fig, use_container_width=True)
            
            # Show topics table
            st.subheader("Discovered Topics")
            df_topics = topic_model.get_topic_info().drop(columns=['Representative_Docs'])
            st.dataframe(df_topics, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error during BERTopic analysis: {e}")


# --- TEXT CLASSIFICATION FUNCTIONS ---

def sentiment_analysis(text):
    """Analyzes text sentiment using a fine-tuned DistilBERT model."""
    if not text.strip() or not SENTIMENT_MODEL or not SENTIMENT_TOKENIZER:
        return pd.DataFrame([{"Label": "Model Error/No Input", "Score": 0.0, "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]), "Text Classification"

    inputs = SENTIMENT_TOKENIZER(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = SENTIMENT_MODEL(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    label = SENTIMENT_MODEL.config.id2label[torch.argmax(probs).item()]
    score = torch.max(probs).item()

    return pd.DataFrame([{
        "Label": label,
        "Score": round(score, 3),
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }])

def cti_classification(text):
    """Classifies CTI-related text based on keyword matching."""
    if not text.strip():
        return pd.DataFrame([{"Label": "No input text provided", "Score": 0.0, "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}])

    keywords = {
        "phishing": "Phishing", "malware": "Malware", "ransomware": "Malware",
        "cve": "Vulnerability", "exploit": "Exploit", "incident": "Security Alert",
        "breach": "Security Alert", "attack": "Attack", "apt": "Threat Actor Group",
        "ip address": "Indicator of Compromise (IoC)", "domain": "Indicator of Compromise (IoC)"
    }

    text_lower = text.lower()
    detected_labels = set()
    for word, label in keywords.items():
        if word in text_lower:
            detected_labels.add(label)

    if not detected_labels:
        detected_labels.add("Informational/General")

    return pd.DataFrame([{
        "Label": label,
        "Score": 1.0 if label != "Informational/General" else 0.5,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    } for label in sorted(list(detected_labels))])

# --- LINGUISTIC ANALYSIS FUNCTIONS ---

def linguistic_analysis_spacy(text):
    """Performs Part-of-Speech (POS) tagging and Dependency Parsing."""
    if not text.strip() or not NLP_SPACY:
        return [], "<p>Please enter text for analysis or spaCy model is not loaded.</p>"
    
    with st.spinner("Running spaCy analysis..."):
        doc = NLP_SPACY(text)
        
        # 1. POS Tagging
        pos_tags = [(t.text, t.pos_, t.dep_) for t in doc]
        pos_df = pd.DataFrame(pos_tags, columns=["Token", "POS Tag", "Dependency"])

        # 2. Dependency Visualization (Rendered as HTML)
        html = displacy.render(doc, style="dep", page=False)
        # Fix the CSS path for embedding in Streamlit
        html = html.replace("width: 6.5em", "width: auto") 
        
        return pos_df, html

# --- STREAMLIT APP LAYOUT ---

def main():
    st.title("🛡️ Unified CTI & NLP Analysis Dashboard")
    st.markdown("A comprehensive tool for analyzing CTI reports, clustering semantic topics, and classifying text sentiment.")

    # Initialize session state for large objects
    if 'pdf_text' not in st.session_state:
        st.session_state.pdf_text = ""
    if 'sentences' not in st.session_state:
        st.session_state.sentences = []
    if 'graph_g' not in st.session_state:
        st.session_state.graph_g = None
    if 'entities_df' not in st.session_state:
        st.session_state.entities_df = pd.DataFrame(columns=['Entity', 'Type', 'Score'])
    if 'entity_choices' not in st.session_state:
        st.session_state.entity_choices = []
        
    # --- SIDEBAR INPUTS ---
    st.sidebar.title("Data Input & Processing")
    st.sidebar.markdown("Upload a document to run CTI analysis.")
    
    # PDF Uploader (for Tab 1: CTI Report Analyzer)
    pdf_file = st.sidebar.file_uploader(
        "Upload CTI Report (PDF)", 
        type="pdf", 
        key="sidebar_pdf_uploader"
    )
    
    # CSV Uploader (placeholder for future implementation)
    csv_file = st.sidebar.file_uploader(
        "Upload Data (CSV) - *Feature TBD*", 
        type="csv",
        key="sidebar_csv_uploader"
    )

    process_button_clicked = False
    if pdf_file:
        st.sidebar.markdown("---")
        # Centralized processing button in the sidebar
        if st.sidebar.button("Process PDF for Analysis", type="primary", key="sidebar_process_btn"):
            process_button_clicked = True


    tab_pdf, tab_classify, tab_linguistic = st.tabs([
        "📄 CTI Report Analyzer (PDF)", 
        "✍️ Text Classification & Sentiment", 
        "🧠 Linguistic Analysis"
    ])

    # --- TAB 1: CTI Report Analyzer (PDF) ---
    with tab_pdf:
        st.header("CTI Report Analysis")
        
        if pdf_file is None:
            st.info("Please upload a PDF file using the **Data Input & Processing** section in the sidebar to begin analysis.")
        
        # PROCESSING LOGIC (Triggered by the sidebar button)
        if process_button_clicked and pdf_file:
            st.session_state.pdf_text = extract_pdf_text(pdf_file)
            st.session_state.sentences = split_into_sentences(st.session_state.pdf_text)

            if not NER_PIPELINE:
                st.error("NER model failed to load. Cannot process entities.")
            elif st.session_state.pdf_text.startswith("Error"):
                st.error(st.session_state.pdf_text)
            else:
                with st.spinner(f"Extracting entities from {len(st.session_state.sentences)} sentences..."):
                    chunks = chunk_text(st.session_state.pdf_text)
                    results = [res for chunk in chunks for res in NER_PIPELINE(chunk)]
                    
                    if results:
                        df = pd.DataFrame(results).rename(columns={'word': 'Entity', 'entity_group': 'Type'})
                        df['Score'] = df['score'].round(4)
                        st.session_state.entities_df = df[['Entity', 'Type', 'Score']].copy()
                        
                        G = build_cti_knowledge_graph_igraph(df["Entity"].tolist(), df["Type"].tolist())
                        st.session_state.graph_g = G
                        st.session_state.entity_choices = G.vs["name"]
                        st.success(f"Processing complete. Found {G.vcount()} unique entities.")
                    else:
                        st.warning("No CTI entities found in the report.")
                        st.session_state.entities_df = pd.DataFrame(columns=['Entity', 'Type', 'Score'])
                        st.session_state.graph_g = None
                        st.session_state.entity_choices = []

        # Display extracted entities and downstream analysis sections
        if not st.session_state.entities_df.empty:
            st.subheader("1. Extracted Entities (NER Results)")
            st.dataframe(st.session_state.entities_df, use_container_width=True)

            # Knowledge Graph Visualization
            if st.session_state.graph_g:
                st.subheader("2. Knowledge Graph Visualization")
                
                col1, col2 = st.columns([2, 1])
                with col2:
                    selected_entity = st.selectbox(
                        "Select an Entity to view its immediate connections (1-hop graph):", 
                        options=st.session_state.entity_choices,
                        key="kg_entity_select"
                    )
                    st.markdown("Select an entity from the list to see its immediate connections within the report.")

                with col1:
                    if selected_entity:
                        fig, status = query_entity_graph_igraph(st.session_state.graph_g, selected_entity)
                        if fig:
                            st.pyplot(fig)
                        else:
                            st.info(status)

            # Semantic Clustering
            if st.session_state.sentences:
                st.subheader("3. Semantic Clustering (DBSCAN + PCA)")
                st.info(f"Analyzing {len(st.session_state.sentences)} sentences for thematic clusters.")

                if st.button("Run Semantic Clustering", key="run_cluster"):
                    if not EMBEDDING_MODEL:
                        st.error("Embedding model failed to load. Clustering cannot run.")
                    else:
                        with st.spinner("Generating embeddings and clustering sentences..."):
                            embeddings, labels, topics, status = perform_clustering(st.session_state.sentences)
                            fig = create_cluster_plot(embeddings, labels, topics)
                            
                            st.success(status)
                            st.pyplot(fig)
                            st.markdown("The 2D plot shows sentences grouped by semantic similarity.")

                            st.subheader("Cluster Topics:")
                            topic_data = [{"Cluster ID": k, "Topic Keywords": v, "Count": np.sum(labels == k)} 
                                          for k, v in topics.items()]
                            st.dataframe(pd.DataFrame(topic_data).sort_values(by="Count", ascending=False), use_container_width=True)

    # --- TAB 2: Text Classification & Sentiment ---
    with tab_classify:
        st.header("Text Classification")
        input_text = st.text_area(
            "Enter text for Classification and Sentiment Analysis:",
            height=200,
            placeholder="e.g., A phishing campaign distributing the Emotet loader targeting government agencies was successfully mitigated, leading to a positive outcome despite the initial security incident.",
            key="classify_input"
        )
        
        if st.button("Analyze Text", key="run_classify", type="primary"):
            if not input_text.strip():
                st.warning("Please enter some text to analyze.")
            else:
                col1, col2 = st.columns(2)
                
                # CTI Classification
                with col1:
                    st.subheader("CTI Keyword Classification")
                    cti_df = cti_classification(input_text)
                    st.dataframe(cti_df, use_container_width=True)
                    st.markdown("Labels are based on simple keyword matching.")

                # Sentiment Analysis
                with col2:
                    st.subheader("Sentiment Polarity Analysis")
                    sentiment_df = sentiment_analysis(input_text)
                    st.dataframe(sentiment_df, use_container_width=True)
                    
                    if not sentiment_df.empty:
                        label = sentiment_df.iloc[0]['Label']
                        st.metric(label="Overall Sentiment", value=label)

    # --- TAB 3: Linguistic Analysis ---
    with tab_linguistic:
        st.header("Linguistic and Topic Analysis")

        # Linguistic Analysis (POS & Dependency)
        st.subheader("1. Part-of-Speech and Dependency Parsing (spaCy)")
        ling_input = st.text_area(
            "Enter text for detailed linguistic analysis:",
            height=150,
            placeholder="e.g., The threat actor used the TrickBot malware in a targeted attack.",
            key="ling_input"
        )

        if st.button("Analyze Syntax", key="run_ling_analysis"):
            if ling_input.strip():
                pos_df, dep_html = linguistic_analysis_spacy(ling_input)
                
                st.markdown("The dependency parser shows the grammatical structure of the sentence.")
                
                st.subheader("POS Tagging & Dependencies")
                st.dataframe(pos_df, use_container_width=True)
                
                st.subheader("Dependency Visualization")
                # Render the spaCy displacy HTML visualization
                components.html(dep_html, height=400, scrolling=True)
            else:
                st.warning("Please enter text for linguistic analysis.")
                
        # Topic Modeling (BERTopic)
        st.subheader("2. Topic Discovery (BERTopic)")
        topic_input = st.text_area(
            "Enter multiple reports or documents (one per line) for Topic Discovery:",
            lines=5,
            placeholder="Line 1: The APT group Cozy Bear uses phishing techniques.\nLine 2: A new vulnerability was discovered in the OS kernel.\nLine 3: TrickBot is a notorious piece of banking malware.",
            key="topic_input"
        )
        
        if st.button("Run BERTopic Modeling", key="run_bertopic", type="secondary"):
            if topic_input.strip():
                topic_modeling_bertopic(topic_input)
            else:
                st.warning("Please enter text segments for topic modeling.")


if __name__ == "__main__":
    main()
