# app.py
import streamlit as st
import pandas as pd
import nltk
import pdfplumber
import io
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from pyvis.network import Network
import tempfile
import os
import networkx as nx

# Download NLTK punkt tokenizer if not found
nltk.download("punkt", quiet=True)

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Cyber Threat Intelligence Dashboard",
    layout="wide",
    page_icon="🧠",
)

# ---------------------------------------------------------
# MODEL INITIALIZATION (done once)
# ---------------------------------------------------------
@st.cache_resource
def load_ner_model():
    model_name = "CyberPeace-Institute/SecureBERT-NER"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    ner = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    return ner

ner_pipeline = load_ner_model()

# ---------------------------------------------------------
# NLP LOGIC
# ---------------------------------------------------------
def extract_text(file):
    """Extract text from CSV, PDF, or TXT."""
    try:
        if file.name.endswith(".pdf"):
            text = ""
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
            return text
        elif file.name.endswith(".csv"):
            df = pd.read_csv(file)
            return " ".join(df.astype(str).fillna("").values.flatten().tolist())
        elif file.name.endswith(".txt"):
            return file.read().decode("utf-8")
        else:
            return None
    except Exception as e:
        return str(e)


def extract_entities(text):
    """Run NER model on text."""
    sentences = nltk.sent_tokenize(text)
    entities = []
    for sent in sentences:
        res = ner_pipeline(sent)
        for r in res:
            entities.append({
                "Entity": r["word"],
                "Type": r["entity_group"],
                "Score": round(r["score"], 4),
                "Sentence": sent,
            })
    return pd.DataFrame(entities)


def build_graph(df):
    """Build interactive graph using Pyvis."""
    if df.empty:
        return None

    G = nx.DiGraph()

    color_map = {
        'ACT': '#1f78b4', 'TOOL': '#33a02c', 'IDTY': '#ff7f00',
        'TIME': '#cab2d6', 'APT': '#e31a1c', 'VULID': '#ffff99',
        'IP': '#fdbf6f', 'URL': '#ff7f00', 'DOMAIN': '#b2df8a',
        'FILE': '#fb9a99', 'HASH': '#a6cee3', 'CVE': '#ffff99',
        'OS': '#cab2d6', 'PROTOCOL': '#fdbf6f'
    }

    # Add nodes
    for _, row in df.iterrows():
        ent, typ = row["Entity"], row["Type"]
        G.add_node(ent, color=color_map.get(typ, "#9fa8da"), title=f"{ent} ({typ})")

    # Create relationships (naive rule: entities in same sentence → connected)
    grouped = df.groupby("Sentence")
    for _, group in grouped:
        ents = group["Entity"].tolist()
        for i in range(len(ents) - 1):
            G.add_edge(ents[i], ents[i + 1], title="related_to")

    # Create Pyvis network
    net = Network(height="600px", bgcolor="#0e1117", font_color="white", directed=True)
    net.from_nx(G)
    net.repulsion(node_distance=150, spring_length=200)

    tmp_path = os.path.join(tempfile.gettempdir(), "cti_graph.html")
    net.show(tmp_path)
    return tmp_path

# ---------------------------------------------------------
# STREAMLIT LAYOUT
# ---------------------------------------------------------
st.title("🧠 AI Cyber Threat Intelligence Dashboard")
st.markdown("""
Transform **Cyber Threat Intelligence (CTI)** data — structured or unstructured — into actionable insights  
with NLP-powered entity extraction and interactive knowledge graph visualization.
""")

# Sidebar upload
st.sidebar.header("📂 Upload CTI Report")
uploaded_file = st.sidebar.file_uploader(
    "Choose a CTI report",
    type=["pdf", "csv", "txt"],
    help="Supported formats: PDF, CSV, TXT",
)

# Tabs for app workflow
tab1, tab2, tab3 = st.tabs(["1️⃣ Upload & Extract", "2️⃣ Entities", "3️⃣ Knowledge Graph"])

# ---------------------------------------------------------
# TAB 1: UPLOAD & PROCESS
# ---------------------------------------------------------
with tab1:
    if uploaded_file:
        with st.spinner("Processing your file..."):
            text = extract_text(uploaded_file)

        if text:
            st.success("✅ File processed successfully.")
            st.text_area("Extracted Text (first 1500 chars)", text[:1500] + "..." if len(text) > 1500 else text, height=200)

            if st.button("Run Entity Extraction", type="primary"):
                with st.spinner("Running NLP model... This may take a moment."):
                    df_entities = extract_entities(text)
                    st.session_state["entities_df"] = df_entities
                    st.success(f"✅ Extracted {len(df_entities)} entities.")
        else:
            st.error("⚠️ Could not extract text from file.")
    else:
        st.info("Upload a CTI file from the sidebar to begin.")

# ---------------------------------------------------------
# TAB 2: ENTITY DISPLAY
# ---------------------------------------------------------
with tab2:
    if "entities_df" in st.session_state:
        df = st.session_state["entities_df"]
        st.subheader("📋 Extracted Entities")
        st.dataframe(df, use_container_width=True)

        st.subheader("📈 Entity Type Distribution")
        st.bar_chart(df["Type"].value_counts())
    else:
        st.warning("Please process a file first in the Upload tab.")

# ---------------------------------------------------------
# TAB 3: KNOWLEDGE GRAPH
# ---------------------------------------------------------
with tab3:
    if "entities_df" in st.session_state:
        df = st.session_state["entities_df"]
        st.subheader("🕸️ Interactive Knowledge Graph")

        if st.button("Generate Knowledge Graph"):
            with st.spinner("Building graph..."):
                html_path = build_graph(df)
                if html_path and os.path.exists(html_path):
                    with open(html_path, "r", encoding="utf-8") as f:
                        graph_html = f.read()
                    st.components.v1.html(graph_html, height=650, scrolling=True)
                else:
                    st.error("⚠️ Could not generate graph.")
    else:
        st.warning("Please extract entities before building the graph.")

st.markdown("---")
st.caption("© 2025 AI CTI Dashboard | Powered by SecureBERT-NER & Streamlit")

