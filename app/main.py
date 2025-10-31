# main.py
import streamlit as st
import pandas as pd
from nlp_logic import extract_text
from transformers import pipeline
import networkx as nx
from pyvis.network import Network
import nltk

# Ensure tokenizer available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# -------------------------
# Streamlit Page Config
# -------------------------
st.set_page_config(page_title="Cyber Threat Intelligence Knowledge Graph", layout="wide")

st.title("Cyber Threat Intelligence (CTI) Knowledge Graph")
st.write("Upload a CTI report in CSV or PDF format to extract cyber entities and visualize relationships.")

# -------------------------
# Load NLP Models
# -------------------------
@st.cache_resource
def load_models():
    ner_pipeline = pipeline("ner", grouped_entities=True)
    sentiment_pipeline = pipeline("sentiment-analysis")
    return ner_pipeline, sentiment_pipeline

ner_pipeline, sentiment_pipeline = load_models()

# -------------------------
# File Upload
# -------------------------
uploaded_file = st.file_uploader("Upload CTI Report", type=["csv", "pdf", "txt"])

if uploaded_file:
    with st.spinner("Extracting text from the file..."):
        text = extract_text(uploaded_file)

    if not text:
        st.error("No readable text found in the uploaded file.")
    else:
        st.success("File processed successfully.")

        # Split into sentences
        sentences = nltk.sent_tokenize(text)
        sample_sentences = sentences[:50]

        # Run NLP
        with st.spinner("Performing Named Entity Recognition..."):
            entities = []
            for sent in sample_sentences:
                entities.extend(ner_pipeline(sent))

        if not entities:
            st.warning("No entities were detected.")
        else:
            df = pd.DataFrame(entities)
            df = df.rename(columns={"word": "Entity", "entity_group": "Type", "score": "Score"})
            df["Score"] = df["Score"].round(3)

            st.subheader("Extracted Entities")
            st.dataframe(df[["Entity", "Type", "Score"]], use_container_width=True)

            # -------------------------
            # Build Knowledge Graph
            # -------------------------
            G = nx.Graph()

            for _, row in df.iterrows():
                entity = row["Entity"]
                entity_type = row["Type"]
                G.add_node(entity, group=entity_type)
                G.add_edge(entity_type, entity)

            net = Network(height="600px", width="100%", bgcolor="#0e1117", font_color="white")
            net.from_nx(G)
            net.toggle_physics(True)
            html_file = "cti_graph.html"
            net.save_graph(html_file)

            st.subheader("Knowledge Graph Visualization")
            with open(html_file, "r", encoding="utf-8") as f:
                graph_html = f.read()
            st.components.v1.html(graph_html, height=600, scrolling=True)
