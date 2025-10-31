# app_streamlit.py
import streamlit as st
import pandas as pd
import tempfile
from nlp_logic import (
    load_model,
    extract_pdf_text,
    chunk_text,
    build_cti_graph,
    draw_subgraph,
)

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="CTI Knowledge Graph", layout="wide")

# ---------------------------------------------------------
# APP HEADER
# ---------------------------------------------------------
st.title("Cyber Threat Intelligence Knowledge Graph Analyzer")
st.markdown("""
This application extracts cyber threat entities from PDF reports and visualizes their relationships 
using a knowledge graph built with SecureBERT-NER.
""")

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
with st.sidebar:
    st.header("Model Status")
    tokenizer, ner_pipeline = load_model()
    st.success("SecureBERT-NER model loaded successfully.")

# ---------------------------------------------------------
# FILE UPLOAD SECTION
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Upload a CTI PDF report", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.subheader("Text Extraction")
    st.info("Extracting text from uploaded PDF...")
    text = extract_pdf_text(tmp_path)
    st.success("Text successfully extracted.")

    # ---------------------------------------------------------
    # NLP PROCESSING
    # ---------------------------------------------------------
    if st.button("Run Entity Extraction"):
        st.info("Running SecureBERT-NER on extracted text...")
        chunks = chunk_text(text, tokenizer)
        results = [r for c in chunks for r in ner_pipeline(c)]

        if not results:
            st.warning("No entities detected in the uploaded report.")
        else:
            df = pd.DataFrame(results)
            df = df.rename(columns={"word": "Entity", "entity_group": "Type"})
            df["Score"] = df["score"].round(4)
            df_display = df[["Entity", "Type", "Score"]]
            st.session_state["entities_df"] = df_display

            st.subheader("Extracted Entities")
            st.dataframe(df_display, use_container_width=True)

            G = build_cti_graph(df["Entity"].tolist(), df["Type"].tolist())
            st.session_state["G"] = G
            st.session_state["entity_names"] = G.vs["name"]

            st.success(f"Processed successfully: {G.vcount()} entities and {G.ecount()} relationships detected.")

# ---------------------------------------------------------
# GRAPH VISUALIZATION
# ---------------------------------------------------------
if "G" in st.session_state:
    st.subheader("Knowledge Graph Visualization")
    selected_entity = st.selectbox("Select an entity to explore", st.session_state["entity_names"])
    if st.button("Generate Subgraph"):
        fig = draw_subgraph(st.session_state["G"], selected_entity)
        st.pyplot(fig)
