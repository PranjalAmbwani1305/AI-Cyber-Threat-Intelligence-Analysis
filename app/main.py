import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from io import BytesIO
import traceback
import os, sys

# === Ensure local imports work (for Streamlit Cloud or Colab) ===
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.append(APP_DIR)

# === Correct imports from nlp_logic ===
from nlp_logic import (
    extract_pdf_text,
    process_cti_pdf,
    split_into_sentences,
    perform_clustering,
    build_cti_graph
)

# === Streamlit Page Configuration ===
st.set_page_config(
    page_title="AI Cyber Threat Intelligence Dashboard",
    layout="wide"
)

# === Sidebar ===
with st.sidebar:
    st.title("Upload CTI Report")
    uploaded_file = st.file_uploader(
        "Choose a report file",
        type=["pdf", "csv", "xlsx", "txt"],
        help="Upload unstructured (PDF/TXT) or structured (CSV/XLSX) cyber threat intelligence data."
    )
    st.markdown("---")
    st.caption("AI Cyber Threat Intelligence Suite © 2025")

# === Header ===
st.title("AI Cyber Threat Intelligence Dashboard")
st.markdown("""
Transform unstructured or structured Cyber Threat Intelligence (CTI) data into actionable insights 
using advanced NLP, clustering, and knowledge graph analysis.
""")

# === File Processing ===
if uploaded_file:
    st.info(f"Processing `{uploaded_file.name}`...")

    try:
        file_ext = uploaded_file.name.split(".")[-1].lower()

        # ---- Extract text ----
        if file_ext == "pdf":
            text = extract_pdf_text(uploaded_file)
        elif file_ext in ["csv", "xlsx"]:
            df = pd.read_csv(uploaded_file) if file_ext == "csv" else pd.read_excel(uploaded_file)
            text = " ".join(df.astype(str).values.flatten())
        elif file_ext == "txt":
            text = uploaded_file.read().decode("utf-8")
        else:
            st.error("Unsupported file format.")
            st.stop()

        # ---- Sentence splitting ----
        sentences = split_into_sentences(text)

        # ---- Run NLP pipeline ----
        with st.spinner("Analyzing with SecureBERT and building knowledge graph..."):
            result = process_cti_pdf(BytesIO(text.encode("utf-8")))
            entities_df = result["entities"]
            G = result["graph"]
            cluster_labels = result["cluster_labels"]
            topic_map = result["topic_map"]

        # ---- Summary Overview ----
        st.subheader("Analysis Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Entities Extracted", len(entities_df))
        col2.metric("Sentences Processed", len(sentences))
        col3.metric("Relationships Identified", G.ecount() if G else 0)

        # ---- Tabs for analysis sections ----
        tab1, tab2, tab3, tab4 = st.tabs([
            "Entities",
            "Sentence Clustering",
            "Knowledge Graph",
            "Raw Text"
        ])

        # --- TAB 1: Entities ---
        with tab1:
            st.subheader("Named Entity Recognition (NER)")
            if not entities_df.empty:
                st.dataframe(entities_df, use_container_width=True)
                st.bar_chart(entities_df["Type"].value_counts())
                st.download_button(
                    "Download Entities CSV",
                    entities_df.to_csv(index=False),
                    "entities.csv"
                )
            else:
                st.warning("No entities were detected.")

        # --- TAB 2: Clustering ---
        with tab2:
            st.subheader("Semantic Sentence Clustering")
            if cluster_labels is not None:
                clusters_df = pd.DataFrame({
                    "Sentence": sentences,
                    "Cluster": cluster_labels
                })
                st.dataframe(clusters_df.head(20), use_container_width=True)
                st.bar_chart(pd.Series(cluster_labels).value_counts())
            else:
                st.info("No clustering data available.")

        # --- TAB 3: Knowledge Graph ---
        with tab3:
            st.subheader("Cyber Threat Knowledge Graph")
            if G and G.vcount() > 0:
                nx_graph = nx.Graph()
                for v in G.vs:
                    nx_graph.add_node(v["name"], type=v["type"])
                for e in G.es:
                    src, tgt = e.tuple
                    nx_graph.add_edge(G.vs[src]["name"], G.vs[tgt]["name"], relation=e["label"])

                fig, ax = plt.subplots(figsize=(10, 7))
                nx.draw(
                    nx_graph,
                    with_labels=True,
                    node_size=900,
                    font_size=8,
                    font_weight="bold",
                    node_color="#42a5f5",
                    edge_color="#424242"
                )
                st.pyplot(fig)
            else:
                st.warning("Graph could not be generated.")

        # --- TAB 4: Raw Text ---
        with tab4:
            st.subheader("Extracted CTI Report Text")
            st.text_area("Report Content", text[:5000], height=300)
            st.download_button("Download Extracted Text", text, file_name="cti_report_text.txt")

    except Exception as e:
        st.error("An error occurred while processing the report:")
        st.code(traceback.format_exc())

else:
    st.info("Upload a CTI report from the sidebar to start analysis.")

# === Footer ===
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray;'>AI Cyber Threat Intelligence Dashboard — SecureBERT & SentenceTransformer © 2025</div>",
    unsafe_allow_html=True,
)
