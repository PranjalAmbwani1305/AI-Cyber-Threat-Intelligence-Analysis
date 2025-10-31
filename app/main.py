import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import igraph as ig
from nlp_logic import extract_pdf_text, split_into_sentences, perform_clustering, build_cti_graph, process_cti_data

# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config(page_title="🧠 AI Cyber Threat Intelligence Dashboard", layout="wide")

st.title("🧠 AI Cyber Threat Intelligence Dashboard")
st.markdown("""
Transform **Cyber Threat Intelligence (CTI)** data — structured or unstructured — into actionable insights using **NLP**, **entity extraction**, **clustering**, and **knowledge graph analysis**.
""")

# ---------------- SIDEBAR ----------------
st.sidebar.header("📂 Upload CTI Report")
uploaded_file = st.sidebar.file_uploader(
    "Choose a structured or unstructured CTI dataset:",
    type=["csv", "pdf", "txt"]
)
quick_mode = st.sidebar.checkbox("⚡ Quick Mode (Skip Graph Layout)", value=True)
st.sidebar.markdown("Limit 200MB • CSV, PDF, or TXT")

# ---------------- MAIN APP ----------------
if uploaded_file:
    st.info(f"📁 Processing `{uploaded_file.name}`...")

    try:
        with st.spinner("⏳ Extracting entities and building knowledge graph..."):
            text, entities_df, sentences, cluster_labels, topic_map, graph = process_cti_data(
                uploaded_file, quick_mode=quick_mode
            )

        if text.strip() == "":
            st.error("❌ No valid text extracted from file.")
            st.stop()

        # --- SUMMARY METRICS ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Entities Extracted", len(entities_df))
        col2.metric("Sentences Processed", len(sentences))
        col3.metric("Relationships Identified", len(graph.es))

        st.divider()

        # --- ENTITIES ---
        st.subheader("🔍 Entities Extracted")
        if not entities_df.empty:
            st.dataframe(entities_df, use_container_width=True)
            st.bar_chart(entities_df["Type"].value_counts())
        else:
            st.warning("No named entities found.")

        # --- CLUSTERING ---
        st.subheader("🧩 Sentence Clustering")
        if sentences:
            cluster_data = pd.DataFrame({
                "Sentence": sentences,
                "Cluster": cluster_labels
            })
            st.dataframe(cluster_data.head(50), use_container_width=True)
        else:
            st.info("No sentences available for clustering.")

        # --- KNOWLEDGE GRAPH ---
        st.subheader("🌐 Cyber Threat Knowledge Graph")
        if len(graph.vs) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            layout = None if quick_mode else graph.layout("kamada_kawai")
            ig.plot(
                graph,
                target=ax,
                layout=layout,
                vertex_label=graph.vs["name"],
                vertex_color=graph.vs["color"],
                vertex_size=25,
                edge_arrow_size=0.5,
                edge_color="gray"
            )
            st.pyplot(fig)
        else:
            st.info("Graph could not be generated — insufficient data.")

        # --- RAW TEXT PREVIEW ---
        with st.expander("📜 Raw Text Preview"):
            st.text_area("Extracted Text", text[:8000], height=300)

    except Exception as e:
        st.error("❌ An error occurred during report processing.")
        st.exception(e)
else:
    st.info("📥 Please upload a CTI dataset (CSV, PDF, or TXT) to begin analysis.")

st.divider()
st.caption("AI Cyber Threat Intelligence Dashboard — SecureBERT & SentenceTransformer © 2025")
