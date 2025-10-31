import streamlit as st
import matplotlib.pyplot as plt
import igraph as ig
import pandas as pd
import os

# Dynamic import that works both locally and on Streamlit Cloud
try:
    from app.nlp_logic import extract_pdf_text, split_into_sentences, perform_clustering, build_cti_graph, process_cti_data
except ModuleNotFoundError:
    from nlp_logic import extract_pdf_text, split_into_sentences, perform_clustering, build_cti_graph, process_cti_data

# ------------------ STREAMLIT LAYOUT ------------------
st.set_page_config(page_title="AI Cyber Threat Intelligence Dashboard", layout="wide")
st.title("🧠 AI Cyber Threat Intelligence Dashboard")
st.markdown("""
Transform **Cyber Threat Intelligence (CTI)** data — structured or unstructured — into actionable insights  
using entity extraction, clustering, and knowledge graph analytics.
""")

# Sidebar
st.sidebar.header("📂 Upload CTI Report")
uploaded_file = st.sidebar.file_uploader("Choose a CTI report", type=["csv", "pdf", "txt"])
st.sidebar.markdown("Limit 200MB per file")

quick_mode = st.sidebar.checkbox("⚡ Quick Mode (skip heavy layout)", value=True)

if uploaded_file:
    st.info(f"📄 Processing `{uploaded_file.name}` ...")

    try:
        result = process_cti_data(uploaded_file)

        entities_df = result["entities"]
        sentences = result["sentences"]
        graph = result["graph"]
        cluster_labels = result.get("cluster_labels", [])
        topic_map = result.get("topic_map", {})

        st.success("✅ Analysis complete!")

        # Summary
        c1, c2, c3 = st.columns(3)
        c1.metric("Entities Extracted", len(entities_df))
        c2.metric("Sentences Processed", len(sentences))
        c3.metric("Relationships Identified", len(graph.es))

        st.divider()
        st.subheader("📍 Extracted Entities")
        if not entities_df.empty:
            st.dataframe(entities_df.head(100), use_container_width=True)
            st.bar_chart(entities_df["Type"].value_counts())
        else:
            st.warning("No entities found.")

        st.subheader("🧩 Sentence Clustering")
        if sentences:
            _, cluster_labels, topic_map = perform_clustering(sentences)
            cluster_df = pd.DataFrame({"Sentence": sentences, "Cluster": cluster_labels})
            st.dataframe(cluster_df.head(50), use_container_width=True)
        else:
            st.info("No sentences available for clustering.")

        st.subheader("🌐 Cyber Threat Knowledge Graph")
        if len(graph.vs) > 0:
            fig, ax = plt.subplots(figsize=(10, 7))
            layout = None if quick_mode else graph.layout("kamada_kawai")
            ig.plot(
                graph,
                target=ax,
                layout=layout,
                vertex_size=20,
                vertex_color=graph.vs["color"],
                vertex_label=graph.vs["label"],
                edge_color="gray",
                vertex_label_size=10
            )
            st.pyplot(fig)
        else:
            st.warning("Graph could not be generated — insufficient relationships.")

        with st.expander("📜 Raw Text Preview"):
            preview = extract_pdf_text(uploaded_file)
            st.text_area("Extracted Text", preview[:3000], height=300)

    except Exception as e:
        st.error("❌ An error occurred during report processing.")
        st.exception(e)
else:
    st.info("Please upload a CTI file (CSV or PDF) to begin analysis.")

st.divider()
st.caption("AI Cyber Threat Intelligence Dashboard — SecureBERT & SentenceTransformer © 2025")
