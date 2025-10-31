# app/main.py
import streamlit as st
import pandas as pd
from nlp_logic import process_cti_file, visualize_graph_matplotlib
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI CTI Dashboard", layout="wide")
st.title("🧠 AI Cyber Threat Intelligence Dashboard")
st.markdown("Transform CTI data (CSV / PDF / TXT / XLSX) into entity insights, clusters and a knowledge graph.")

# --- Sidebar upload ---
uploaded = st.sidebar.file_uploader("Upload CTI report (CSV, PDF, TXT, XLSX)", type=["csv", "pdf", "txt", "xlsx"])
st.sidebar.markdown("Supports CSV, Excel, PDF, or text files. Max recommended size: 200MB.")

if uploaded:
    with st.spinner("Processing file — this may take a few seconds..."):
        try:
            res = process_cti_file(uploaded, clustering_k=6)
        except Exception as e:
            st.error(f"Processing error: {e}")
            raise

    # Metrics
    entities_df = res.get("entities", pd.DataFrame())
    sentences = res.get("sentences", [])
    graph = res.get("graph", None)

    c1, c2, c3 = st.columns(3)
    c1.metric("Entities Extracted", len(entities_df))
    c2.metric("Sentences Processed", len(sentences))
    c3.metric("Graph Nodes", len(graph.vs) if graph else 0)

    st.markdown("---")

    # Entities table
    st.subheader("Extracted Entities")
    if not entities_df.empty:
        st.dataframe(entities_df, use_container_width=True)
        st.markdown("**Entity type counts**")
        st.bar_chart(entities_df["Type"].value_counts())
    else:
        st.info("No entities identified.")

    st.markdown("---")

    # Clustering preview
    st.subheader("Semantic Clusters (sample)")
    topic_map = res.get("topic_map", {})
    if topic_map:
        for cid, items in topic_map.items():
            st.markdown(f"**Cluster {cid+1}**")
            for s in items[:5]:
                st.write("- " + s)
    else:
        st.info("No clusters available (insufficient text).")

    st.markdown("---")

    # Graph visualization
    st.subheader("Knowledge Graph")
    fig = visualize_graph_matplotlib(graph)
    if fig:
        st.pyplot(fig)
    else:
        st.info("Graph could not be generated (no entities/relationships).")

    st.markdown("---")
    with st.expander("Raw extracted text (preview)"):
        txt = res.get("text", "")
        st.text_area("Text preview", txt[:5000], height=300)

else:
    st.info("Upload a CTI file in the left sidebar to start analysis.")
