import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import igraph as ig

# Import core NLP logic
from nlp_logic import process_cti_pdf, perform_clustering

# ---------------- STREAMLIT SETUP ----------------
st.set_page_config(page_title="Project ATHENA - CTI Workbench", layout="wide")
st.title("Project ATHENA - Cyber Threat Intelligence Workbench")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload CTI Report (PDF)", type=["pdf"])

if uploaded_file:
    st.info(f"Processing file: {uploaded_file.name}")
    with st.spinner("Analyzing..."):
        df_entities, graph, sentences = process_cti_pdf(uploaded_file)

    st.success("Analysis complete")

    # Create interface tabs
    tabs = st.tabs(["Entity Extraction", "Knowledge Graph", "Sentence Clustering"])

    # --------- ENTITY EXTRACTION TAB ---------
    with tabs[0]:
        st.subheader("Extracted Entities")
        if not df_entities.empty:
            st.dataframe(df_entities)
        else:
            st.warning("No entities found in the document.")

    # --------- KNOWLEDGE GRAPH TAB ---------
    with tabs[1]:
        st.subheader("Knowledge Graph Visualization")
        if graph and len(graph.vs) > 0:
            entities = graph.vs["name"]
            selected_entity = st.selectbox("Select Entity to visualize", entities)

            if st.button("Show Subgraph"):
                idx = graph.vs.find(name=selected_entity).index
                neighbors = graph.neighbors(idx, mode="all")
                subgraph = graph.induced_subgraph(list(set([idx] + neighbors)))

                layout = subgraph.layout("kamada_kawai")
                fig, ax = plt.subplots(figsize=(10, 8))
                ig.plot(
                    subgraph,
                    target=ax,
                    layout=layout,
                    vertex_label=subgraph.vs["name"],
                    vertex_color=subgraph.vs["color"],
                    edge_label=subgraph.es["label"]
                )
                st.pyplot(fig)
        else:
            st.warning("No knowledge graph data available.")

    # --------- CLUSTERING TAB ---------
    with tabs[2]:
        st.subheader("Sentence Clustering")
        if st.button("Run Sentence Clustering"):
            embeddings, labels, topics = perform_clustering(sentences)
            if embeddings is not None:
                from sklearn.decomposition import PCA
                import numpy as np

                pca = PCA(n_components=2)
                reduced = pca.fit_transform(embeddings)

                fig, ax = plt.subplots(figsize=(10, 8))
                for cid in set(labels):
                    mask = labels == cid
                    ax.scatter(
                        reduced[mask, 0],
                        reduced[mask, 1],
                        label=topics.get(cid, str(cid)),
                        alpha=0.7
                    )
                ax.legend()
                st.pyplot(fig)
            else:
                st.warning("No sentences available for clustering.")
else:
    st.markdown("Upload a CTI PDF file to begin the analysis.")
