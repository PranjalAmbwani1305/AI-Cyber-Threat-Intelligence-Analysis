import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from io import BytesIO
import traceback

# ------------------- IMPORT CORE NLP LOGIC -------------------
from app.nlp_logic import (
    extract_pdf_text,
    split_into_sentences,
    process_cti_pdf
)

# ------------------- STREAMLIT CONFIG -------------------
st.set_page_config(page_title="AI CTI Intelligence Dashboard", layout="wide")

st.title("AI-Powered Cyber Threat Intelligence Dashboard")
st.markdown(
    "Upload a CTI report (.pdf, .csv, .xlsx, or .txt) to analyze entities, clusters, and relationships "
    "using SecureBERT-NER and SentenceTransformer-based CTI graph intelligence."
)

# ------------------- FILE UPLOAD -------------------
uploaded_file = st.file_uploader("Upload CTI Report", type=["pdf", "csv", "xlsx", "txt"])

if uploaded_file:
    st.info(f"Processing file: {uploaded_file.name}")

    # ------------------- TEXT EXTRACTION -------------------
    text = ""
    try:
        if uploaded_file.name.lower().endswith(".pdf"):
            text = extract_pdf_text(uploaded_file)
        elif uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            text = " ".join(df.astype(str).values.flatten())
        elif uploaded_file.name.lower().endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
            text = " ".join(df.astype(str).values.flatten())
        elif uploaded_file.name.lower().endswith(".txt"):
            text = uploaded_file.read().decode("utf-8")
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        st.text(traceback.format_exc())
        st.stop()

    if not text.strip():
        st.error("No text could be extracted from this file.")
        st.stop()

    st.success("Text extraction completed successfully.")
    st.download_button("Download Extracted Text", text, file_name="extracted_text.txt")

    st.divider()

    # ------------------- MAIN NLP PIPELINE -------------------
    try:
        with st.spinner("Running NLP pipeline..."):
            result = process_cti_pdf(uploaded_file)

        df_entities = result["entities"]
        G_igraph = result["graph"]
        sentences = result["sentences"]
        cluster_labels = result["cluster_labels"]
        topic_map = result["topic_map"]

    except Exception as e:
        st.error(f"Processing failed: {e}")
        st.text(traceback.format_traceback())
        st.stop()

    # ------------------- DISPLAY RESULTS -------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "Entities (NER)",
        "Semantic Clustering",
        "Sentences",
        "Knowledge Graph"
    ])

    # -------- TAB 1: NER Entities --------
    with tab1:
        st.subheader("Named Entity Recognition (SecureBERT-NER)")
        if not df_entities.empty:
            st.dataframe(df_entities, use_container_width=True)
            st.bar_chart(df_entities["Type"].value_counts())
            st.download_button(
                "Download Entities CSV",
                df_entities.to_csv(index=False),
                "entities.csv"
            )
        else:
            st.warning("No entities detected in the text.")

    # -------- TAB 2: Semantic Clustering --------
    with tab2:
        st.subheader("Sentence Clustering (Semantic Embeddings + DBSCAN)")
        if cluster_labels is not None:
            cluster_df = pd.DataFrame({
                "Sentence": sentences,
                "Cluster_ID": cluster_labels
            })
            st.dataframe(cluster_df, use_container_width=True)

            st.write("Topic Mapping Summary:")
            for cid, topic in topic_map.items():
                st.write(f"- Cluster {cid}: {topic}")

            st.download_button(
                "Download Clustering CSV",
                cluster_df.to_csv(index=False),
                "sentence_clusters.csv"
            )
        else:
            st.warning("No clusters generated for this report.")

    # -------- TAB 3: Sentences --------
    with tab3:
        st.subheader("Extracted Sentences")
        st.write(f"Total sentences detected: {len(sentences)}")
        for s in sentences[:100]:
            st.write("• " + s)

    # -------- TAB 4: Knowledge Graph --------
    with tab4:
        st.subheader("CTI Knowledge Graph Visualization")
        try:
            if G_igraph.vcount() > 0:
                G_nx = nx.Graph()
                for v in G_igraph.vs:
                    G_nx.add_node(v["name"], label=v["type"])
                for e in G_igraph.es:
                    src = G_igraph.vs[e.source]["name"]
                    tgt = G_igraph.vs[e.target]["name"]
                    G_nx.add_edge(src, tgt, label=e["label"])

                fig, ax = plt.subplots(figsize=(12, 8))
                pos = nx.spring_layout(G_nx, k=0.5)
                nx.draw_networkx_nodes(G_nx, pos, node_size=800, node_color="skyblue", alpha=0.8)
                nx.draw_networkx_labels(G_nx, pos, font_size=8)
                nx.draw_networkx_edges(G_nx, pos, width=1.0, alpha=0.6)
                plt.title("CTI Knowledge Graph", fontsize=14)
                st.pyplot(fig)
            else:
                st.info("No relationships found to visualize.")
        except Exception as e:
            st.error("Graph rendering failed.")
            st.text(traceback.format_exc())

else:
    st.info("Upload a CTI report to begin analysis.")

st.divider()
st.caption("Cyber Threat Intelligence Suite — Powered by SecureBERT-NER | SentenceTransformer | igraph | 2025")
