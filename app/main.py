import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from io import BytesIO

# Import your NLP logic
from nlp_logic import (
    extract_pdf_text,
    process_cti_pdf,
    split_into_sentences,
    perform_clustering,
    build_cti_graph
)

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Cyber Threat Intelligence Dashboard", layout="wide")

st.title("AI Cyber Threat Intelligence Dashboard")
st.write("Transform unstructured or structured Cyber Threat Intelligence (CTI) data into actionable insights using advanced NLP, clustering, and knowledge graph analysis.")

# ---------------- SIDEBAR ----------------
st.sidebar.header("Upload CTI Report")
uploaded_file = st.sidebar.file_uploader(
    "Choose a report file", 
    type=["pdf", "csv", "xlsx", "txt"],
    help="Supports structured logs (CSV/XLSX) and unstructured text (PDF/TXT)"
)

# ---------------- HANDLER ----------------
if uploaded_file:
    st.info(f"Processing `{uploaded_file.name}`...")

    # --- TEXT EXTRACTION ---
    if uploaded_file.name.endswith(".pdf"):
        text = extract_pdf_text(uploaded_file)
        mode = "unstructured"
    elif uploaded_file.name.endswith(".txt"):
        text = uploaded_file.read().decode("utf-8", errors="ignore")
        mode = "unstructured"
    elif uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        text = " ".join(df.astype(str).values.flatten())
        mode = "structured"
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
        text = " ".join(df.astype(str).values.flatten())
        mode = "structured"
    else:
        st.error("Unsupported file type.")
        st.stop()

    if not text.strip():
        st.error("No readable text found.")
        st.stop()

    st.success(f"✅ Text extracted successfully from {mode.upper()} data.")
    st.download_button("Download Extracted Text", text, file_name="extracted_text.txt")

    # --- PROCESSING ---
    with st.spinner("Running NLP and graph analysis..."):
        result = process_cti_pdf(BytesIO(uploaded_file.read()) if uploaded_file.name.endswith(".pdf") else uploaded_file)

    # --- FALLBACK for structured data (no NER results) ---
    if mode == "structured" and (result["entities"].empty or len(result["graph"].vs) == 0):
        entities = []
        if 'User' in df.columns:
            entities += df['User'].dropna().unique().tolist()
        if 'Src_IP' in df.columns:
            entities += df['Src_IP'].dropna().unique().tolist()
        if 'Dest_IP' in df.columns:
            entities += df['Dest_IP'].dropna().unique().tolist()
        if 'Description' in df.columns:
            entities += [w for w in " ".join(df['Description'].astype(str)).split() if len(w) > 6]

        # Generate synthetic entity table
        structured_df = pd.DataFrame({
            "Entity": pd.Series(entities[:200]),
            "Type": ["STRUCTURED_ENTITY"] * min(200, len(entities)),
            "Score": [1.0] * min(200, len(entities))
        })
        result["entities"] = structured_df
        result["graph"] = build_cti_graph(structured_df["Entity"].tolist(), structured_df["Type"].tolist())

    # --- SUMMARY ---
    st.subheader("Analysis Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Entities Extracted", len(result["entities"]))
    col2.metric("Sentences Processed", len(result["sentences"]))
    col3.metric("Relationships Identified", len(result["graph"].es))

    # ---------------- TABS ----------------
    tab1, tab2, tab3, tab4 = st.tabs(["Entities", "Sentence Clustering", "Knowledge Graph", "Raw Text"])

    # --- ENTITIES ---
    with tab1:
        st.subheader("Entities Extracted")
        entities_df = result["entities"]
        if not entities_df.empty:
            st.dataframe(entities_df, use_container_width=True)
            st.bar_chart(entities_df["Type"].value_counts())
        else:
            st.info("No entities found.")
    
    # --- CLUSTERING ---
    with tab2:
        st.subheader("Sentence Clustering")
        _, cluster_labels, topic_map = perform_clustering(result["sentences"])
        if cluster_labels is not None:
            cluster_df = pd.DataFrame({
                "Sentence": result["sentences"],
                "Cluster": cluster_labels
            })
            st.dataframe(cluster_df, use_container_width=True)
        else:
            st.info("No clusters detected.")

    # --- KNOWLEDGE GRAPH ---
    with tab3:
        st.subheader("Cyber Threat Knowledge Graph")
        G = result["graph"]
        if len(G.vs) > 0:
            nxG = nx.Graph()
            for v in G.vs:
                nxG.add_node(v["name"], label=v["type"])
            for e in G.es:
                src, tgt = e.tuple
                nxG.add_edge(G.vs[src]["name"], G.vs[tgt]["name"], label=e["label"])

            fig, ax = plt.subplots(figsize=(10, 6))
            nx.draw_networkx(
                nxG, ax=ax, with_labels=True, 
                node_color="#a6cee3", edge_color="#999999", font_size=8
            )
            st.pyplot(fig)
        else:
            st.warning("Graph could not be generated — insufficient relationship data.")

    # --- RAW TEXT ---
    with tab4:
        st.subheader("Raw Extracted Text")
        st.text_area("Extracted Text Preview", text[:5000], height=300)
        st.download_button("Download Full Text", text, file_name="raw_text.txt")

else:
    st.info("Upload a CTI report from the sidebar to start analysis.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("AI Cyber Threat Intelligence Dashboard — SecureBERT & SentenceTransformer © 2025")
