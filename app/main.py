import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import igraph as ig
import re
from io import StringIO
from nlp_logic import process_cti_pdf, perform_clustering, build_cti_graph

# ------------------ STREAMLIT CONFIG ------------------
st.set_page_config(
    page_title="AI Cyber Threat Intelligence Dashboard",
    layout="wide"
)

# ------------------ SIDEBAR ------------------
st.sidebar.title("Upload CTI Report")
uploaded_file = st.sidebar.file_uploader(
    "Choose a CTI dataset (CSV, PDF, or TXT):",
    type=["csv", "pdf", "txt"]
)
st.sidebar.markdown("Limit 200MB • PDF, CSV, TXT")

st.title("AI Cyber Threat Intelligence Dashboard")
st.markdown(
    "Transform **structured or unstructured Cyber Threat Intelligence (CTI)** data into actionable insights using "
    "entity extraction, semantic clustering, and knowledge graph analytics."
)

# ------------------ HELPER FUNCTIONS ------------------
def extract_structured_entities(df: pd.DataFrame):
    """Extract entities (IPs, domains, hashes, emails, etc.) from structured CSV."""
    text_data = " ".join(df.astype(str).fillna("").values.flatten())

    patterns = {
        "IP Address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        "Domain": r"\b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
        "Hash": r"\b[a-fA-F0-9]{32,64}\b",
        "Email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "URL": r"http[s]?://[^\s]+"
    }

    extracted = []
    for name, pattern in patterns.items():
        matches = re.findall(pattern, text_data)
        for m in set(matches):
            extracted.append({"Entity": m, "Type": name})
    
    return pd.DataFrame(extracted)

# ------------------ MAIN LOGIC ------------------
if uploaded_file:
    st.info(f"Processing `{uploaded_file.name}`...")

    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            st.success("✅ Structured data loaded successfully.")

            # --- Extract entities from structured data ---
            entities_df = extract_structured_entities(df)

            # Combine text for clustering
            text_columns = [col for col in df.columns if df[col].dtype == "object"]
            combined_text = " ".join(df[text_columns].fillna("").astype(str).values.flatten())
            sentences = combined_text.split(". ")

            # --- Build Knowledge Graph ---
            graph = build_cti_graph(entities_df)

        else:
            # For unstructured data (PDF, TXT)
            result = process_cti_pdf(uploaded_file)
            entities_df = result.get("entities", pd.DataFrame())
            sentences = result.get("sentences", [])
            graph = result.get("graph", ig.Graph())

        # --- Summary ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Entities Extracted", len(entities_df))
        col2.metric("Sentences Processed", len(sentences))
        col3.metric("Relationships Identified", len(graph.es))

        st.divider()

        # --- ENTITY SECTION ---
        st.subheader("Entities Extracted")
        if not entities_df.empty:
            st.dataframe(entities_df, use_container_width=True)
            with st.expander("Entity Type Distribution"):
                st.bar_chart(entities_df["Type"].value_counts())
        else:
            st.warning("No entities found in this dataset.")

        # --- CLUSTERING SECTION ---
        st.subheader("Sentence Clustering")
        if len(sentences) > 3:
            _, cluster_labels, topic_map = perform_clustering(sentences)
            cluster_data = pd.DataFrame({
                "Sentence": sentences,
                "Cluster": cluster_labels
            })
            st.dataframe(cluster_data.head(50), use_container_width=True)
        else:
            st.info("Not enough text for clustering.")

        # --- KNOWLEDGE GRAPH SECTION ---
        st.subheader("Cyber Threat Knowledge Graph")
        if len(graph.vs) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ig.plot(
                graph,
                target=ax,
                vertex_label=graph.vs["name"],
                vertex_color=graph.vs["color"],
                vertex_size=20,
                edge_arrow_size=0.5,
                edge_color="gray"
            )
            st.pyplot(fig)
        else:
            st.info("Graph could not be generated — insufficient relationship data.")

        # --- RAW TEXT PREVIEW ---
        with st.expander("📜 Raw Text Preview"):
            st.text_area("Extracted Text", combined_text[:5000] if 'combined_text' in locals() else "", height=300)

    except Exception as e:
        st.error("An error occurred while processing the report.")
        st.exception(e)

else:
    st.info("Please upload a CTI dataset (CSV, PDF, or TXT) to begin analysis.")

st.divider()
st.caption("AI Cyber Threat Intelligence Dashboard — SecureBERT & SentenceTransformer © 2025")
