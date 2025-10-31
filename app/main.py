import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import igraph as ig
from nlp_logic import process_cti_pdf, perform_clustering, build_cti_graph

# ------------------ STREAMLIT CONFIG ------------------
st.set_page_config(
    page_title="AI Cyber Threat Intelligence Dashboard",
    layout="wide"
)

# ------------------ SIDEBAR ------------------
st.sidebar.title("Upload CTI Report")
uploaded_file = st.sidebar.file_uploader(
    "Choose a structured CTI dataset (CSV):",
    type=["csv"]
)

st.sidebar.markdown("Limit 200MB • CSV only")

st.title("AI Cyber Threat Intelligence Dashboard")
st.markdown(
    "Transform **structured Cyber Threat Intelligence (CTI) data** into actionable insights using "
    "entity extraction, semantic clustering, and graph analytics."
)

# ------------------ MAIN LOGIC ------------------
if uploaded_file:
    st.info(f"Processing `{uploaded_file.name}`...")

    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Text extracted successfully from STRUCTURED data.")

        # Combine text-like columns
        text_columns = [col for col in df.columns if df[col].dtype == "object"]
        combined_text = " ".join(df[text_columns].fillna("").astype(str).values.flatten())

        # --- PROCESS ---
        result = process_cti_pdf(uploaded_file)  # Reusing NLP pipeline for structured content

        entities_df = result.get("entities", pd.DataFrame())
        sentences = result.get("sentences", [])
        graph = result.get("graph", ig.Graph())
        cluster_labels = result.get("cluster_labels", [])
        topic_map = result.get("topic_map", {})

        # --- SUMMARY METRICS ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Entities Extracted", len(entities_df))
        col2.metric("Sentences Processed", len(sentences))
        col3.metric("Relationships Identified", len(graph.es))

        st.divider()

        # --- ENTITY SECTION ---
        st.subheader("Entities Extracted")
        if not entities_df.empty:
            st.dataframe(entities_df, use_container_width=True)
            st.bar_chart(entities_df["Type"].value_counts())
        else:
            st.warning("No named entities found in this dataset.")

        # --- CLUSTERING SECTION ---
        st.subheader("Sentence Clustering")
        if sentences:
            _, cluster_labels, topic_map = perform_clustering(sentences)
            cluster_data = pd.DataFrame({
                "Sentence": sentences,
                "Cluster": cluster_labels
            })
            st.dataframe(cluster_data.head(50), use_container_width=True)
        else:
            st.info("Not enough text content for clustering.")

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

        # --- RAW TEXT ---
        with st.expander("📜 Raw Text Preview"):
            st.text_area("Extracted Text", combined_text[:5000], height=300)

    except Exception as e:
        st.error("An error occurred while processing the report.")
        st.exception(e)

else:
    st.info("Please upload a structured CTI dataset (CSV) to begin analysis.")

st.divider()
st.caption("AI Cyber Threat Intelligence Dashboard — SecureBERT & SentenceTransformer © 2025")
