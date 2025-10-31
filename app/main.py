import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from nlp_logic import process_cti_pdf, perform_clustering, build_cti_graph, plot_knowledge_graph

st.set_page_config(page_title="AI Cyber Threat Intelligence Dashboard", layout="wide")

# ------------------ SIDEBAR ------------------
st.sidebar.title("Upload CTI Report")
uploaded_file = st.sidebar.file_uploader(
    "Upload structured or unstructured CTI data",
    type=["csv", "pdf", "txt", "xlsx"]
)
st.sidebar.markdown("Limit 200MB • CSV, PDF, TXT, XLSX")

st.title("AI Cyber Threat Intelligence Dashboard")
st.markdown(
    "Transform **structured or unstructured Cyber Threat Intelligence (CTI)** data "
    "into actionable insights using advanced NLP, clustering, and knowledge graph analytics."
)

# ------------------ MAIN APP LOGIC ------------------
if uploaded_file:
    st.info(f"Processing `{uploaded_file.name}`...")

    try:
        result = process_cti_pdf(uploaded_file)

        if "error" in result:
            st.error(result["error"])
        else:
            text = result["text"]
            sentences = result["sentences"]
            entities_df = result["entities"]

            st.success("✅ Report processed successfully!")

            # --- METRICS ---
            col1, col2, col3 = st.columns(3)
            col1.metric("Entities Extracted", len(entities_df))
            col2.metric("Sentences Processed", len(sentences))
            col3.metric("Unique Entities", len(entities_df["Entity"].unique()))

            st.divider()

            # --- ENTITIES TABLE ---
            st.subheader("🧩 Extracted Entities")
            if not entities_df.empty:
                st.dataframe(entities_df, use_container_width=True)
                st.bar_chart(entities_df["Type"].value_counts())
            else:
                st.warning("No entities found.")

            # --- CLUSTERING ---
            st.subheader("🧠 Sentence Clustering")
            if sentences:
                _, labels, topic_map = perform_clustering(sentences)
                cluster_data = pd.DataFrame({
                    "Sentence": sentences,
                    "Cluster": labels
                })
                st.dataframe(cluster_data.head(50), use_container_width=True)
            else:
                st.info("Not enough text for clustering.")

            # --- KNOWLEDGE GRAPH ---
            st.subheader("🌐 Cyber Threat Knowledge Graph")
            graph = build_cti_graph(entities_df)
            fig = plot_knowledge_graph(graph)

            if fig:
                st.pyplot(fig)
            else:
                st.warning("Graph could not be generated — insufficient relationships.")

            # --- RAW TEXT ---
            with st.expander("📜 Raw Text Preview"):
                st.text_area("Extracted Text", text[:4000], height=250)

    except Exception as e:
        st.error("❌ An error occurred while processing the report.")
        st.exception(e)
else:
    st.info("Please upload a CTI dataset (CSV/PDF/TXT/XLSX) to begin.")

st.divider()
st.caption("AI Cyber Threat Intelligence Dashboard — SecureBERT & SentenceTransformer © 2025")
