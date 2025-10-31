import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import igraph as ig
from nlp_logic import extract_pdf_text, process_cti_pdf, split_into_sentences, perform_clustering, build_cti_graph

# ------------- STREAMLIT CONFIG -------------
st.set_page_config(
    page_title="AI Cyber Threat Intelligence Dashboard",
    layout="wide"
)

# ------------- SIDEBAR ----------------------
st.sidebar.title("📄 Upload CTI Report")
uploaded_file = st.sidebar.file_uploader(
    "Upload a CTI file (PDF or CSV):",
    type=["pdf", "csv", "txt", "xlsx"]
)
st.sidebar.markdown("**Limit 200MB** • Supported formats: PDF, CSV, TXT, XLSX")

# ------------- HEADER -----------------------
st.title("🧠 AI Cyber Threat Intelligence Dashboard")
st.markdown("""
Transform **Cyber Threat Intelligence (CTI)** data — structured or unstructured —
into actionable insights with NLP, entity recognition, clustering, and knowledge graph analysis.
""")

# ------------- MAIN PROCESS -----------------
if uploaded_file:
    st.info(f"📂 Processing `{uploaded_file.name}`...")

    try:
        # Detect file type
        if uploaded_file.name.lower().endswith(".pdf"):
            text = extract_pdf_text(uploaded_file)
            st.success("✅ PDF content extracted successfully.")
        elif uploaded_file.name.lower().endswith((".csv", ".xlsx")):
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            text_columns = [col for col in df.columns if df[col].dtype == "object"]
            text = " ".join(df[text_columns].fillna("").astype(str).values.flatten())
            st.success("✅ Structured data combined for NLP analysis.")
        else:
            text = uploaded_file.read().decode("utf-8")
            st.success("✅ Text file processed successfully.")

        # ---- NLP PIPELINE ----
        sentences = split_into_sentences(text)
        entities_df, entity_labels = process_cti_pdf(uploaded_file)
        cluster_labels, topic_map = perform_clustering(sentences)
        graph = build_cti_graph(entities_df, entity_labels)

        # ---- METRICS ----
        col1, col2, col3 = st.columns(3)
        col1.metric("🧩 Entities Extracted", len(entities_df))
        col2.metric("🗒️ Sentences Processed", len(sentences))
        col3.metric("🔗 Relationships", len(graph.es))

        st.divider()

        # ---- ENTITIES ----
        st.subheader("🧠 Extracted Entities")
        if not entities_df.empty:
            st.dataframe(entities_df, use_container_width=True)
            st.bar_chart(entities_df["Type"].value_counts())
        else:
            st.warning("No named entities were detected.")

        st.divider()

        # ---- CLUSTERING ----
        st.subheader("🧮 Sentence Clustering")
        if cluster_labels:
            cluster_data = pd.DataFrame({
                "Sentence": sentences,
                "Cluster": cluster_labels
            })
            st.dataframe(cluster_data.head(50), use_container_width=True)
        else:
            st.info("Not enough content for clustering.")

        st.divider()

        # ---- KNOWLEDGE GRAPH ----
        st.subheader("🌐 Cyber Threat Knowledge Graph")

        if len(graph.vs) > 0:
            layout = graph.layout_fruchterman_reingold(niter=200)
            fig, ax = plt.subplots(figsize=(11, 8))
            ig.plot(
                graph,
                target=ax,
                layout=layout,
                vertex_label=graph.vs["name"],
                vertex_color=graph.vs["color"],
                vertex_size=45,
                edge_color="gray",
                edge_width=1.8,
                edge_arrow_size=0.5,
                vertex_label_size=10,
                bbox=(900, 700),
                margin=80
            )

            ax.set_facecolor("#f8f9fa")
            fig.patch.set_facecolor("#f8f9fa")
            ax.axis("off")

            # --- Legend for entity types ---
            unique_types = list(set(graph.vs["node_type"]))
            colors = [graph.vs.select(node_type=ut)["color"][0] for ut in unique_types]
            handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, label=ut, markersize=10)
                       for ut, c in zip(unique_types, colors)]
            ax.legend(handles=handles, title="Entity Types", loc="upper left", fontsize=9, frameon=False)
            ax.set_title("🔍 CTI Knowledge Graph", fontsize=16, fontweight='bold', pad=20)

            st.pyplot(fig)
        else:
            st.info("Graph could not be generated — insufficient relationship data.")

        st.divider()
        st.subheader("📜 Raw Extracted Text")
        st.text_area("Extracted Text", text[:5000], height=300)

    except Exception as e:
        st.error("❌ An error occurred during report processing.")
        st.exception(e)

else:
    st.info("Please upload a CTI file to start analysis.")

# --- FOOTER ---
st.markdown("---")
st.caption("AI Cyber Threat Intelligence Dashboard — SecureBERT & SentenceTransformer © 2025")
