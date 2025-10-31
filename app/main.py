import streamlit as st
import matplotlib.pyplot as plt
import igraph as ig
import pandas as pd

try:
    from app.nlp_logic import extract_pdf_text, split_into_sentences, perform_clustering, build_cti_graph, process_cti_data
except ModuleNotFoundError:
    from nlp_logic import extract_pdf_text, split_into_sentences, perform_clustering, build_cti_graph, process_cti_data

# ------------------ UI CONFIG ------------------
st.set_page_config(page_title="AI Cyber Threat Intelligence Dashboard", layout="wide")
st.title("🧠 AI Cyber Threat Intelligence Dashboard")
st.markdown("""
Transform **Cyber Threat Intelligence (CTI)** data — structured or unstructured —  
into actionable insights using entity extraction, clustering, and knowledge graph analytics.
""")

# ------------------ SIDEBAR ------------------
st.sidebar.header("📂 Upload CTI Report")
uploaded_file = st.sidebar.file_uploader("Choose a CTI report", type=["csv", "pdf", "txt"])
st.sidebar.markdown("Limit 200MB per file")
quick_mode = st.sidebar.checkbox("⚡ Quick Mode (faster loading)", value=True)

# ------------------ MAIN PROCESS ------------------
if uploaded_file:
    with st.spinner(f"Processing `{uploaded_file.name}`... Please wait ⏳"):
        try:
            result = process_cti_data(uploaded_file)

            entities_df = result["entities"]
            sentences = result["sentences"]
            graph = result["graph"]
            cluster_labels = result.get("cluster_labels", [])
            topic_map = result.get("topic_map", {})

            st.success("✅ CTI Analysis Completed Successfully!")

            c1, c2, c3 = st.columns(3)
            c1.metric("Entities Extracted", len(entities_df))
            c2.metric("Sentences Processed", len(sentences))
            c3.metric("Relationships Identified", len(graph.es))

            st.divider()
            st.subheader("📍 Extracted Entities")
            if not entities_df.empty:
                st.dataframe(entities_df.head(50), use_container_width=True)
                st.bar_chart(entities_df["Type"].value_counts())
            else:
                st.warning("No entities detected in the uploaded file.")

            st.subheader("🧩 Sentence Clustering")
            if sentences:
                cluster_df = pd.DataFrame({"Sentence": sentences, "Cluster": cluster_labels})
                st.dataframe(cluster_df.head(30), use_container_width=True)
            else:
                st.info("No textual data found for clustering.")

            st.subheader("🌐 Cyber Threat Knowledge Graph")
            if len(graph.vs) > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                layout = "fruchterman_reingold" if quick_mode else "kamada_kawai"
                ig.plot(
                    graph,
                    target=ax,
                    layout=graph.layout(layout),
                    vertex_color=graph.vs["color"],
                    vertex_label=graph.vs["label"],
                    vertex_size=20,
                    edge_color="gray",
                )
                st.pyplot(fig)
            else:
                st.warning("Graph could not be generated — no relationships found.")

            with st.expander("📜 Raw Text Preview"):
                preview = extract_pdf_text(uploaded_file)
                st.text_area("Extracted Text", preview[:2000], height=250)

        except Exception as e:
            st.error("❌ An error occurred during report processing.")
            st.exception(e)
else:
    st.info("Please upload a CTI dataset (CSV, PDF, or TXT) to begin analysis.")

st.caption("AI Cyber Threat Intelligence Dashboard — BERT & SentenceTransformer © 2025")
