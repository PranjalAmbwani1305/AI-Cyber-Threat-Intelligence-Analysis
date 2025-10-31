import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from io import BytesIO
import traceback
from app.nlp_logic import Textract_pdf_text, process_cti_pdf, split_into_sentences, perform_clustering, build_cti_graph


# ======= PAGE CONFIG =======
st.set_page_config(
    page_title="AI Cyber Threat Intelligence Dashboard",
    layout="wide",
)

# ======= SIDEBAR =======
with st.sidebar:
    st.title("Upload CTI Data")
    uploaded_file = st.file_uploader(
        "Select a report file",
        type=["pdf", "csv", "txt", "xlsx"],
        help="Upload unstructured (PDF/TXT) or structured (CSV/XLSX) cyber threat intelligence data."
    )
    st.markdown("---")
    st.caption("AI Cyber Threat Intelligence Pipeline")

# ======= MAIN HEADER =======
st.title("AI Cyber Threat Intelligence Dashboard")
st.markdown("""
This dashboard transforms raw cyber threat intelligence (CTI) data into structured insights 
using SecureBERT, Sentence Transformers, and Knowledge Graph analytics.
""")

# ======= PROCESS FILE =======
if uploaded_file:
    st.info(f"Processing `{uploaded_file.name}`...")

    try:
        # ---------- Extract Text Based on File Type ----------
        file_ext = uploaded_file.name.split(".")[-1].lower()
        text = ""

        if file_ext == "pdf":
            text = extract_pdf_text(uploaded_file)
        elif file_ext in ["csv", "xlsx"]:
            df = pd.read_csv(uploaded_file) if file_ext == "csv" else pd.read_excel(uploaded_file)
            text = " ".join(df.astype(str).values.flatten())
        elif file_ext == "txt":
            text = uploaded_file.read().decode("utf-8")
        else:
            st.error("Unsupported file format.")
            st.stop()

        # ---------- Split & Analyze ----------
        sentences = split_into_sentences(text)

        with st.spinner("Running SecureBERT and Sentence Clustering..."):
            result = process_cti_pdf(BytesIO(text.encode("utf-8")))
            entities_df = result["entities"]
            G = result["graph"]
            cluster_labels = result["cluster_labels"]
            topic_map = result["topic_map"]

        # ---------- Overview Metrics ----------
        st.subheader("Summary Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Extracted Entities", len(entities_df))
        col2.metric("Sentences", len(sentences))
        col3.metric("Relationships", G.ecount() if G else 0)

        # ---------- Tabs ----------
        tab1, tab2, tab3, tab4 = st.tabs([
            "Entities",
            "Sentence Clusters",
            "Knowledge Graph",
            "Raw Text"
        ])

        # --- TAB 1: Entities ---
        with tab1:
            st.subheader("Extracted Named Entities")
            if not entities_df.empty:
                st.dataframe(entities_df, use_container_width=True)
                st.bar_chart(entities_df["Type"].value_counts())
                st.download_button(
                    "Download Entities CSV",
                    entities_df.to_csv(index=False),
                    file_name="extracted_entities.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No entities detected in this file.")

        # --- TAB 2: Sentence Clusters ---
        with tab2:
            st.subheader("Semantic Clustering of Sentences")
            if cluster_labels is not None:
                clusters_df = pd.DataFrame({
                    "Sentence": sentences,
                    "Cluster": cluster_labels
                })
                st.dataframe(clusters_df.head(25), use_container_width=True)
                st.markdown("Cluster Distribution")
                st.bar_chart(pd.Series(cluster_labels).value_counts())
            else:
                st.info("No clustering information available.")

        # --- TAB 3: Knowledge Graph ---
        with tab3:
            st.subheader("Cyber Threat Knowledge Graph")
            if G and G.vcount() > 0:
                nx_graph = nx.Graph()
                for v in G.vs:
                    nx_graph.add_node(v["name"], type=v["type"])
                for e in G.es:
                    src, tgt = e.tuple
                    nx_graph.add_edge(G.vs[src]["name"], G.vs[tgt]["name"], relation=e["label"])

                fig, ax = plt.subplots(figsize=(10, 7))
                nx.draw(
                    nx_graph,
                    with_labels=True,
                    node_size=900,
                    font_size=8,
                    node_color="#42a5f5",
                    edge_color="#424242",
                    font_weight="bold"
                )
                st.pyplot(fig)
            else:
                st.warning("No graph could be generated.")

        # --- TAB 4: Text ---
        with tab4:
            st.subheader("Extracted Report Text")
            st.text_area("Text Preview", text[:6000], height=300)
            st.download_button(
                "Download Extracted Text",
                text,
                file_name="extracted_text.txt",
                mime="text/plain"
            )

    except Exception as e:
        st.error("Error during processing:")
        st.code(traceback.format_exc())

else:
    st.info("Upload a CTI report using the sidebar to begin analysis.")

# ======= FOOTER =======
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray'>© 2025 AI Cyber Threat Intelligence Suite | SecureBERT & SentenceTransformer Powered</div>",
    unsafe_allow_html=True,
)
