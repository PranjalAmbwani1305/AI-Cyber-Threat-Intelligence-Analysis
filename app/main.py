import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from nlp_logic import (
    load_models,
    extract_text_from_file,
    split_into_sentences,
    run_ner_on_text,
    sentiment_analysis,
    topic_modeling,
    perform_clustering,
    build_cti_graph,
    plot_cti_graph_matplotlib,
    simple_summarize
)

st.set_page_config(page_title="CTI Knowledge Graph Builder", layout="wide", initial_sidebar_state="expanded")
# no page_icon set (user requested removed icon)

# ---- initialize models once ----
@st.cache_resource(show_spinner=False)
def init():
    return load_models()

models = init()
ner_pipe = models.get("ner_pipeline")
sentiment_pipe = models.get("sentiment_pipeline")
embedding_model = models.get("embedding_model")

# ---- Sidebar ----
st.sidebar.header("Upload CTI Report")
uploaded = st.sidebar.file_uploader("Upload CSV / PDF / TXT", type=["csv", "pdf", "txt", "xlsx"])
st.sidebar.markdown("**Options**")
use_sample = st.sidebar.checkbox("Use sample dataset (simple feed)", value=False)
st.sidebar.markdown("---")
st.sidebar.markdown("Model status:")
st.sidebar.write("- NER: " + ("Loaded" if ner_pipe else "Fallback"))
st.sidebar.write("- Sentiment: " + ("Loaded" if sentiment_pipe else "Fallback"))
st.sidebar.write("- Embeddings: " + ("Loaded" if embedding_model else "Fallback")

# ---- Sample data (small) for Threat Feed table ----
SAMPLE_CSV = """Indicator,Type,Confidence,Source
105.4.302.40,Domain,1.8,True
125.8.33.228,Domain,1.9,False
200.200.0.86,Domain,100,True
231.407.96.197,Malware,100,False
222.0.13.1,Domain,100,True
"""

# ---- Helper to load & parse uploaded file ----
@st.cache_data(show_spinner=False)
def load_and_extract(file):
    text = extract_text_from_file(file)
    sentences = split_into_sentences(text)
    return text, sentences

# ---- Page Layout ----
st.title("Cyber Threat Intelligence (CTI) Analytics")
st.markdown("Upload a CTI report (PDF/CSV/TXT) and get NER, topics, clustering and a CTI knowledge graph.")

# ---- Top-level process button ----
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("### Input")
    st.write("Upload a CTI report on the left then press **Process file** below.")
with col2:
    process_btn = st.button("Process file", type="primary")

# ---- decide file source ----
if use_sample:
    st.info("Using sample threat feed as input (small CSV).")
    df_sample = pd.read_csv(pd.compat.StringIO(SAMPLE_CSV))
    text_input = df_sample.astype(str).values.flatten()
    combined_text = " ".join(text_input)
    uploaded_file = None
else:
    if uploaded is None:
        st.info("No file uploaded yet. Upload a CSV, PDF, TXT, or choose sample.")
        combined_text = ""
    else:
        combined_text = ""
        try:
            combined_text, _ = load_and_extract(uploaded)
        except Exception as e:
            st.error(f"Failed to extract text: {e}")
            combined_text = ""

# ---- Process file when button is clicked ----
if process_btn:
    if not combined_text:
        st.error("No text available to process. Upload a file or choose sample.")
    else:
        with st.spinner("Running NER, topic modeling, clustering, and building graph..."):
            # Sentences
            sentences = split_into_sentences(combined_text)
            # NER
            ner_results = run_ner_on_text(combined_text, ner_pipe)
            ner_df = pd.DataFrame(ner_results)
            if not ner_df.empty:
                ner_df = ner_df.rename(columns={"word": "Entity", "entity_group": "Type", "score": "Score"})
                ner_df["Score"] = ner_df["Score"].astype(float).round(3)

            # sentiment
            senti_df = sentiment_analysis(sentences, sentiment_pipe, limit=200)

            # topics
            topics_map, topic_assign = topic_modeling(sentences, n_topics=6)

            # clustering
            emb, cluster_labels = perform_clustering(sentences, embedding_model, eps=0.9, min_samples=2)

            # graph
            ent_list = ner_df["Entity"].tolist() if not ner_df.empty else []
            type_list = ner_df["Type"].tolist() if not ner_df.empty else []
            G = build_cti_graph(ent_list, type_list)

            # summarization (simple)
            summary = simple_summarize(combined_text, n_sentences=4)

        st.success("Processing complete — check the tabs for results.")

        # Put outputs into session state so tabs can access without reprocessing
        st.session_state["ner_df"] = ner_df
        st.session_state["sentiment_df"] = senti_df
        st.session_state["topics_map"] = topics_map
        st.session_state["topic_assign"] = topic_assign
        st.session_state["cluster_labels"] = cluster_labels
        st.session_state["graph"] = G
        st.session_state["sentences"] = sentences
        st.session_state["summary"] = summary

# ---- Tabs ----
tabs = st.tabs(["Overview", "NER", "Sentiment & Summary", "Topics", "Clustering", "Knowledge Graph", "Threat Feed"])

# Overview
with tabs[0]:
    st.header("Overview")
    st.write("Quick summary and metadata of the processed report.")
    if "summary" in st.session_state:
        st.subheader("Extractive Summary")
        st.write(st.session_state["summary"])
        st.markdown("---")
        st.write(f"Total sentences extracted: {len(st.session_state.get('sentences', []))}")
        st.write(f"Total unique entities (NER): {len(st.session_state.get('ner_df', pd.DataFrame()))}")
    else:
        st.info("No processed data yet. Upload and press **Process file**.")

# NER tab
with tabs[1]:
    st.header("Named Entity Recognition (CTI-focused)")
    if "ner_df" not in st.session_state or st.session_state["ner_df"].empty:
        st.info("No NER results available. Process a file first.")
    else:
        df = st.session_state["ner_df"]
        st.dataframe(df[["Entity", "Type", "Score"]].drop_duplicates().reset_index(drop=True), use_container_width=True)
        st.markdown("**Top entity types**")
        st.bar_chart(df["Type"].value_counts())

# Sentiment & Summary
with tabs[2]:
    st.header("Sentiment / Quick Summary")
    if "sentiment_df" not in st.session_state:
        st.info("No results. Process a file first.")
    else:
        st.subheader("Sentence-level sentiment summary (top 200 sentences)")
        st.dataframe(st.session_state["sentiment_df"].head(200)[["label", "score", "text"]])
        st.markdown("---")
        st.subheader("Extractive summary")
        st.write(st.session_state.get("summary", "—"))

# Topics
with tabs[3]:
    st.header("Topic Modeling (TF-IDF + NMF)")
    if "topics_map" not in st.session_state:
        st.info("No topics yet. Process a file first.")
    else:
        topics_map = st.session_state["topics_map"]
        if not topics_map:
            st.warning("Topic modeling produced no topics.")
        else:
            for tid, kws in topics_map.items():
                st.markdown(f"**Topic {tid}** – {', '.join(kws[:8])}")

# Clustering
with tabs[4]:
    st.header("Sentence Clustering")
    if "cluster_labels" not in st.session_state:
        st.info("No clusters. Process a file first.")
    else:
        labels = st.session_state["cluster_labels"]
        sents = st.session_state.get("sentences", [])
        cluster_df = pd.DataFrame({"sentence": sents, "cluster": labels})
        st.dataframe(cluster_df.groupby("cluster").size().reset_index(name="count").sort_values("count", ascending=False))
        st.markdown("Sample sentences per cluster (first 5 clusters):")
        for cl in sorted(set(labels))[:6]:
            st.write(f"Cluster {cl}")
            st.write(cluster_df[cluster_df["cluster"] == cl]["sentence"].head(5).to_list())

# Knowledge Graph
with tabs[5]:
    st.header("CTI Knowledge Graph")
    if "graph" not in st.session_state or st.session_state["graph"].number_of_nodes() == 0:
        st.info("No graph to show. Process a file with entities first.")
    else:
        G = st.session_state["graph"]
        fig = plot_cti_graph_matplotlib(G, figsize=(12, 8))
        st.pyplot(fig)

        st.markdown("Graph nodes by type:")
        tcounts = {}
        for n, d in G.nodes(data=True):
            t = d.get("ctype", "MISC")
            tcounts[t] = tcounts.get(t, 0) + 1
        st.table(pd.DataFrame(list(tcounts.items()), columns=["Type", "Count"]))

# Threat Feed (table)
with tabs[6]:
    st.header("Threat Feed / Indicators table")
    # If file was CSV originally, try to display it in structured table mode
    try:
        if use_sample:
            df_feed = pd.read_csv(pd.compat.StringIO(SAMPLE_CSV))
        elif uploaded and uploaded.name.lower().endswith(".csv"):
            df_feed = pd.read_csv(uploaded)
        else:
            # build from entities as a fallback (entities -> type -> score)
            ner = st.session_state.get("ner_df", pd.DataFrame())
            if ner is None or ner.empty:
                st.info("No feed to show. Upload a CSV or process a file.")
                df_feed = pd.DataFrame()
            else:
                df_feed = ner.rename(columns={"Entity": "Indicator", "Type": "Type", "Score": "Confidence"})[["Indicator", "Type", "Confidence"]]
        if df_feed is not None and not df_feed.empty:
            st.dataframe(df_feed.head(200), use_container_width=True)
            st.write("Indicator Count", len(df_feed))
            if "Confidence" in df_feed.columns:
                avg_conf = pd.to_numeric(df_feed["Confidence"], errors="coerce").mean()
                st.write("Average Confidence", round(float(avg_conf) if not math.isnan(avg_conf) else 0.0, 2))
    except Exception as e:
        st.error(f"Failed to load feed: {e}")
