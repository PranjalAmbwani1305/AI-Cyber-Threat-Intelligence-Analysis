import streamlit as st
import pandas as pd
from CTI import extract_threat_intel
from NER_Transformer import extract_entities
from Clustering_Prototype import perform_clustering
from Knowledge_graph_gradio_app import build_knowledge_graph
from Text_classification_Sentiment import analyze_sentiment
from Topic_Modelling_Sentence_Analysis_ import topic_modelling_analysis

st.set_page_config(
    page_title="Cyber Threat Intelligence (CTI) Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Cyber Threat Intelligence (CTI) Dashboard")

st.sidebar.header("File Upload")
uploaded_file = st.sidebar.file_uploader("Upload Threat Feed (.csv or .pdf)", type=["csv", "pdf"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = extract_threat_intel(uploaded_file)

    st.sidebar.success(f"File uploaded: {uploaded_file.name}")
else:
    st.warning("Please upload a threat feed (.csv or .pdf) to continue.")
    st.stop()

tabs = st.tabs(["Overview", "NER", "Knowledge Graph", "Threat Feed", "Clustering", "Sentiment", "Topic Modelling"])

with tabs[0]:
    st.subheader("Overview")
    st.write("### File Summary")
    st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    st.dataframe(df.head())

with tabs[1]:
    st.subheader("Named Entity Recognition (NER)")
    entities = extract_entities(df)
    st.dataframe(entities)

with tabs[2]:
    st.subheader("Knowledge Graph")
    graph_fig = build_knowledge_graph(df)
    st.pyplot(graph_fig)

with tabs[3]:
    st.subheader("Threat Feed")
    indicator_type = st.selectbox("Indicator Type", ["All"] + list(df["Type"].unique()))
    confidence_filter = st.slider("Confidence Level", 0, 100, (0, 100))
    filtered_df = df[
        (df["Confidence"].between(confidence_filter[0], confidence_filter[1])) &
        ((df["Type"] == indicator_type) | (indicator_type == "All"))
    ]
    st.dataframe(filtered_df)
    st.write(f"Indicator Count: {len(filtered_df)}")
    st.write(f"Average Confidence: {filtered_df['Confidence'].mean():.2f}")

with tabs[4]:
    st.subheader("Clustering Analysis")
    cluster_fig = perform_clustering(df)
    st.pyplot(cluster_fig)

with tabs[5]:
    st.subheader("Sentiment Analysis")
    sentiment_df = analyze_sentiment(df)
    st.dataframe(sentiment_df)

with tabs[6]:
    st.subheader("Topic Modelling & Sentence Analysis")
    topic_results = topic_modelling_analysis(df)
    st.dataframe(topic_results)
