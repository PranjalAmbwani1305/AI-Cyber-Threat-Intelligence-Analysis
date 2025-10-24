import streamlit as st
from NER_Transformer import extract_entities_from_uploaded_file
from Clustering_Prototype import cluster_documents
from Topic_Modelling_Sentence_Analysis import extract_topics
from Text_classification_Sentiment import analyze_sentiment
from Knowledge_graph_gradio_app import build_graph, visualize_graph

st.set_page_config(page_title="Project ATHENA - CTI Workbench", layout="wide")
st.title("Project ATHENA - Cyber Threat Intelligence Workbench")

uploaded_file = st.file_uploader("Upload a PDF CTI report", type=["pdf"])
uploaded_logs = st.file_uploader("Upload Network Logs (.csv/.json)", type=["csv", "json"])
analyses = st.multiselect("Select analyses", ["NER", "Clustering", "Topic Modeling", "Sentiment", "Knowledge Graph"], default=["NER"])
ai_query = st.text_input("Ask AI:")

if st.button("Run Analysis"):
    text_data = ""
    entities = []

    if uploaded_file:
        entities = extract_entities_from_uploaded_file(uploaded_file)
        text_data = " ".join(entities)  # Use entities text for other analyses
    elif uploaded_logs:
        text_data = uploaded_logs.read().decode("utf-8")
        if "NER" in analyses:
            entities = extract_entities_from_uploaded_file(uploaded_logs)

    clusters = cluster_documents(text_data) if "Clustering" in analyses else {}
    topics = extract_topics(text_data) if "Topic Modeling" in analyses else None
    sentiment = analyze_sentiment(text_data) if "Sentiment" in analyses else []
    graph_data = build_graph(entities) if "Knowledge Graph" in analyses else None

    tabs = st.tabs(["Knowledge Graph", "Dashboard", "Entity Overview", "AI Assistant"])

    with tabs[0]:
        st.subheader("Knowledge Graph")
        if graph_data:
            visualize_graph(graph_data)
        else:
            st.write("No entities to visualize.")

    with tabs[1]:
        st.subheader("Dashboard Analytics")
        if clusters:
            st.write("Clusters:", clusters)
        if topics is not None:
            st.write("Topics:", topics)
        if sentiment:
            st.write("Sentiment:", sentiment)

    with tabs[2]:
        st.subheader("Entity Overview")
        st.write(entities if entities else "No entities extracted.")

    with tabs[3]:
        st.subheader("AI Analyst Assistant")
        if ai_query:
            st.write(f"AI response to: {ai_query}")
        else:
            st.write("Enter a query above.")
