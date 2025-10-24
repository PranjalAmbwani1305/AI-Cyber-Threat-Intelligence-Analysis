import streamlit as st

# Import your functions from the converted .py files
from NER_Transformer import extract_entities
from Clustering_Prototype import cluster_documents
from Topic_Modelling_+_Sentence_Analysis_ import extract_topics
from Text_classification_+_Sentiment import analyze_sentiment
from Knowledge_graph_gradio_app import build_graph, visualize_graph

st.set_page_config(page_title="Project ATHENA - CTI Workbench", layout="wide")
st.title("Project ATHENA - Cyber Threat Intelligence Workbench")

# --------------Sidebar-------------# 
st.sidebar.header("Upload Data")
uploaded_cti = st.sidebar.file_uploader("Upload CTI Reports (.txt/.pdf)", type=["txt", "pdf"])
uploaded_logs = st.sidebar.file_uploader("Upload Network Logs (.csv/.json)", type=["csv", "json"])

st.sidebar.header("Select Analyses")
analyses = st.sidebar.multiselect(
    "Choose analyses to run",
    ["NER", "Clustering", "Topic Modeling", "Sentiment", "Knowledge Graph"],
    default=["NER"]
)

st.sidebar.header("AI Analyst Assistant (optional)")
ai_query = st.sidebar.text_input("Ask AI:")

# --------------Run Analysis------------# 

if st.sidebar.button("Run Analysis"):
    text_data = ""
    if uploaded_cti:
        text_data = uploaded_cti.read().decode("utf-8")

    # 1. NER
    entities = extract_entities(text_data) if "NER" in analyses else []

    # 2. Clustering
    clusters = cluster_documents(text_data) if "Clustering" in analyses else {}

    # 3. Topic Modeling
    topics = extract_topics(text_data) if "Topic Modeling" in analyses else None

    # 4. Sentiment
    sentiment = analyze_sentiment(text_data) if "Sentiment" in analyses else []

    # 5. Knowledge Graph
    graph_data = build_graph(entities) if "Knowledge Graph" in analyses else None

    # -------------- Display Results in Tabs---------------#
   
    
    tabs = st.tabs(["Knowledge Graph", "Dashboard", "Entity Overview", "AI Assistant"])

    # Knowledge Graph Tab
    with tabs[0]:
        st.subheader("Knowledge Graph")
        if graph_data:
            visualize_graph(graph_data)
        else:
            st.write("No entities to visualize.")

    # Dashboard Tab
    with tabs[1]:
        st.subheader("Dashboard Analytics")
        if clusters:
            st.write("Clusters:", clusters)
        if topics is not None:
            st.write("Topics:", topics)
        if sentiment:
            st.write("Sentiment:", sentiment)

    # Entity Overview Tab
    with tabs[2]:
        st.subheader("Entity Overview")
        st.write(entities if entities else "No entities extracted.")

    # AI Assistant Tab
    with tabs[3]:
        st.subheader("AI Analyst Assistant")
        if ai_query:
            # Dummy AI response; integrate your CTI_LLM.py logic if needed
            st.write(f"AI response to: {ai_query}")
        else:
            st.write("Enter a query in the sidebar.")
