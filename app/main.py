import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import igraph as ig
from PyPDF2 import PdfReader
import random
from nlp_logic import extract_entities, analyze_sentiment, topic_modeling

# ----------------------------------------
# PAGE CONFIG
# ----------------------------------------
st.set_page_config(
    page_title="Cyber Threat Intelligence Dashboard",
    layout="wide"
)

# ----------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------
def extract_text_from_pdf(file):
    """Extracts text from a PDF file."""
    text = ""
    try:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text.strip()

def build_sample_feed():
    """Generates sample threat intelligence data."""
    data = {
        "Indicator": [f"{random.randint(10,250)}.{random.randint(0,250)}.{random.randint(0,250)}.{random.randint(0,250)}" for _ in range(50)],
        "Type": random.choices(["Domain", "IP", "Malware", "URL", "CVE"], k=50),
        "Confidence": [round(random.uniform(0, 100), 1) for _ in range(50)],
        "Source": random.choices(["Feed A", "Feed B", "Feed C"], k=50)
    }
    return pd.DataFrame(data)

def draw_cti_graph():
    """Creates a simplified cybersecurity relationship graph."""
    nodes = ["Firewall", "Source IP", "Destination IP", "Protocol", "User", "Alert"]
    edges = [
        ("Firewall", "Destination IP", "blocks"),
        ("Firewall", "Source IP", "monitors"),
        ("Source IP", "User", "triggers"),
        ("Destination IP", "Protocol", "uses"),
        ("Protocol", "Alert", "initiates"),
        ("User", "Alert", "triggers")
    ]

    G = ig.Graph(directed=True)
    G.add_vertices(nodes)
    G.add_edges([(a, b) for a, b, _ in edges])
    G.es["label"] = [lbl for _, _, lbl in edges]

    layout = G.layout("tree")
    fig, ax = plt.subplots(figsize=(8, 6))
    ig.plot(
        G, target=ax, layout=layout,
        vertex_label=G.vs["name"], vertex_color="#90caf9",
        vertex_size=35, edge_label=G.es["label"]
    )
    return fig

# ----------------------------------------
# SIDEBAR CONTROLS
# ----------------------------------------
st.sidebar.title("⚙️ Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload CTI Report (.pdf or .csv)", type=["csv", "pdf"])
use_sample = st.sidebar.checkbox("Use Sample Data")

# ----------------------------------------
# MAIN TABS
# ----------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🧠 NLP Analysis", "🌐 Knowledge Graph", "🚨 Threat Feed"])

# ----------------------------------------
# TAB 1: OVERVIEW
# ----------------------------------------
with tab1:
    st.title("📊 Cyber Threat Intelligence Overview")
    if uploaded_file or use_sample:
        if use_sample:
            df = build_sample_feed()
        elif uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            text = extract_text_from_pdf(uploaded_file)
            df = extract_entities(text)

        st.metric("Total Indicators", len(df))
        st.metric("Average Confidence", round(df["Confidence"].mean(), 2) if "Confidence" in df else "N/A")

        st.dataframe(df.head(10), use_container_width=True)
    else:
        st.info("Upload a CTI feed (PDF or CSV) or enable sample data from the sidebar.")

# ----------------------------------------
# TAB 2: NLP ANALYSIS (NER + SENTIMENT)
# ----------------------------------------
with tab2:
    st.title("🧠 NLP Entity Recognition & Sentiment Analysis")
    if uploaded_file or use_sample:
        if use_sample:
            text = "Sample threat report: Malware using phishing to target financial systems across networks."
        elif uploaded_file.name.endswith(".pdf"):
            text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            text = " ".join(df["Indicator"].astype(str).tolist())
        else:
            text = ""

        st.write("### Extracted Text Preview")
        st.write(text[:800] + "..." if len(text) > 800 else text)

        entities_df = extract_entities(text)
        sentiment = analyze_sentiment(text)
        topic = topic_modeling(text)

        st.markdown("### 🔍 Extracted Entities")
        st.dataframe(entities_df)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Sentiment", sentiment["label"], f"{sentiment['score']*100:.1f}%")
        with col2:
            st.metric("Topic", topic["topic"], f"{topic['confidence']*100:.1f}%")
    else:
        st.info("Upload a file or enable sample data to analyze entities and sentiment.")

# ----------------------------------------
# TAB 3: KNOWLEDGE GRAPH
# ----------------------------------------
with tab3:
    st.title("🌐 Cyber Threat Knowledge Graph")
    st.markdown("Visual representation of cyber entities and their relationships.")
    fig = draw_cti_graph()
    st.pyplot(fig)
    st.success("Graph layout designed to show relationships between key threat elements.")

# ----------------------------------------
# TAB 4: THREAT FEED ANALYSIS
# ----------------------------------------
with tab4:
    st.title("🚨 Threat Feed Analysis")

    if uploaded_file or use_sample:
        df = build_sample_feed() if use_sample else pd.read_csv(uploaded_file)

        st.subheader("Filters")
        col1, col2, col3 = st.columns(3)
        with col1:
            type_filter = st.selectbox("Indicator Type", ["All"] + sorted(df["Type"].unique()))
        with col2:
            conf_min, conf_max = st.slider("Confidence Range", 0.0, 100.0, (0.0, 100.0))
        with col3:
            source_filter = st.selectbox("Source", ["All"] + sorted(df["Source"].unique()))

        filtered_df = df.copy()
        if type_filter != "All":
            filtered_df = filtered_df[filtered_df["Type"] == type_filter]
        if source_filter != "All":
            filtered_df = filtered_df[filtered_df["Source"] == source_filter]
        filtered_df = filtered_df[
            (filtered_df["Confidence"] >= conf_min) &
            (filtered_df["Confidence"] <= conf_max)
        ]

        st.dataframe(filtered_df, use_container_width=True)
        st.write(f"**Indicator Count:** {len(filtered_df)} | **Average Confidence:** {round(filtered_df['Confidence'].mean(), 2)}")
    else:
        st.info("Upload a CSV or enable sample data to explore threat feeds.")
