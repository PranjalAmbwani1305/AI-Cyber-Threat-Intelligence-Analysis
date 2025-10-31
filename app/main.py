import streamlit as st
import matplotlib.pyplot as plt
import igraph as ig
from nlp_logic import extract_entities, analyze_sentiment, topic_modeling
from PyPDF2 import PdfReader

st.set_page_config(page_title="🧠 Cyber Threat Intelligence with NLP", layout="wide")

st.title("🧠 AI-Driven Cyber Threat Intelligence Analyzer")
st.markdown("Upload a **CTI report (PDF, CSV, or TXT)** to extract entities, analyze sentiment, and visualize a knowledge graph.")

def extract_text(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    else:
        return uploaded_file.read().decode("utf-8")

uploaded_file = st.file_uploader("Upload Report", type=["pdf", "txt", "csv"])
if uploaded_file:
    text = extract_text(uploaded_file)
    st.subheader("📄 Extracted Text Sample")
    st.write(text[:1000] + "...")

    with st.spinner("Analyzing NLP Layers..."):
        entities_df = extract_entities(text)
        sentiment = analyze_sentiment(text)
        topic = topic_modeling(text)

    st.markdown("### 🧩 Named Entities Extracted (NER)")
    st.dataframe(entities_df)

    st.markdown("### 💬 Sentiment & Topic")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Sentiment", sentiment["label"])
    with col2:
        st.metric("Main Topic", f"{topic['topic']} ({topic['confidence']*100:.1f}%)")

    st.markdown("### 🌐 Knowledge Graph (Prototype)")
    G = ig.Graph(directed=True)
    for _, row in entities_df.iterrows():
        G.add_vertex(row["Entity"])
    edges = [(entities_df.iloc[i]["Entity"], entities_df.iloc[i + 1]["Entity"]) for i in range(len(entities_df) - 1)]
    G.add_edges(edges)

    layout = G.layout("circle")
    fig, ax = plt.subplots(figsize=(10, 8))
    ig.plot(G, target=ax, layout=layout, vertex_label=G.vs["name"], vertex_color="lightblue", vertex_size=20)
    st.pyplot(fig)
