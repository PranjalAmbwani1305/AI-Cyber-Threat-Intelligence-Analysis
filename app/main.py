import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import json
from io import BytesIO
import traceback

# Import your local NLP modules if present
try:
    from NER_Transformer import extract_cti_entities
except ImportError:
    from transformers import pipeline

    def extract_cti_entities(text):
        nlp = pipeline("ner", grouped_entities=True)
        ents = nlp(text)
        return [{"word": e["word"], "entity_group": e["entity_group"], "score": e["score"]} for e in ents]

# Optional modules (safe imports)
try:
    from Text_classification_Sentiment import analyze_sentiment
except ImportError:
    def analyze_sentiment(text):
        from transformers import pipeline
        sent = pipeline("sentiment-analysis")
        res = sent(text[:512])
        return res[0]["label"], float(res[0]["score"])

try:
    from Topic_Modelling_Sentence_Analysis_ import topic_modeling
except ImportError:
    def topic_modeling(text):
        # placeholder: simple topic extraction using TF-IDF keywords
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import NMF
        import numpy as np
        docs = text.split(".")
        vec = TfidfVectorizer(max_features=1000, stop_words="english")
        X = vec.fit_transform(docs)
        nmf = NMF(n_components=3, random_state=42)
        W = nmf.fit_transform(X)
        H = nmf.components_
        top_words = []
        for topic_idx, topic in enumerate(H):
            words = [vec.get_feature_names_out()[i] for i in topic.argsort()[:-6:-1]]
            top_words.append(", ".join(words))
        return top_words

try:
    from Clustering_Prototype import cluster_texts
except ImportError:
    def cluster_texts(text):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans
        sents = text.split(".")
        vec = TfidfVectorizer(max_features=1000, stop_words="english")
        X = vec.fit_transform(sents)
        kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
        clusters = {i: [] for i in range(3)}
        for sent, label in zip(sents, kmeans.labels_):
            clusters[label].append(sent.strip())
        return clusters

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    text = ""
    pdf = fitz.open(stream=BytesIO(file_bytes), filetype="pdf")
    for page in pdf:
        text += page.get_text(sort=True)
    return text

def mitre_mapping(entity_group):
    mapping = {
        "MALWARE": "Execution",
        "VULNERABILITY": "Initial Access",
        "IP": "Command and Control",
        "DOMAIN": "Reconnaissance",
        "URL": "Resource Development",
        "TOOL": "Defense Evasion",
        "THREAT_ACTOR": "Impact",
        "ORGANIZATION": "Targeting",
    }
    return mapping.get(entity_group.upper(), "Unmapped")

# -------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------
st.set_page_config(page_title="AI CTI Analysis Dashboard", layout="wide", page_icon="🧠")

st.title("🧠 AI-Powered Cyber Threat Intelligence Dashboard")
st.markdown("Upload a CTI report to analyze NER, sentiment, topics, and clusters with MITRE mapping.")

uploaded_file = st.file_uploader("📄 Upload CTI Report (PDF, CSV, XLSX, or TXT):", type=["pdf", "csv", "xlsx", "txt"])

if uploaded_file:
    st.info(f"Processing `{uploaded_file.name}` ...")

    # Extract text
    if uploaded_file.name.endswith(".pdf"):
        text = extract_text_from_pdf(uploaded_file.read())
    elif uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        text = " ".join(df.astype(str).values.flatten())
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
        text = " ".join(df.astype(str).values.flatten())
    elif uploaded_file.name.endswith(".txt"):
        text = uploaded_file.read().decode("utf-8")
    else:
        st.error("Unsupported file format.")
        st.stop()

    st.success("✅ Text extracted successfully.")
    st.download_button("⬇️ Download Extracted Text", text, file_name="report_text.txt")

    st.divider()

    # NLP Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 NER Entity Extraction",
        "💬 Sentiment Analysis",
        "🧩 Topic Modeling",
        "📊 Clustering",
        "🕸 Knowledge Graph"
    ])

    # ---------------- NER ----------------
    with tab1:
        st.subheader("Named Entity Recognition (NER)")
        with st.spinner("Extracting CTI entities..."):
            entities = extract_cti_entities(text)
        if entities:
            df_ents = pd.DataFrame(entities)
            df_ents["MITRE_Tactic"] = df_ents["entity_group"].apply(mitre_mapping)
            st.dataframe(df_ents, use_container_width=True)
            st.bar_chart(df_ents["entity_group"].value_counts())
            st.download_button("Download NER CSV", df_ents.to_csv(index=False), "entities.csv")
        else:
            st.warning("No entities found.")

    # ---------------- Sentiment ----------------
    with tab2:
        st.subheader("Sentiment Analysis")
        try:
            label, score = analyze_sentiment(text)
            st.metric("Overall Sentiment", label, f"{score:.2f}")
        except Exception as e:
            st.error("Error during sentiment analysis.")
            st.text(traceback.format_exc())

    # ---------------- Topic Modeling ----------------
    with tab3:
        st.subheader("Topic Modeling")
        try:
            topics = topic_modeling(text)
            st.write("**Identified Topics:**")
            for i, topic in enumerate(topics):
                st.write(f"**Topic {i+1}:** {topic}")
        except Exception as e:
            st.error("Topic modeling failed.")
            st.text(traceback.format_exc())

    # ---------------- Clustering ----------------
    with tab4:
        st.subheader("Text Clustering")
        try:
            clusters = cluster_texts(text)
            for label, cluster_texts_ in clusters.items():
                with st.expander(f"Cluster {label+1} ({len(cluster_texts_)} items)"):
                    for c in cluster_texts_[:10]:
                        st.write("- " + c)
        except Exception as e:
            st.error("Clustering failed.")
            st.text(traceback.format_exc())

    # ---------------- Knowledge Graph ----------------
    with tab5:
        st.subheader("Knowledge Graph (Prototype)")
        st.markdown(
            "This will visualize relationships between extracted entities (malware ↔ IP ↔ actor)."
        )
        try:
            import networkx as nx
            import matplotlib.pyplot as plt

            if entities:
                G = nx.Graph()
                for e in entities:
                    G.add_node(e["word"], label=e["entity_group"])
                for i in range(len(entities) - 1):
                    G.add_edge(entities[i]["word"], entities[i + 1]["word"])
                fig, ax = plt.subplots(figsize=(10, 6))
                nx.draw_networkx(G, ax=ax, with_labels=True, font_size=8)
                st.pyplot(fig)
            else:
                st.info("Run NER first to visualize graph.")
        except Exception as e:
            st.error("Knowledge graph generation failed.")
            st.text(traceback.format_exc())

else:
    st.info("Upload a CTI report to start analysis.")

st.divider()
st.caption("🧩 AI Cyber Threat Intelligence Suite — NER | Sentiment | Topics | Graphs | 2025")
