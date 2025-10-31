# cti_dashboard_final_layout.py
"""
Cyber Threat Intelligence Dashboard (Final Layout Version)
- Adds explanations for each module
- Fully working Knowledge Graph with selectable columns
- Clean clustering output with readable sentences
"""

import streamlit as st
import pandas as pd
import re
import networkx as nx
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------- Streamlit Layout ----------------
st.set_page_config(layout="wide", page_title="CTI Dashboard")
st.title("Cyber Threat Intelligence Dashboard")

uploaded_file = st.sidebar.file_uploader("📂 Upload CSV dataset", type=["csv"])
task = st.sidebar.selectbox(
    "🧠 Select Analysis Task",
    [
        "Named Entity Recognition (NER)",
        "Knowledge Graph",
        "Sentence Clustering",
        "Topic Modeling",
        "Sentiment Analysis",
        "CTI Classification",
    ],
)
run = st.sidebar.button("▶️ Run Analysis")

# ---------------- Helper Functions ----------------
def read_csv_data(file):
    """Read uploaded CSV file."""
    df = pd.read_csv(file)
    st.write("### 📋 Data preview:")
    st.dataframe(df.head())
    st.success(f"✅ Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
    return df


def split_sentences(text):
    """Split long text into sentences."""
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sents if len(s.strip()) > 2]


# ---------------- NER ----------------
def extract_entities(text):
    """Extract IPs, CVEs, and domains."""
    ips = re.findall(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", text)
    cves = re.findall(r"\bCVE-\d{4}-\d+\b", text)
    domains = re.findall(r"\b[a-zA-Z0-9.-]+\.[a-z]{2,}\b", text)
    return {"IP": list(set(ips)), "CVE": list(set(cves)), "Domain": list(set(domains))}


# ---------------- Knowledge Graph ----------------
def build_graph(df, src_col, dst_col):
    """Build Source → Destination graph."""
    df_pairs = df[[src_col, dst_col]].dropna()
    df_pairs = df_pairs[df_pairs[src_col].astype(str).str.strip() != ""]
    df_pairs = df_pairs[df_pairs[dst_col].astype(str).str.strip() != ""]
    G = nx.DiGraph()
    for _, row in df_pairs.head(300).iterrows():
        G.add_edge(str(row[src_col]), str(row[dst_col]))
    return G


def plot_graph(G, src_col, dst_col):
    """Draw Knowledge Graph."""
    if not G or G.number_of_nodes() == 0:
        st.warning("No data to visualize.")
        return
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G, k=0.7, seed=42)
    nx.draw_networkx_nodes(G, pos, node_color="#90CAF9", node_size=700, alpha=0.9)
    nx.draw_networkx_edges(G, pos, edge_color="#B0BEC5", arrows=True, alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_size=7)
    plt.title(f"Knowledge Graph: {src_col} → {dst_col}", fontsize=13)
    st.pyplot(plt)
    plt.close()


# ---------------- NLP Helper ----------------
def cluster_sentences(sentences):
    """Cluster sentences using embeddings."""
    if not sentences:
        return {}
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(sentences)
    db = DBSCAN(eps=1.2, min_samples=2).fit(emb)
    clusters = {}
    for sent, label in zip(sentences, db.labels_):
        clusters.setdefault(label, []).append(sent)
    return clusters


def topic_model(sentences):
    """Find major topics in text."""
    vect = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    X = vect.fit_transform(sentences)
    k = min(5, len(sentences) // 2 or 1)
    km = KMeans(n_clusters=k, n_init=10)
    labs = km.fit_predict(X)
    terms = vect.get_feature_names_out()
    df = pd.DataFrame({"sentence": sentences, "topic": labs})
    topics = {}
    for t in sorted(df["topic"].unique()):
        topic_sents = df[df["topic"] == t]["sentence"].tolist()[:3]
        topics[t] = {"keywords": ", ".join(terms[:8]), "examples": topic_sents}
    return topics


def sentiment_analysis(text):
    """Simple keyword-based sentiment analysis."""
    neg_kw = ["attack", "breach", "malware", "ransom"]
    pos_kw = ["patched", "resolved", "update"]
    score = 0.5
    for k in neg_kw:
        if k in text.lower():
            score -= 0.25
    for k in pos_kw:
        if k in text.lower():
            score += 0.25
    label = "NEGATIVE" if score < 0.5 else "POSITIVE"
    return {"label": label, "score": round(score, 2)}


CTI_LABELS = {
    "phishing": "Phishing",
    "malware": "Malware",
    "ransom": "Ransomware",
    "cve": "Vulnerability",
    "exploit": "Exploit",
    "breach": "Breach",
    "attack": "Attack",
    "botnet": "Botnet",
    "ddos": "DDoS",
}


def cti_classify(text):
    """Classify text into CTI categories."""
    found = [v for k, v in CTI_LABELS.items() if k in text.lower()]
    return list(set(found)) or ["Informational"]


# ---------------- MAIN ----------------
if not uploaded_file:
    st.info("⬆️ Upload a CSV file to start.")
    st.stop()

df = read_csv_data(uploaded_file)

if run:
    # --- NER ---
    if task == "Named Entity Recognition (NER)":
        st.subheader("🔍 Named Entity Recognition (NER)")
        st.markdown("> Extracts important entities like **IPs, Domains, and CVEs** from the selected text column.")
        col = st.selectbox("Select column for entity extraction", df.columns)
        text = " ".join(df[col].astype(str).tolist())
        entities = extract_entities(text)
        for t, vals in entities.items():
            st.markdown(f"**{t}:** {', '.join(vals[:20]) if vals else 'None found'}")

    # --- Knowledge Graph ---
    elif task == "Knowledge Graph":
        st.subheader("🌐 Knowledge Graph")
        st.markdown("> Visualizes connections between two columns, such as **Source_IP → Destination_IP**.")
        src_col = st.selectbox("Select Source Column", df.columns, index=0)
        dst_col = st.selectbox("Select Destination Column", df.columns, index=min(1, len(df.columns)-1))
        if src_col and dst_col:
            G = build_graph(df, src_col, dst_col)
            if G:
                st.success(f"✅ Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
                plot_graph(G, src_col, dst_col)
            else:
                st.warning("⚠️ No valid connections found. Try different columns.")

    # --- Sentence Clustering ---
    elif task == "Sentence Clustering":
        st.subheader("🧩 Sentence Clustering")
        st.markdown("""
> Groups similar sentences together based on meaning.  
> Each cluster shows related sentences —  
> **Cluster -1** means **noise or unclassified data** (sentences that didn't fit well).
""")
        col = st.selectbox("Select column for clustering", df.columns)
        text = " ".join(df[col].astype(str).tolist())
        sentences = split_sentences(text)
        clusters = cluster_sentences(sentences)
        if not clusters:
            st.warning("No clusters found.")
        else:
            for label, sents in clusters.items():
                if label == -1:
                    st.markdown("### Cluster -1 (Noise / Outliers)")
                else:
                    st.markdown(f"### Cluster {label}")
                for s in sents[:5]:
                    st.write(f"- {s}")

    # --- Topic Modeling ---
    elif task == "Topic Modeling":
        st.subheader("🧠 Topic Modeling")
        st.markdown("> Identifies key **themes and keywords** across the selected text column.")
        col = st.selectbox("Select column for topic modeling", df.columns)
        text = " ".join(df[col].astype(str).tolist())
        sentences = split_sentences(text)
        topics = topic_model(sentences)
        for t, info in topics.items():
            st.markdown(f"### Topic {t}")
            st.markdown(f"**Top Keywords:** {info['keywords']}")
            for s in info["examples"]:
                st.write(f"- {s}")

    # --- Sentiment Analysis ---
    elif task == "Sentiment Analysis":
        st.subheader("💬 Sentiment Analysis")
        st.markdown("> Analyzes the overall **tone** of the selected column (e.g., negative for attacks, positive for resolutions).")
        col = st.selectbox("Select column for sentiment analysis", df.columns)
        text = " ".join(df[col].astype(str).tolist())
        res = sentiment_analysis(text)
        st.markdown(f"**Sentiment:** {res['label']}  |  **Score:** {res['score']}")

    # --- CTI Classification ---
    elif task == "CTI Classification":
        st.subheader("🧾 CTI Classification")
        st.markdown("> Classifies text into common **cyber threat categories** such as Malware, Phishing, Ransomware, etc.")
        col = st.selectbox("Select column for CTI classification", df.columns)
        text = " ".join(df[col].astype(str).tolist())
        cats = cti_classify(text)
        st.markdown("**Detected CTI Categories:** " + ", ".join(cats))
