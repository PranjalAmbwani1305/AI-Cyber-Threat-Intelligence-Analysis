import streamlit as st
import pandas as pd
import re
import networkx as nx
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------- Page Config ----------------
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
    df = pd.read_csv(file)
    st.write("### 📋 Data Preview")
    st.dataframe(df.head())
    st.success(f"✅ Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
    return df


def split_sentences(text):
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sents if len(s.strip()) > 2]


def extract_entities(text):
    ips = re.findall(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", text)
    cves = re.findall(r"\bCVE-\d{4}-\d+\b", text)
    domains = re.findall(r"\b[a-zA-Z0-9.-]+\.[a-z]{2,}\b", text)
    return {"IP": list(set(ips)), "CVE": list(set(cves)), "Domain": list(set(domains))}


def suggest_best_columns(df):
    src_candidates = [c for c in df.columns if any(k in c.lower() for k in ["src", "source", "from", "ip"])]
    dst_candidates = [c for c in df.columns if any(k in c.lower() for k in ["dst", "dest", "destination", "to", "ip"])]

    src_col = src_candidates[0] if src_candidates else df.columns[0]
    dst_col = dst_candidates[0] if dst_candidates else (df.columns[1] if len(df.columns) > 1 else df.columns[0])
    return src_col, dst_col


def build_graph(df, src_col, dst_col):
    pairs = (
        df[[src_col, dst_col]]
        .dropna()
        .astype(str)
        .query(f"`{src_col}` != '' and `{dst_col}` != ''")
    )
    G = nx.from_pandas_edgelist(pairs.head(400), source=src_col, target=dst_col, create_using=nx.DiGraph())
    return G


def plot_graph(G, src_col, dst_col):
    if not G or G.number_of_nodes() == 0:
        st.warning("⚠️ No valid connections to visualize.")
        return
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G, k=0.7, seed=42)
    nx.draw_networkx_nodes(G, pos, node_color="#64B5F6", node_size=700, alpha=0.9)
    nx.draw_networkx_edges(G, pos, edge_color="#B0BEC5", arrows=True, alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_size=7)
    plt.title(f"Knowledge Graph: {src_col} → {dst_col}", fontsize=13)
    st.pyplot(plt)
    plt.close()


def cluster_sentences(sentences):
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
    found = [v for k, v in CTI_LABELS.items() if k in text.lower()]
    return list(set(found)) or ["Informational"]


# ---------------- MAIN ----------------
if not uploaded_file:
    st.info("⬆️ Upload a CSV file to start.")
    st.stop()

df = read_csv_data(uploaded_file)

# --- Named Entity Recognition ---
if task == "Named Entity Recognition (NER)":
    st.subheader("🔍 Named Entity Recognition (NER)")
    st.markdown("> Extracts entities such as **IPs, Domains, and CVEs**.")

    col = st.selectbox("Select column for entity extraction", df.columns, key="ner_col")
    if run:
        text = " ".join(df[col].astype(str).tolist())
        entities = extract_entities(text)
        for t, vals in entities.items():
            st.markdown(f"**{t}:** {', '.join(vals[:20]) if vals else 'None found'}")

# --- Knowledge Graph ---
elif task == "Knowledge Graph":
    st.subheader("🌐 Knowledge Graph")
    st.markdown("> Visualizes **connections** between two columns (e.g., Source_IP → Destination_IP).")

    suggested_src, suggested_dst = suggest_best_columns(df)
    st.info(f"💡 Suggested Columns → Source: `{suggested_src}`, Destination: `{suggested_dst}`")

    src_col = st.selectbox("Select Source Column", df.columns, key="src_col_selector")
    dst_col = st.selectbox("Select Destination Column", df.columns, key="dst_col_selector")

    if run:
        st.write(f"Building graph from **{src_col} → {dst_col}** ...")
        G = build_graph(df, src_col, dst_col)
        if G:
            st.success(f"✅ Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
            plot_graph(G, src_col, dst_col)
        else:
            st.warning("⚠️ No valid source–destination pairs found.")

# --- Sentence Clustering ---
elif task == "Sentence Clustering":
    st.subheader("🧩 Sentence Clustering")
    st.markdown("> Groups similar sentences. **Cluster -1** = unclassified/noise data.")
    col = st.selectbox("Select column for clustering", df.columns, key="cluster_col")
    if run:
        text = " ".join(df[col].astype(str).tolist())
        sentences = split_sentences(text)
        clusters = cluster_sentences(sentences)
        for label, sents in clusters.items():
            st.markdown(f"### Cluster {label}" if label != -1 else "### Cluster -1 (Noise)")
            for s in sents[:5]:
                st.write(f"- {s}")

# --- Topic Modeling ---
elif task == "Topic Modeling":
    st.subheader("🧠 Topic Modeling")
    st.markdown("> Identifies **themes and keywords** in text.")
    col = st.selectbox("Select column for topic modeling", df.columns, key="topic_col")
    if run:
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
    st.markdown("> Detects tone of text (negative for attacks, positive for fixes).")
    col = st.selectbox("Select column for sentiment analysis", df.columns, key="sentiment_col")
    if run:
        text = " ".join(df[col].astype(str).tolist())
        res = sentiment_analysis(text)
        st.markdown(f"**Sentiment:** {res['label']}  |  **Score:** {res['score']}")

# --- CTI Classification ---
elif task == "CTI Classification":
    st.subheader("🧾 CTI Classification")
    st.markdown("> Classifies text into CTI categories (Malware, Phishing, etc.).")
    col = st.selectbox("Select column for CTI classification", df.columns, key="cti_col")
    if run:
        text = " ".join(df[col].astype(str).tolist())
        cats = cti_classify(text)
        st.markdown("**Detected CTI Categories:** " + ", ".join(cats))
