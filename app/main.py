import streamlit as st
import pandas as pd
import re
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sentence_transformers import SentenceTransformer

# --- Layout ---
st.set_page_config(layout="wide", page_title="CTI Dashboard")
st.title("Cyber Threat Intelligence Dashboard")

uploaded_file = st.sidebar.file_uploader("Upload CSV dataset", type=["csv"])
task = st.sidebar.selectbox(
    "Select Analysis Task",
    [
        "Named Entity Recognition (NER)",
        "Knowledge Graph",
        "Sentence Clustering",
        "Topic Modeling",
        "Sentiment Analysis",
        "CTI Classification",
    ],
)
run = st.sidebar.button("Run Analysis")

# ---------------- Load CSV ----------------
def read_text_from_csv(file):
    df = pd.read_csv(file)
    st.write("Data preview:")
    st.dataframe(df.head())

    def score_column(series):
        text = series.astype(str)
        avg_len = text.map(len).mean()
        letters = text.str.contains(r"[A-Za-z]").mean()
        return avg_len * letters

    scores = {c: score_column(df[c]) for c in df.columns}
    best_col = max(scores, key=scores.get)
    st.write("Auto-selected best column:", best_col)

    text_col = st.selectbox("Select column to analyze", [best_col] + list(df.columns))
    text_data = " ".join(df[text_col].astype(str).tolist())
    st.write("Constructed text length:", len(text_data), "characters.")
    return text_data, df


def split_sentences(text):
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sents if len(s.strip()) > 2]


# ---------------- NER ----------------
def extract_entities(text):
    ips = re.findall(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", text)
    cves = re.findall(r"\bCVE-\d{4}-\d+\b", text)
    domains = re.findall(r"\b[a-zA-Z0-9.-]+\.[a-z]{2,}\b", text)
    entities = {
        "IP": list(set(ips)),
        "CVE": list(set(cves)),
        "Domain": list(set(domains)),
    }
    return entities


# ---------------- Knowledge Graph ----------------
def build_graph_from_df(df):
    G = nx.DiGraph()
    src_cols = [c for c in df.columns if "src" in c.lower()]
    dst_cols = [c for c in df.columns if "dst" in c.lower()]
    threat_cols = [c for c in df.columns if "threat" in c.lower()]
    indicator_cols = [c for c in df.columns if "indicator" in c.lower() or "ip" in c.lower()]

    if src_cols and dst_cols:
        for _, r in df[[src_cols[0], dst_cols[0]]].dropna().head(200).iterrows():
            G.add_edge(str(r[src_cols[0]]), str(r[dst_cols[0]]), relation="connects_to")
    elif threat_cols and indicator_cols:
        for _, r in df[[threat_cols[0], indicator_cols[0]]].dropna().head(200).iterrows():
            G.add_edge(str(r[threat_cols[0]]), str(r[indicator_cols[0]]), relation="related_to")

    if G.number_of_edges() == 0:
        for _, row in df.head(100).iterrows():
            vals = [str(v) for v in row.values if len(str(v)) > 3]
            for i in range(len(vals)):
                for j in range(i + 1, len(vals)):
                    G.add_edge(vals[i], vals[j])
    return G


def plot_graph(G):
    if G.number_of_nodes() == 0:
        st.warning("No relationships found.")
        return
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G, k=0.6, seed=42)
    nx.draw(
        G, pos, with_labels=True, node_color="#90CAF9", node_size=800, font_size=8, edge_color="#9E9E9E"
    )
    plt.title("Knowledge Graph")
    st.pyplot(plt)
    plt.close()


# ---------------- Sentence Clustering ----------------
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


# ---------------- Topic Modeling ----------------
def topic_model(sentences):
    vect = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    X = vect.fit_transform(sentences)
    k = min(5, len(sentences) // 2 or 1)
    km = KMeans(n_clusters=k)
    labs = km.fit_predict(X)
    terms = vect.get_feature_names_out()
    df = pd.DataFrame({"sentence": sentences, "topic": labs})
    topics = {}
    for t in sorted(df["topic"].unique()):
        topic_sents = df[df["topic"] == t]["sentence"].tolist()[:3]
        topics[t] = {
            "keywords": ", ".join(terms[:8]),
            "examples": topic_sents,
        }
    return topics


# ---------------- Sentiment ----------------
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


# ---------------- CTI Classification ----------------
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
    st.info("Upload a CSV to start.")
    st.stop()

text, df = read_text_from_csv(uploaded_file)
sentences = split_sentences(text)

if run:
    if task == "Named Entity Recognition (NER)":
        st.subheader("Named Entity Recognition")
        entities = extract_entities(text)
        for t, vals in entities.items():
            if vals:
                st.markdown(f"**{t}:** {', '.join(vals[:20])}")
            else:
                st.markdown(f"**{t}:** None found")

    elif task == "Knowledge Graph":
        st.subheader("Knowledge Graph")
        G = build_graph_from_df(df)
        st.write(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        plot_graph(G)

    elif task == "Sentence Clustering":
        st.subheader("Sentence Clustering")
        clusters = cluster_sentences(sentences)
        if not clusters:
            st.warning("No clusters found.")
        else:
            for label, sents in clusters.items():
                st.markdown(f"### Cluster {label}")
                for s in sents[:3]:
                    st.write(f"- {s}")

    elif task == "Topic Modeling":
        st.subheader("Topic Modeling")
        topics = topic_model(sentences)
        if not topics:
            st.warning("No topics found.")
        else:
            for t, info in topics.items():
                st.markdown(f"### Topic {t}")
                st.markdown(f"**Top Keywords:** {info['keywords']}")
                for s in info["examples"]:
                    st.write(f"- {s}")

    elif task == "Sentiment Analysis":
        st.subheader("Sentiment Analysis")
        res = sentiment_analysis(text)
        st.markdown(f"**Sentiment:** {res['label']}  |  **Score:** {res['score']}")

    elif task == "CTI Classification":
        st.subheader("CTI Classification")
        cats = cti_classify(text)
        st.markdown("**Detected CTI Categories:** " + ", ".join(cats))
