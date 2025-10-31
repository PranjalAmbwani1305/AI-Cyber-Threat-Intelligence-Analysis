import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import re
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Streamlit UI setup ---
st.set_page_config(layout="wide", page_title="CTI Dashboard")
st.title("Cyber Threat Intelligence Dashboard")

uploaded_file = st.sidebar.file_uploader("Upload CSV dataset", type=["csv"])
task = st.sidebar.selectbox(
    "Select Analysis Task",
    [
        "Named Entity Recognition (NER)",
        "Knowledge Graph (Source_IP → Destination_IP)",
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

    # Auto-select a text column (for NLP tasks)
    def score_column(series):
        text = series.astype(str)
        avg_len = text.map(len).mean()
        letters = text.str.contains(r"[A-Za-z]").mean()
        return avg_len * letters

    scores = {c: score_column(df[c]) for c in df.columns}
    best_col = max(scores, key=scores.get)
    st.write("Auto-selected text column:", best_col)

    text_data = " ".join(df[best_col].astype(str).tolist())
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


# ---------------- Source_IP → Destination_IP Knowledge Graph ----------------
def build_source_dest_graph(df):
    """
    Build a simple Knowledge Graph using Source_IP → Destination_IP connection
    """
    G = nx.DiGraph()
    src_candidates = [c for c in df.columns if "source" in c.lower() or "src" in c.lower()]
    dst_candidates = [c for c in df.columns if "destination" in c.lower() or "dst" in c.lower()]

    if not src_candidates or not dst_candidates:
        st.error("No Source_IP or Destination_IP columns found.")
        return None

    s_col, d_col = src_candidates[0], dst_candidates[0]
    pairs = df[[s_col, d_col]].dropna().head(500)

    for _, row in pairs.iterrows():
        src = str(row[s_col])
        dst = str(row[d_col])
        if src and dst:
            G.add_edge(src, dst)

    return G


def plot_source_dest_graph(G):
    if G is None or G.number_of_nodes() == 0:
        st.warning("No Source → Destination connections found.")
        return

    max_nodes = 100  # keep graph readable
    if G.number_of_nodes() > max_nodes:
        degree_dict = dict(G.degree())
        top_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)[:max_nodes]
        G = G.subgraph(top_nodes).copy()

    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G, k=0.7, seed=42)
    nx.draw_networkx_nodes(G, pos, node_color="#90CAF9", node_size=700, alpha=0.9)
    nx.draw_networkx_edges(G, pos, edge_color="#9E9E9E", arrows=True, alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_size=7)
    plt.title("Knowledge Graph: Source_IP → Destination_IP", fontsize=13)
    st.pyplot(plt)
    plt.close()


# ---------------- NLP tasks (short & clean) ----------------
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

    elif task == "Knowledge Graph (Source_IP → Destination_IP)":
        st.subheader("Knowledge Graph: Source_IP → Destination_IP")
        G = build_source_dest_graph(df)
        if G:
            st.write(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
            plot_source_dest_graph(G)

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
