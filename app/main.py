import streamlit as st
import pandas as pd
import re
import networkx as nx
import matplotlib.pyplot as plt

# Optional imports with fallbacks
try:
    from transformers import pipeline
    HF = True
except Exception:
    HF = False

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    SK = True
except Exception:
    SK = False

try:
    from bertopic import BERTopic
    BER = True
except Exception:
    BER = False


# ---------------- Sidebar ----------------
st.sidebar.title("CTI Unified Dashboard")
uploaded_file = st.sidebar.file_uploader("Upload CSV dataset", type=["csv"])
task = st.sidebar.radio(
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


# ---------------- Utility: Read CSV ----------------
def read_text_from_csv(file):
    df = pd.read_csv(file)
    st.write("Data preview:", df.head())

    # Choose column
    text_col = st.selectbox("Select column to analyze", df.columns)

    if text_col:
        text_data = " ".join(df[text_col].astype(str).tolist())
    else:
        text_data = ""

    # If short text or not informative, combine all object columns
    if len(text_data.strip()) < 10:
        obj_cols = [c for c in df.columns if df[c].dtype == "object"]
        if obj_cols:
            text_data = " ".join(
                df[obj_cols].astype(str).fillna("").agg(" ".join, axis=1)
            )

    return text_data, df


def split_sentences(text):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 3]


# ---------------- NER ----------------
def extract_entities(text):
    if not text:
        return pd.DataFrame()
    if HF:
        try:
            ner = pipeline("ner", aggregation_strategy="simple")
            ents = ner(text[:4000])
            df = pd.DataFrame(ents)
            df = df.rename(
                columns={"word": "Entity", "entity_group": "Type", "score": "Score"}
            )
            return df[["Entity", "Type", "Score"]]
        except Exception:
            pass

    ips = re.findall(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", text)
    cves = re.findall(r"\bCVE-\d{4}-\d+\b", text)
    domains = re.findall(r"\b[a-zA-Z0-9.-]+\.[a-z]{2,}\b", text)
    rows = (
        [{"Entity": i, "Type": "IP"} for i in ips]
        + [{"Entity": c, "Type": "CVE"} for c in cves]
        + [{"Entity": d, "Type": "Domain"} for d in domains]
    )
    return pd.DataFrame(rows)


# ---------------- Knowledge Graph ----------------
def build_graph(entities_df):
    G = nx.Graph()
    if entities_df.empty:
        return G
    for _, r in entities_df.iterrows():
        G.add_node(r["Entity"], type=r.get("Type", "Unknown"))
    nodes = list(entities_df["Entity"])
    for i in range(len(nodes) - 1):
        G.add_edge(nodes[i], nodes[i + 1])
    return G


def plot_graph(G):
    if len(G.nodes) == 0:
        st.info("No entities to visualize.")
        return
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 5))
    nx.draw(G, pos, with_labels=True, node_color="#B3CDE0", node_size=900, font_size=8)
    st.pyplot(plt)
    plt.close()


# ---------------- Sentence Clustering ----------------
def cluster_sentences(sentences):
    if not sentences:
        return pd.DataFrame()
    if SK:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        emb = model.encode(sentences)
        db = DBSCAN(eps=1.0, min_samples=2).fit(emb)
        df = pd.DataFrame({"sentence": sentences, "cluster": db.labels_})
        return df
    else:
        return pd.DataFrame({"sentence": sentences, "cluster": 0})


# ---------------- Topic Modeling ----------------
def topic_model(sentences):
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return pd.DataFrame()

    if BER and len(sentences) > 4:
        try:
            model = BERTopic(verbose=False)
            topics, _ = model.fit_transform(sentences)
            topic_info = model.get_topic_info()
            return topic_info
        except Exception:
            pass

    vect = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    X = vect.fit_transform(sentences)
    if len(sentences) > 3:
        k = min(5, len(sentences) // 2 or 1)
        km = KMeans(n_clusters=k)
        labs = km.fit_predict(X)
        df = pd.DataFrame({"sentence": sentences, "topic": labs})
        return df
    return pd.DataFrame({"sentence": sentences, "topic": 0})


# ---------------- Sentiment ----------------
def sentiment_analysis(text):
    if not text:
        return pd.DataFrame([{"label": "NoInput", "score": 0.0}])
    if HF:
        try:
            sent = pipeline("sentiment-analysis")
            res = sent(text[:1000])
            return pd.DataFrame(res)
        except Exception:
            pass

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
    return pd.DataFrame([{"label": label, "score": round(score, 2)}])


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
    if not text:
        return ["Informational"]
    detected = [v for k, v in CTI_LABELS.items() if k in text.lower()]
    return list(set(detected)) or ["Informational"]


# ---------------- MAIN ----------------
st.title("Cyber Threat Intelligence Dashboard")

if not uploaded_file:
    st.info("Upload a CSV dataset to start.")
    st.stop()

text, df = read_text_from_csv(uploaded_file)
sentences = split_sentences(text)

if run:
    if task == "Named Entity Recognition (NER)":
        st.subheader("Named Entity Recognition")
        ents = extract_entities(text)
        if ents.empty:
            st.warning("No entities detected in this column.")
        else:
            st.dataframe(ents)
            st.download_button("Download Entities", ents.to_csv(index=False), "entities.csv")

    elif task == "Knowledge Graph":
        st.subheader("Knowledge Graph")
        ents = extract_entities(text)
        G = build_graph(ents)
        plot_graph(G)

    elif task == "Sentence Clustering":
        st.subheader("Sentence Clustering")
        dfc = cluster_sentences(sentences)
        if dfc.empty:
            st.warning("No sentences or clusters found.")
        else:
            st.dataframe(dfc)
            st.download_button("Download Clusters", dfc.to_csv(index=False), "clusters.csv")

    elif task == "Topic Modeling":
        st.subheader("Topic Modeling")
        df_topics = topic_model(sentences)
        if df_topics.empty:
            st.warning("No topics extracted (not enough textual content).")
        else:
            st.dataframe(df_topics)
            st.download_button("Download Topics", df_topics.to_csv(index=False), "topics.csv")

    elif task == "Sentiment Analysis":
        st.subheader("Sentiment Analysis")
        sentiment = sentiment_analysis(text)
        st.dataframe(sentiment)
        st.download_button("Download Sentiment", sentiment.to_csv(index=False), "sentiment.csv")

    elif task == "CTI Classification":
        st.subheader("CTI Classification")
        cats = cti_classify(text)
        st.write("Detected CTI Categories:", ", ".join(cats))
        st.download_button("Download CTI Categories", ",".join(cats), "cti_labels.txt")
