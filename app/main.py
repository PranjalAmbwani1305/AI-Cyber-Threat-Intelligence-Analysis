import streamlit as st
import pandas as pd
import re
import networkx as nx
import matplotlib.pyplot as plt
from io import BytesIO

# Optional imports with fallback
try:
    from transformers import pipeline
    HF = True
except:
    HF = False

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import PCA
    SK = True
except:
    SK = False

try:
    from bertopic import BERTopic
    BER = True
except:
    BER = False

# ----- Sidebar -----
st.sidebar.title("📚 CTI Unified Dashboard")
st.sidebar.markdown("Select task & upload your file")

uploaded_file = st.sidebar.file_uploader("Upload CSV/PDF/TXT", type=["csv", "pdf", "txt"])
task = st.sidebar.radio(
    "Choose Task",
    [
        "Named Entity Recognition (NER)",
        "Knowledge Graph",
        "Sentence Clustering",
        "Topic Modeling",
        "Sentiment Analysis",
        "CTI Classification"
    ]
)
run = st.sidebar.button("🚀 Run Task")

# ----- Text loading -----
def read_text(uploaded_file):
    if uploaded_file is None:
        return ""
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        text = " ".join(df[df.columns[0]].astype(str))
    elif uploaded_file.name.endswith(".pdf"):
        from PyPDF2 import PdfReader
        reader = PdfReader(uploaded_file)
        text = "".join([p.extract_text() for p in reader.pages])
    else:
        text = uploaded_file.read().decode("utf-8", errors="ignore")
    return text

# ----- Common helpers -----
def split_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text.strip())

def extract_entities(text):
    if HF:
        try:
            ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
            ents = ner(text[:5000])
            df = pd.DataFrame(ents)
            return df[["word", "entity_group", "score"]].rename(columns={"word": "Entity", "entity_group": "Type"})
        except:
            pass
    # fallback
    ips = re.findall(r'\b\d{1,3}(?:\.\d{1,3}){3}\b', text)
    cves = re.findall(r'\bCVE-\d{4}-\d+\b', text)
    domains = re.findall(r'\b[a-zA-Z0-9.-]+\.[a-z]{2,}\b', text)
    data = [{"Entity": i, "Type": "IP"} for i in ips] + [{"Entity": c, "Type": "CVE"} for c in cves] + [{"Entity": d, "Type": "Domain"} for d in domains]
    return pd.DataFrame(data)

def build_graph(entities_df):
    G = nx.Graph()
    for _, row in entities_df.iterrows():
        G.add_node(row["Entity"], type=row["Type"])
    nodes = list(entities_df["Entity"])
    for i in range(len(nodes) - 1):
        G.add_edge(nodes[i], nodes[i + 1])
    return G

def plot_graph(G):
    pos = nx.spring_layout(G)
    plt.figure(figsize=(7, 5))
    nx.draw(G, pos, with_labels=True, node_color="#89CFF0", node_size=900, font_size=9)
    st.pyplot(plt)
    plt.close()

def cluster_sentences(sentences):
    if SK:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        emb = model.encode(sentences)
        db = DBSCAN(eps=1.0, min_samples=2).fit(emb)
        labels = db.labels_
        return pd.DataFrame({"sentence": sentences, "cluster": labels})
    else:
        return pd.DataFrame({"sentence": sentences, "cluster": 0})

def topic_model(sentences):
    if BER:
        model = BERTopic(verbose=False)
        topics, _ = model.fit_transform(sentences)
        topic_info = model.get_topic_info()
        return topic_info
    else:
        vect = TfidfVectorizer(stop_words="english")
        X = vect.fit_transform(sentences)
        kmeans = KMeans(n_clusters=min(5, len(sentences)//2 or 1))
        labs = kmeans.fit_predict(X)
        df = pd.DataFrame({"sentence": sentences, "topic": labs})
        return df

def sentiment_analysis(text):
    if HF:
        sent = pipeline("sentiment-analysis")
        res = sent(text[:1000])
        return pd.DataFrame(res)
    else:
        score = 0.5
        if "attack" in text.lower() or "malware" in text.lower():
            score = 0.1
        label = "NEGATIVE" if score < 0.5 else "POSITIVE"
        return pd.DataFrame([{"label": label, "score": score}])

def cti_classify(text):
    cats = {
        "phishing": "Phishing",
        "malware": "Malware",
        "ransom": "Ransomware",
        "cve": "Vulnerability",
        "exploit": "Exploit",
        "breach": "Breach"
    }
    found = [v for k, v in cats.items() if k in text.lower()]
    return list(set(found)) or ["Informational"]

# ----- MAIN -----
st.title("🔐 Cyber Threat Intelligence Dashboard")

if not uploaded_file:
    st.info("Please upload a file first from the sidebar.")
    st.stop()

text = read_text(uploaded_file)
sentences = split_sentences(text)

if run:
    if task == "Named Entity Recognition (NER)":
        st.header("🧠 Named Entity Recognition")
        ents = extract_entities(text)
        st.dataframe(ents)
        st.download_button("Download Entities", ents.to_csv(index=False), "entities.csv")

    elif task == "Knowledge Graph":
        st.header("🌐 Knowledge Graph")
        ents = extract_entities(text)
        G = build_graph(ents)
        st.write(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        plot_graph(G)

    elif task == "Sentence Clustering":
        st.header("🗂️ Sentence Clustering")
        df = cluster_sentences(sentences)
        st.dataframe(df)
        st.download_button("Download Clusters", df.to_csv(index=False), "clusters.csv")

    elif task == "Topic Modeling":
        st.header("🪄 Topic Modeling")
        df = topic_model(sentences)
        st.dataframe(df)
        st.download_button("Download Topics", df.to_csv(index=False), "topics.csv")

    elif task == "Sentiment Analysis":
        st.header("💬 Sentiment Analysis")
        res = sentiment_analysis(text)
        st.dataframe(res)
        st.download_button("Download Sentiment", res.to_csv(index=False), "sentiment.csv")

    elif task == "CTI Classification":
        st.header("⚔️ CTI Classification")
        cats = cti_classify(text)
        st.success(f"Detected CTI Categories: {', '.join(cats)}")
