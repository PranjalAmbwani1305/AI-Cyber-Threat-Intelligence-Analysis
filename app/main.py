import streamlit as st
import pandas as pd
import re
import networkx as nx
import matplotlib.pyplot as plt
from io import StringIO

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

# ---------------- UI layout ----------------
st.set_page_config(layout="wide", page_title="CTI Dashboard")
st.title("Cyber Threat Intelligence Dashboard")

st.sidebar.header("Upload & Options")
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
st.sidebar.markdown("---")
st.sidebar.write("Tips: choose a column with messages, or let the app auto-detect best column.")


# ---------------- Helpers ----------------
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
CVE_RE = re.compile(r"\bCVE[-:]?\d{4}-\d{4,7}\b", flags=re.I)
DOMAIN_RE = re.compile(r"\b(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,6}\b")

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
    "credential": "Credential_Theft",
}

def read_csv(file):
    try:
        df = pd.read_csv(file, low_memory=False)
        return df
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return None

def score_column_for_text(series: pd.Series):
    """
    Heuristic score indicating how useful a column is for textual analysis.
    - Higher for object dtype, long avg length, contains letters, contains domain/IP/CVE.
    """
    s = series.fillna("").astype(str)
    n = len(s)
    if n == 0:
        return 0.0
    avg_len = s.map(len).mean()
    pct_numeric = s.str.match(r'^[\d\.\-]+$').mean()
    pct_has_letter = s.str.contains(r'[A-Za-z]').mean()
    pct_ip = s.str.contains(IP_RE).mean()
    pct_domain = s.str.contains(DOMAIN_RE).mean()
    pct_cve = s.str.contains(CVE_RE).mean()
    # weighted combination
    score = (
        0.4 * (avg_len / (avg_len + 50))  # normalize by 50
        + 0.3 * pct_has_letter
        + 0.15 * pct_ip
        + 0.1 * pct_domain
        + 0.05 * pct_cve
    )
    # penalize columns that are almost purely numeric
    if pct_numeric > 0.9 and pct_has_letter < 0.05:
        score *= 0.1
    return float(score)

def auto_select_column(df: pd.DataFrame):
    """
    Choose the best column automatically using heuristics. Also return a ranked list.
    """
    scores = {}
    for c in df.columns:
        try:
            scores[c] = score_column_for_text(df[c])
        except Exception:
            scores[c] = 0.0
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best = ranked[0][0] if ranked and ranked[0][1] > 0.01 else None
    return best, ranked

def form_text_from_df(df, primary_col=None):
    """
    Build a text blob to analyze:
    - If primary_col provided and non-empty -> use it
    - Else: combine object columns, and important structured columns (IPs/domains/cves/username)
    """
    if primary_col and primary_col in df.columns:
        text = " ".join(df[primary_col].astype(str).tolist())
        if len(text.strip()) >= 10:
            return text

    # combine helpful columns
    cols = list(df.columns)
    obj_cols = [c for c in cols if df[c].dtype == "object"]
    chosen = obj_cols if obj_cols else cols[:3]
    # add likely indicator columns
    for candidate in ["Source_IP","Destination_IP","Src_IP","Dst_IP","source","destination","username","user","indicator","Event_Description","message","log","alert"]:
        if candidate in df.columns and candidate not in chosen:
            chosen.append(candidate)
    # combine rows (row-wise concatenation) into one string
    text_rows = df[chosen].fillna("").astype(str).agg(" ".join, axis=1)
    text = " ".join(text_rows.tolist())
    # if still short, include all columns joined
    if len(text.strip()) < 10:
        text = " ".join(df.astype(str).agg(" ".join, axis=1).tolist())
    return text

def extract_entities_from_text(text):
    """
    Extract IPs, CVEs, domains, and usernames (simple heuristics).
    Also tries HF NER if available for PERSON/ORG (not required).
    """
    entities = []
    if not text or len(text.strip())==0:
        return pd.DataFrame(columns=["Entity","Type"])
    # IPs
    ips = set(IP_RE.findall(text))
    for ip in ips:
        entities.append({"Entity": ip, "Type": "IP"})
    # CVEs
    cves = set([m.group(0) for m in CVE_RE.finditer(text)])
    for c in cves:
        entities.append({"Entity": c, "Type": "CVE"})
    # Domains
    domains = set(DOMAIN_RE.findall(text))
    for d in domains:
        # filter out numeric-only matches
        if not re.match(r'^\d+\.\d+\.\d+\.\d+$', d):
            entities.append({"Entity": d, "Type": "Domain"})
    # usernames (simple common field pattern)
    usernames = set(re.findall(r'\b[A-Za-z0-9_.-]{3,30}\b', text))
    # filter trivials (IPs/domains/numbers)
    for u in list(usernames)[:200]:
        if re.match(r'^\d', u):  # starts with digit -> skip
            continue
        if u in ips or u in domains or re.match(r'^\d+$', u):
            continue
        # heuristic: if it contains underscore or looks like username, include
        if "_" in u or re.search(r'[A-Za-z]', u):
            entities.append({"Entity": u, "Type": "Username"})
    # Optional HF NER for richer entities (if available) -- run on first 4000 chars
    if HF:
        try:
            ner = pipeline("ner", aggregation_strategy="simple")
            res = ner(text[:4000])
            for r in res:
                word = r.get("word") or r.get("entity")
                ent_type = r.get("entity_group") or r.get("entity")
                entities.append({"Entity": word, "Type": ent_type})
        except Exception:
            pass
    # dedupe preserving order
    seen = set()
    out = []
    for e in entities:
        key = (e["Entity"], e["Type"])
        if key not in seen:
            seen.add(key)
            out.append(e)
    return pd.DataFrame(out)

def build_graph_from_df(df):
    """
    Build graph using common relationships in log data:
    - Source_IP -> Destination_IP edges if columns present
    - Username -> Source_IP edges if present
    - Also add simple co-occurrence edges from entity extraction per row
    """
    G = nx.DiGraph()
    # row-wise edges using direct columns
    src_cols = [c for c in df.columns if c.lower() in ("source_ip","src_ip","source","src")]
    dst_cols = [c for c in df.columns if c.lower() in ("destination_ip","dst_ip","destination","dst")]
    user_cols = [c for c in df.columns if "user" in c.lower()]
    # Add edges for explicit pairs
    if src_cols and dst_cols:
        s_col = src_cols[0]
        d_col = dst_cols[0]
        for _, r in df[[s_col,d_col]].dropna().iterrows():
            s = str(r[s_col])
            d = str(r[d_col])
            if s and d:
                G.add_node(s, type="IP")
                G.add_node(d, type="IP")
                G.add_edge(s, d, type="connection")
    # username -> src ip
    if user_cols and src_cols:
        u_col = user_cols[0]
        s_col = src_cols[0]
        for _, r in df[[u_col,s_col]].dropna().iterrows():
            u = str(r[u_col])
            s = str(r[s_col])
            if u and s:
                G.add_node(u, type="Username")
                G.add_node(s, type="IP")
                G.add_edge(u, s, type="used_from")
    # fallback: extract entities per row and connect co-occurring entities
    for _, row in df.fillna("").iterrows():
        row_text = " ".join([str(v) for v in row.values])
        ents = extract_entities_from_text(row_text)
        names = ents["Entity"].tolist()
        for i in range(len(names)):
            G.add_node(names[i], type=ents.iloc[i]["Type"] if not ents.empty else "Entity")
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                if names[i] != names[j]:
                    G.add_edge(names[i], names[j], type="cooccurs")
    return G

def plot_graph(G, figsize=(10,6)):
    if G is None or G.number_of_nodes() == 0:
        st.info("No graph results to display.")
        return
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(G, seed=42, k=0.8)
    node_types = nx.get_node_attributes(G, "type")
    # color map by type
    cmap = {"IP":"#8dd3c7","CVE":"#ffffb3","Domain":"#bebada","Username":"#fb8072"}
    colors = [cmap.get(node_types.get(n,""), "#ccebc5") for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=500)
    nx.draw_networkx_labels(G, pos, font_size=8)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='->', arrowsize=8)
    plt.axis("off")
    st.pyplot(plt)
    plt.close()

# ---------------- NLP / clustering / topic ----------------
def cluster_sentences(sentences):
    sentences = [s for s in sentences if s.strip()]
    if not sentences:
        return pd.DataFrame()
    if SK:
        try:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            emb = model.encode(sentences)
            db = DBSCAN(eps=1.0, min_samples=2).fit(emb)
            return pd.DataFrame({"sentence": sentences, "cluster": db.labels_})
        except Exception:
            pass
    # fallback simple KMeans on TF-IDF
    vect = TfidfVectorizer(stop_words="english", max_features=500)
    X = vect.fit_transform(sentences)
    k = min(5, max(1, len(sentences)//3))
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X)
    return pd.DataFrame({"sentence": sentences, "cluster": labels})

def topic_model_safe(sentences):
    sentences = [s for s in sentences if len(s.strip())>0]
    if not sentences:
        return pd.DataFrame()
    if BER and len(sentences) >= 5:
        try:
            model = BERTopic(verbose=False)
            topics, _ = model.fit_transform(sentences)
            info = model.get_topic_info()
            return info
        except Exception:
            pass
    # fallback TF-IDF + top terms per cluster
    vect = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=1000)
    X = vect.fit_transform(sentences)
    k = min(6, max(1, len(sentences)//3))
    if len(sentences) > 1:
        try:
            km = KMeans(n_clusters=k, random_state=42)
            labs = km.fit_predict(X)
        except Exception:
            labs = [0]*len(sentences)
    else:
        labs = [0]*len(sentences)
    df = pd.DataFrame({"sentence": sentences, "topic": labs})
    # pick top tf-idf terms per topic
    rows = []
    terms = vect.get_feature_names_out()
    for t in sorted(df["topic"].unique()):
        members = df[df["topic"]==t]["sentence"].tolist()
        if not members:
            continue
        vect2 = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=200)
        X2 = vect2.fit_transform(members)
        top = vect2.get_feature_names_out()[:8].tolist() if X2.shape[0]>0 else []
        rows.append({"topic": int(t), "top_terms": ", ".join(top), "examples": " || ".join(members[:3])})
    return pd.DataFrame(rows)

def sentiment_analysis_simple(text):
    if not text or len(text.strip())==0:
        return pd.DataFrame([{"label":"NoInput","score":0.0}])
    if HF:
        try:
            s_pipe = pipeline("sentiment-analysis")
            res = s_pipe(text[:1000])
            return pd.DataFrame(res)
        except Exception:
            pass
    # naive keyword-based polarity
    neg = sum(1 for k in ["attack","breach","malware","ransom","exploit"] if k in text.lower())
    pos = sum(1 for k in ["mitigat","patched","resolved","update","blocked"] if k in text.lower())
    score = max(0, min(1, 0.5 + 0.25*(pos-neg)))
    label = "POSITIVE" if score>=0.5 else "NEGATIVE"
    return pd.DataFrame([{"label":label,"score":round(score,3)}])

def cti_keyword_tags(text):
    if not text or len(text.strip())==0:
        return ["Informational"]
    found = set()
    t = text.lower()
    for k,v in CTI_LABELS.items():
        if k in t:
            found.add(v)
    # also detect CVEs
    if CVE_RE.search(text):
        found.add("Vulnerability")
    if not found:
        return ["Informational"]
    return list(sorted(found))

# ---------------- MAIN logic ----------------
if uploaded_file is None:
    st.info("Please upload a CSV file to begin.")
    st.stop()

df = read_csv(uploaded_file)
if df is None:
    st.stop()

st.subheader("Data preview")
st.dataframe(df.head(10))

# Auto-detect best column and let user override
best_col, ranked = auto_select_column(df)
st.write("Top candidate columns (score descending):")
rank_df = pd.DataFrame(ranked, columns=["column","score"])
st.dataframe(rank_df.head(10))
st.write("Auto-selected best column:", best_col)

# Allow manual override
user_col = st.selectbox("Select column to analyze (override auto-selection)", options=[None] + list(df.columns), index=0)
primary_col = user_col if user_col else best_col

text_blob = form_text_from_df(df, primary_col=primary_col)
if len(text_blob.strip()) < 10:
    st.warning("Constructed text is short. Results may be limited. You may choose a different column above.")
else:
    st.write(f"Constructed text length: {len(text_blob)} characters.")

sentences = re.split(r'(?<=[.!?])\s+', text_blob)
sentences = [s.strip() for s in sentences if len(s.strip())>2]

if run:
    # NER
    if task == "Named Entity Recognition (NER)":
        st.header("Named Entity Recognition (Entities found)")
        ents_df = extract_entities_from_text(text_blob)
        if ents_df.empty:
            st.warning("No entities detected. Try selecting a different column or combine columns with richer data.")
        else:
            st.dataframe(ents_df.drop_duplicates().reset_index(drop=True))
            st.download_button("Download Entities CSV", data=ents_df.to_csv(index=False).encode("utf-8"), file_name="entities.csv")

    # Knowledge Graph
    elif task == "Knowledge Graph":
        st.header("Knowledge Graph")
        G = build_graph_from_df(df)
        st.write(f"Graph nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")
        plot_graph(G)
        if G.number_of_nodes()>0:
            nodes_df = pd.DataFrame([{"node": n, **G.nodes[n]} for n in G.nodes()])
            st.download_button("Download Nodes CSV", data=nodes_df.to_csv(index=False).encode("utf-8"), file_name="graph_nodes.csv")

    # Sentence Clustering
    elif task == "Sentence Clustering":
        st.header("Sentence Clustering")
        if len(sentences) == 0:
            st.warning("No sentences found to cluster. Try choosing a different column.")
        else:
            clust_df = cluster_sentences(sentences)
            st.dataframe(clust_df.head(200))
            st.download_button("Download Clusters CSV", data=clust_df.to_csv(index=False).encode("utf-8"), file_name="clusters.csv")

    # Topic Modeling
    elif task == "Topic Modeling":
        st.header("Topic Modeling")
        topics_df = topic_model_safe(sentences)
        if topics_df is None or (isinstance(topics_df, pd.DataFrame) and topics_df.empty):
            st.warning("No topics extracted (not enough or too-short text). Try a different column or upload a dataset with more textual fields.")
        else:
            st.dataframe(topics_df)
            st.download_button("Download Topics CSV", data=topics_df.to_csv(index=False).encode("utf-8"), file_name="topics.csv")

    # Sentiment
    elif task == "Sentiment Analysis":
        st.header("Sentiment Analysis")
        sent_res = sentiment_analysis_simple(text_blob)
        st.dataframe(sent_res)
        st.download_button("Download Sentiment CSV", data=sent_res.to_csv(index=False).encode("utf-8"), file_name="sentiment.csv")

    # CTI Classification
    elif task == "CTI Classification":
        st.header("CTI Classification (keyword tags)")
        tags = cti_keyword_tags(text_blob)
        st.write("Detected CTI categories:", ", ".join(tags))
        st.download_button("Download CTI Tags", data=",".join(tags).encode("utf-8"), file_name="cti_tags.txt")

st.write("---")
st.caption("If results still look empty: try selecting a different column above (e.g., Event_Description, message, indicator, or a username column). The app now builds relationships from structured columns like Source_IP and Destination_IP for a meaningful graph.")
