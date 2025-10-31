import spacy
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from PyPDF2 import PdfReader
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import warnings

warnings.filterwarnings("ignore")

# ----------------------------------------------------------
# LOAD MODELS SAFELY
# ----------------------------------------------------------
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

MODEL_NAME = "CyberPeace-Institute/SecureBERT-NER"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
ner_pipeline = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# ----------------------------------------------------------
# TEXT EXTRACTION
# ----------------------------------------------------------
def extract_text(file):
    if file.name.endswith(".pdf"):
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text
    elif file.name.endswith(".csv"):
        df = pd.read_csv(file)
        text = " ".join(df.astype(str).apply(lambda x: ' '.join(x), axis=1))
        return text
    else:
        return ""

# ----------------------------------------------------------
# NLP ANALYSIS
# ----------------------------------------------------------
def extract_entities(text):
    results = ner_pipeline(text)
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.rename(columns={'word': 'Entity', 'entity_group': 'Type'})
        df['Score'] = df['score'].round(3)
        df = df[['Entity', 'Type', 'Score']]
    return df

def extract_keywords(text, top_n=10):
    doc = nlp(text)
    words = [token.text.lower() for token in doc if token.is_alpha and not token.is_stop]
    freq = pd.Series(words).value_counts().head(top_n)
    return freq.reset_index().rename(columns={'index': 'Keyword', 0: 'Frequency'})

def topic_modeling(text, num_topics=3):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform([text])
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(X)
    topics = []
    for idx, topic in enumerate(lda.components_):
        top_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-5:]]
        topics.append({"Topic": f"Topic {idx+1}", "Keywords": ", ".join(top_words)})
    return pd.DataFrame(topics)

# ----------------------------------------------------------
# KNOWLEDGE GRAPH
# ----------------------------------------------------------
def build_cti_graph(df):
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_node(row['Entity'], label=row['Type'])
    for i in range(len(df) - 1):
        G.add_edge(df.iloc[i]['Entity'], df.iloc[i+1]['Entity'], relation="related_to")
    return G

def plot_cti_graph(G):
    pos = nx.spring_layout(G, k=0.5, seed=42)
    plt.figure(figsize=(10, 8))
    node_labels = nx.get_node_attributes(G, 'label')
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1200, font_size=8, font_weight='bold', edge_color='gray')
    return plt.gcf()
