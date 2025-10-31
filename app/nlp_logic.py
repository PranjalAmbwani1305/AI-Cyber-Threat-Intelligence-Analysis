import pandas as pd
import nltk
import io
import networkx as nx
from pyvis.network import Network
from transformers import pipeline

# Ensure NLTK tokenizer is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

def split_into_sentences(text):
    return nltk.sent_tokenize(text)

# Load NLP pipelines
ner_pipeline = pipeline("ner", grouped_entities=True)
sentiment_pipeline = pipeline("sentiment-analysis")

def extract_text(file):
    """Extract text from CSV, TXT, or PDF."""
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
        text = " ".join(df.astype(str).fillna("").values.flatten())
    elif file.name.endswith(".txt"):
        text = file.read().decode("utf-8", errors="ignore")
    elif file.name.endswith(".pdf"):
        import PyPDF2
        reader = PyPDF2.PdfReader(file)
        text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    else:
        text = ""
    return text

def build_knowledge_graph(entities):
    """Create an interactive knowledge graph using PyVis."""
    G = nx.Graph()

    for ent in entities:
        entity_type = ent.get("entity_group", "Unknown")
        word = ent.get("word", "")
        if not word.strip():
            continue
        G.add_node(entity_type, color="#007ACC", shape="ellipse")
        G.add_node(word, color="#00B4D8", shape="dot")
        G.add_edge(entity_type, word)

    net = Network(height="600px", width="100%", bgcolor="#0e1117", font_color="white")
    net.from_nx(G)
    net.toggle_physics(True)
    return net.generate_html()

def process_cti_data(uploaded_file):
    """Main NLP processing pipeline."""
    text = extract_text(uploaded_file)
    if not text.strip():
        raise ValueError("No readable text found in the uploaded file.")

    sentences = split_into_sentences(text)

    # Named Entity Recognition
    entities = []
    for sent in sentences[:50]:  # limited for performance
        entities.extend(ner_pipeline(sent))

    # Sentiment Analysis
    sentiments = sentiment_pipeline(sentences[:50])
    sentiment_df = pd.DataFrame(sentiments)
    sentiment_summary = sentiment_df["label"].value_counts().reset_index()
    sentiment_summary.columns = ["Sentiment", "Count"]

    # Knowledge Graph
    graph_html = build_knowledge_graph(entities)

    return {
        "entities": pd.DataFrame(entities),
        "sentiment_summary": sentiment_summary,
        "graph_html": graph_html
    }
