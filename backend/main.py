import os
from fastapi import FastAPI
from pydantic import BaseModel
from common import (
    run_ner_on_text,
    extract_pdf_text_pypdf2,
    extract_pdf_text_fitz,
    chunk_text,
    extract_cti_entities_fitz,
    build_cti_knowledge_graph_igraph,
    build_cti_knowledge_graph_nx,
    sentence_model
)
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
from bertopic import BERTopic
import pandas as pd

# ----------------------
# --- Hugging Face Token ---
# ----------------------
HF_TOKEN = os.getenv("HF_TOKEN")  # Set this in Vercel environment variables

app = FastAPI(title="CTI NLP API with HuggingFace")

# ----------------------
# --- REQUEST MODELS ---
# ----------------------
class TextInput(BaseModel):
    text: str

class PDFInput(BaseModel):
    pdf_path: str

class ClusterInput(BaseModel):
    texts: list[str]

class GraphInput(BaseModel):
    entities: list[str]
    labels: list[str]

# ----------------------
# --- Initialize HuggingFace Pipelines ---
# ----------------------
ner_pipeline = None
sentiment_pipeline = None

if HF_TOKEN:
    try:
        # NER pipeline
        ner_pipeline = pipeline(
            "token-classification",
            model="CyberPeace-Institute/SecureBERT-NER",
            aggregation_strategy="simple",
            use_auth_token=HF_TOKEN
        )
        # Sentiment pipeline
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            use_auth_token=HF_TOKEN
        )
    except Exception as e:
        print(f"Error initializing HF pipelines: {e}")

# ----------------------
# --- NER ENDPOINTS ---
# ----------------------
@app.post("/ner")
def ner_endpoint(data: TextInput):
    if not ner_pipeline:
        return {"error": "NER model not loaded"}
    results = ner_pipeline(data.text)
    return {"result": results}

@app.post("/ner_pdf_fitz")
def ner_pdf_fitz_endpoint(data: PDFInput):
    return extract_cti_entities_fitz(data.pdf_path)

# ----------------------
# --- PDF EXTRACTION ---
# ----------------------
@app.post("/extract_pdf_pypdf2")
def extract_pdf_pypdf2_endpoint(data: PDFInput):
    return extract_pdf_text_pypdf2(data.pdf_path)

@app.post("/extract_pdf_fitz")
def extract_pdf_fitz_endpoint(data: PDFInput):
    return extract_pdf_text_fitz(data.pdf_path)

# ----------------------
# --- CLUSTERING ---
# ----------------------
@app.post("/cluster")
def cluster_endpoint(data: ClusterInput):
    if not sentence_model:
        return {"error": "Sentence model not loaded"}
    embeddings = sentence_model.encode(data.texts)
    return {"embeddings": embeddings.tolist()}

# ----------------------
# --- GRAPH ENDPOINTS ---
# ----------------------
@app.post("/graph_igraph")
def graph_igraph_endpoint(data: GraphInput):
    G = build_cti_knowledge_graph_igraph(data.entities, data.labels)
    return {
        "nodes": [{"name": v["name"], "type": v["node_type"], "color": v["color"]} for v in G.vs],
        "edges": [{"source": e.source, "target": e.target, "label": e["label"]} for e in G.es]
    }

@app.post("/graph_nx")
def graph_nx_endpoint(data: GraphInput):
    df = pd.DataFrame({"Entity": data.entities, "Type": data.labels})
    G = build_cti_knowledge_graph_nx(df)
    nodes = [{"name": n, **G.nodes[n]} for n in G.nodes()]
    edges = [{"source": u, "target": v, **d} for u, v, d in G.edges(data=True)]
    return {"nodes": nodes, "edges": edges}

# ----------------------
# --- SENTIMENT ENDPOINT ---
# ----------------------
@app.post("/sentiment")
def sentiment_endpoint(data: TextInput):
    if not sentiment_pipeline:
        return {"error": "Sentiment model not loaded"}
    results = sentiment_pipeline(data.text)
    return {"result": results}

# ----------------------
# --- TOPIC MODELING (BERTopic) ---
# ----------------------
@app.post("/topic_model")
def topic_model_endpoint(data: ClusterInput):
    texts = data.texts
    if not texts:
        return {"error": "No text provided"}
    topic_model = BERTopic(verbose=False)
    topics, probs = topic_model.fit_transform(texts)
    fig = topic_model.visualize_barchart(top_n_topics=5)
    return {"topics": topics, "visualization": fig.to_html()}
