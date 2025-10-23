from fastapi import FastAPI
from pydantic import BaseModel
from common import (
    run_ner_on_text,
    extract_pdf_text_pypdf2,
    extract_pdf_text_fitz,
    chunk_text,
    extract_cti_entities_fitz,
    build_cti_knowledge_graph_igraph,
    build_cti_knowledge_graph_nx
)
from common import sentence_model, sentiment_model, sentiment_tokenizer
from transformers import pipeline

app = FastAPI(title="CTI NLP API")

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
# --- NER ENDPOINTS ---
# ----------------------
@app.post("/ner")
def ner_endpoint(data: TextInput):
    return run_ner_on_text(data.text)

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
# --- CLUSTERING (sentence embeddings) ---
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
    import pandas as pd
    df = pd.DataFrame({"Entity": data.entities, "Type": data.labels})
    G = build_cti_knowledge_graph_nx(df)
    nodes = [{"name": n, **G.nodes[n]} for n in G.nodes()]
    edges = [{"source": u, "target": v, **d} for u, v, d in G.edges(data=True)]
    return {"nodes": nodes, "edges": edges}

# ----------------------
# --- SENTIMENT (simple pipeline) ---
# ----------------------
@app.post("/sentiment")
def sentiment_endpoint(data: TextInput):
    sentiment_pipeline = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=sentiment_tokenizer)
    return sentiment_pipeline(data.text)
