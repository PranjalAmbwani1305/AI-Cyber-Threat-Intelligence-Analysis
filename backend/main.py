from fastapi import FastAPI
from pydantic import BaseModel
from clustering import run_clustering
from ner import run_ner
from sentiment import analyze_sentiment

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/ner")
def ner_endpoint(data: TextInput):
    return run_ner(data.text)

@app.post("/sentiment")
def sentiment_endpoint(data: TextInput):
    return analyze_sentiment(data.text)

@app.post("/cluster")
def cluster_endpoint(data: TextInput):
    return run_clustering([data.text])
