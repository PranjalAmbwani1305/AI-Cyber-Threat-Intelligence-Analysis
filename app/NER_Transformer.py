from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from collections import defaultdict
import fitz  # PyMuPDF
import io

def extract_text_from_pdf_file(uploaded_file):
    text = ""
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page in doc:
        text += page.get_text(sort=True)
    return text

def extract_entities_from_text(text):
    model_name = "CyberPeace-Institute/SecureBERT-NER"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    
    entities = nlp(text)
    grouped_entities = defaultdict(list)
    for ent in entities:
        grouped_entities[ent['entity_group']].append(ent['word'])
    
    results = []
    for entity_type, words in grouped_entities.items():
        results.append(f"{entity_type}: {', '.join(words)}")
    
    return results

def extract_entities_from_uploaded_file(uploaded_file):
    text = extract_text_from_pdf_file(uploaded_file)
    return extract_entities_from_text(text)
