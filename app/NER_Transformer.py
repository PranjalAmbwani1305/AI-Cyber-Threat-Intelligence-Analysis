import pymupdf

def extract_text_from_pdf(pdf_path):
    text = ""
    doc = pymupdf.open(pdf_path)
    for page in doc:
      text += page.get_text(sort=True)
    return text

text = extract_text_from_pdf("Aperture Labs Report.pdf")
print(text)

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from collections import defaultdict

def extract_cti_entities(pdf_path):
    model_name = "CyberPeace-Institute/SecureBERT-NER"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

    text = extract_text_from_pdf(pdf_path)
    entities = nlp(text)

    grouped_entities = defaultdict(list)
    for ent in entities:
      grouped_entities[ent['entity_group']].append(ent['word'])

    results = []
    for entity_type, words in grouped_entities.items():
      results.append(f"{entity_type}: {', '.join(words)}")
    return results


#pdf_file = "Aperture Labs Report.pdf"
pdf_file = "Innovant CyberSecurity Report.pdf"
entities = extract_cti_entities(pdf_file)

print("Extracted CTI Entities:")
for e in entities:
    print(e)

pdf_file = "Kaspersky Mobile Cyberthreat Report Q2 2025.pdf"
entities = extract_cti_entities(pdf_file)

print("Extracted CTI Entities:")
for e in entities:
    print(e)