import spacy
from spacy.training.example import Example

# Example CTI training data (you should expand this dataset)
TRAIN_DATA = [
    ("The malware TrickBot contacted 192.168.1.5",
     {"entities": [(12, 20, "MALWARE"), (31, 42, "IP")]}),

    ("APT29 used spear phishing with the domain evil.com",
     {"entities": [(0, 5, "THREAT_ACTOR"), (42, 50, "DOMAIN")]}),

    ("CVE-2021-44228 was exploited in the attack",
     {"entities": [(0, 15, "CVE")]}),
]


# Load base model
nlp = spacy.load("en_core_web_sm")

# Add new NER pipe if not exists
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner")
else:
    ner = nlp.get_pipe("ner")

# Add new labels
labels = ["MALWARE", "IP", "DOMAIN", "CVE", "THREAT_ACTOR"]
for label in labels:
    ner.add_label(label)

# Disable other pipes during training
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.resume_training()
    for epoch in range(30):
        losses = {}
        for text, annotations in TRAIN_DATA:
            example = Example.from_dict(nlp.make_doc(text), annotations)
            nlp.update([example], drop=0.3, losses=losses)
        print(f"Epoch {epoch} Losses: {losses}")


test_text = "Indicators show TrickBot linked to 8.8.8.8 and CVE-2019-1234 exploited."
doc = nlp(test_text)

print("Entities Found:")
for ent in doc.ents:
    print(ent.text, ent.label_)


nlp.to_disk("./cti_ner_model")
print("Model saved to ./cti_ner_model")


nlp = spacy.load("./cti_ner_model")

import PyPDF2

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text


def extract_cti_entities(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    doc = nlp(text)

    results = []
    for ent in doc.ents:
        results.append({"entity": ent.text, "label": ent.label_})
    return results


pdf_file = "/content/Aperture Labs Report.pdf"
entities = extract_cti_entities(pdf_file)

print("Extracted CTI Entities:")
for e in entities:
    print(e)