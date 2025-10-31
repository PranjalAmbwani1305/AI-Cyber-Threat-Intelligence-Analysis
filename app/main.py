import streamlit as st
import pandas as pd
import pdfplumber
from io import StringIO
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import igraph as ig
import matplotlib.pyplot as plt
import warnings

# ---------------------------------------------------
# Suppress Hugging Face warnings
# ---------------------------------------------------
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(page_title="Cyber Threat Intelligence Analyzer", layout="wide")

# ---------------------------------------------------
# GLOBAL MODEL INITIALIZATION
# ---------------------------------------------------
MODEL_NAME = "CyberPeace-Institute/SecureBERT-NER"
tokenizer, model, ner_pipeline = None, None, None
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
    ner_pipeline = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
except Exception as e:
    st.error(f"⚠️ Failed to load SecureBERT-NER model: {e}")

# ---------------------------------------------------
# SAFETY FUNCTIONS
# ---------------------------------------------------
def safe_read_csv(file):
    """Safely read CSV files with error handling."""
    try:
        df = pd.read_csv(file)
        if df.empty:
            st.warning("⚠️ Uploaded CSV file is empty.")
            return None
        return df
    except pd.errors.EmptyDataError:
        st.error("❌ Uploaded CSV file is empty.")
        return None
    except pd.errors.ParserError:
        st.error("❌ CSV format error. Please verify delimiter and structure.")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected CSV error: {e}")
        return None


def extract_pdf_text(pdf_file):
    """Extract text from PDF safely."""
    try:
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                content = page.extract_text()
                if content:
                    text += content + "\n"
        if not text.strip():
            st.warning("⚠️ No text extracted from PDF.")
        return text
    except Exception as e:
        st.error(f"❌ Error reading PDF: {e}")
        return ""


# ---------------------------------------------------
# NER LOGIC
# ---------------------------------------------------
def extract_entities(text):
    if not ner_pipeline:
        st.error("NER model not loaded.")
        return pd.DataFrame()
    results = ner_pipeline(text)
    df = pd.DataFrame(results)
    df = df.rename(columns={'word': 'Entity', 'entity_group': 'Type', 'score': 'Confidence'})
    df['Confidence'] = df['Confidence'].round(4)
    return df[['Entity', 'Type', 'Confidence']]


# ---------------------------------------------------
# KNOWLEDGE GRAPH CREATION
# ---------------------------------------------------
def build_graph(entities, types):
    G = ig.Graph(directed=True)
    vertices = list(set(entities))
    G.add_vertices(vertices)
    G.vs["label"] = vertices
    G.vs["type"] = types[:len(vertices)]

    # Random simple relation
    for i in range(len(vertices) - 1):
        G.add_edges([(vertices[i], vertices[i + 1])])
    return G


def plot_graph(G):
    layout = G.layout("fr")
    fig, ax = plt.subplots(figsize=(8, 6))
    ig.plot(G, target=ax, layout=layout, vertex_label=G.vs["label"], vertex_size=20)
    st.pyplot(fig)


# ---------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------
st.title("🧠 Cyber Threat Intelligence Analyzer")

tab1, tab2, tab3, tab4 = st.tabs(["📄 Upload Data", "🧩 NER Extraction", "🕸 Knowledge Graph", "📊 Threat Feed"])

# GLOBALS
uploaded_file = None
ner_df = pd.DataFrame()
graph_obj = None

# ---------------------------------------------------
# TAB 1 — UPLOAD DATA
# ---------------------------------------------------
with tab1:
    st.header("Upload CTI Data (PDF or CSV)")
    uploaded_file = st.file_uploader("Upload CTI report (.pdf) or Threat Feed (.csv)", type=["pdf", "csv"])
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = safe_read_csv(uploaded_file)
            if df is not None:
                st.success(f"✅ Loaded {len(df)} indicators from CSV.")
                st.dataframe(df.head(10), use_container_width=True)
        elif uploaded_file.name.endswith(".pdf"):
            text = extract_pdf_text(uploaded_file)
            if text:
                st.text_area("Extracted Text Preview", text[:2000], height=200)


# ---------------------------------------------------
# TAB 2 — NER Extraction
# ---------------------------------------------------
with tab2:
    st.header("Named Entity Recognition (NER)")
    sample_text = st.text_area("Paste text or use extracted PDF text", height=200)
    if st.button("Run NER"):
        if not sample_text.strip():
            st.warning("Please enter or extract text first.")
        else:
            ner_df = extract_entities(sample_text)
            st.dataframe(ner_df, use_container_width=True)
            st.success(f"Extracted {len(ner_df)} entities.")


# ---------------------------------------------------
# TAB 3 — KNOWLEDGE GRAPH
# ---------------------------------------------------
with tab3:
    st.header("Cyber Knowledge Graph")
    if not ner_df.empty:
        G = build_graph(ner_df["Entity"].tolist(), ner_df["Type"].tolist())
        st.success("Knowledge graph generated.")
        plot_graph(G)
    else:
        st.info("Please run NER first to visualize relationships.")


# ---------------------------------------------------
# TAB 4 — THREAT FEED (CSV VIEW)
# ---------------------------------------------------
with tab4:
    st.header("Threat Feed Viewer")
    threat_file = st.file_uploader("Upload Threat Feed CSV", type=["csv"], key="threat_csv")

    if threat_file:
        df = safe_read_csv(threat_file)
        if df is not None:
            col1, col2 = st.columns(2)
            with col1:
                indicator_type = st.selectbox("Indicator Type", ["All"] + sorted(df["Type"].unique()))
            with col2:
                min_conf = st.slider("Min Confidence", 0.0, 100.0, 0.0)

            filtered = df.copy()
            if indicator_type != "All":
                filtered = filtered[filtered["Type"] == indicator_type]
            filtered = filtered[filtered["Confidence"] >= min_conf]

            st.dataframe(filtered, use_container_width=True)
            st.metric("Indicator Count", len(filtered))
            if "Confidence" in filtered.columns:
                st.metric("Average Confidence", round(filtered["Confidence"].mean(), 2))
