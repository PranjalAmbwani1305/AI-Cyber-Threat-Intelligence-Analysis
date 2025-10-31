# app.py
import streamlit as st
import pandas as pd
from nlp_logic import load_model, extract_text, chunk_text, build_structured_cti_graph, plot_cti_graph_pyvis

st.set_page_config(page_title="Cyber Threat Intelligence Dashboard", layout="wide")

st.title("🛡️ Cyber Threat Intelligence (CTI) Dashboard")
st.caption("AI-powered NLP + Knowledge Graph for Cyber Analysis")

# -------------------- MODEL LOADING --------------------
@st.cache_resource
def init_model():
    return load_model()

tokenizer, ner_pipeline = init_model()

tabs = st.tabs(["📁 Upload & Extract", "🧠 Entity Recognition", "🌐 Knowledge Graph"])

# -------------------- TAB 1 --------------------
with tabs[0]:
    st.header("Upload Cyber Threat Report")
    uploaded_file = st.file_uploader("Upload CSV, PDF, or TXT", type=["csv", "pdf", "txt"])
    if uploaded_file:
        text = extract_text(uploaded_file)
        st.session_state["text"] = text
        st.success("✅ File processed successfully!")

# -------------------- TAB 2 --------------------
with tabs[1]:
    st.header("Entity Extraction (NER)")
    if "text" not in st.session_state:
        st.warning("Please upload a file first.")
    else:
        text = st.session_state["text"]
        chunks = chunk_text(text, tokenizer)
        st.write(f"Processing {len(chunks)} chunks...")
        results = [ent for chunk in chunks for ent in ner_pipeline(chunk)]
        df = pd.DataFrame(results).rename(columns={"word": "Entity", "entity_group": "Type", "score": "Score"})
        df["Score"] = df["Score"].round(3)
        st.dataframe(df, use_container_width=True)
        st.session_state["entities"] = df

# -------------------- TAB 3 --------------------
with tabs[2]:
    st.header("CTI Knowledge Graph")
    st.markdown("### 🔗 Visual Cyber Flow — Interactive Graph")

    G = build_structured_cti_graph()
    html_path = plot_cti_graph_pyvis(G)

    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    st.components.v1.html(html_content, height=720, scrolling=True)

    st.info("💡 Relationships are directional — hover or click nodes for more info.")
