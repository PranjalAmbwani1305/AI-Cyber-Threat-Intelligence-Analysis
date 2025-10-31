import streamlit as st
import pandas as pd
from nlp_logic import load_model, extract_text, chunk_text, build_cti_graph, plot_cti_graph

# --- PAGE CONFIG ---
st.set_page_config(page_title="CTI Knowledge Graph Builder", layout="wide")

st.title("Cyber Threat Intelligence (CTI) Knowledge Graph Builder")
st.write("Upload a **CSV**, **PDF**, or **TXT** report to extract threat entities and visualize their relationships.")

# --- MODEL INITIALIZATION ---
@st.cache_resource
def init_model():
    return load_model()

tokenizer, ner_pipeline = init_model()

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("📁 Upload CTI Report (CSV, PDF, or TXT)", type=["csv", "pdf", "txt"])

if uploaded_file:
    with st.spinner("Extracting text..."):
        text = extract_text(uploaded_file)

    if not text.strip():
        st.error("No readable text could be extracted from this file.")
    else:
        st.success("✅ Text extracted successfully!")

        chunks = chunk_text(text, tokenizer)
        st.write(f"🔹 Processing {len(chunks)} text chunks...")

        results = []
        for chunk in chunks:
            results.extend(ner_pipeline(chunk))

        if not results:
            st.warning("⚠️ No cyber threat entities detected.")
        else:
            df = pd.DataFrame(results)
            df = df.rename(columns={"word": "Entity", "entity_group": "Type", "score": "Score"})
            df["Score"] = df["Score"].round(3)

            st.subheader("📋 Extracted Cyber Entities")
            st.dataframe(df[["Entity", "Type", "Score"]], use_container_width=True)

            with st.spinner("Building CTI Knowledge Graph..."):
                G = build_cti_graph(df["Entity"].tolist(), df["Type"].tolist())
                fig = plot_cti_graph(G)

            st.subheader("🌐 Knowledge Graph Visualization")
            st.pyplot(fig)

            st.success(f"✅ Extracted {len(G.vs)} entities and {len(G.es)} relationships.")
