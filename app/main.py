import streamlit as st
import pandas as pd
from nlp_logic import load_model, extract_text, chunk_text, build_cti_graph, plot_cti_graph

st.set_page_config(page_title="Cyber Threat Intelligence Analyzer", layout="wide")

st.title("🧠 Cyber Threat Intelligence (CTI) Knowledge Graph Builder")
st.markdown("""
Upload a **CSV**, **TXT**, or **PDF** report to extract cybersecurity entities using NLP  
and visualize threat relationships through an intelligent Knowledge Graph.
""")

@st.cache_resource
def init_model():
    return load_model()

tokenizer, ner_pipeline = init_model()

uploaded_file = st.file_uploader("Upload CTI Report", type=["csv", "pdf", "txt"])

if uploaded_file:
    with st.spinner("Extracting text..."):
        text = extract_text(uploaded_file)

    if not text.strip():
        st.error("No readable text found in the uploaded file.")
    else:
        st.success("✅ Text successfully extracted.")

        chunks = chunk_text(text, tokenizer)
        st.write(f"Processing {len(chunks)} text segments...")

        results = []
        for chunk in chunks:
            results.extend(ner_pipeline(chunk))

        if not results:
            st.warning("No cybersecurity entities found.")
        else:
            df = pd.DataFrame(results)
            df = df.rename(columns={"word": "Entity", "entity_group": "Type", "score": "Confidence"})
            df["Confidence"] = df["Confidence"].round(3)

            st.subheader("📋 Extracted Cyber Entities")
            st.dataframe(df[["Entity", "Type", "Confidence"]], use_container_width=True)

            with st.spinner("Building Knowledge Graph..."):
                G = build_cti_graph(df["Entity"].tolist(), df["Type"].tolist())
                fig = plot_cti_graph(G)

            st.subheader("🌐 Knowledge Graph Visualization")
            st.pyplot(fig)
            st.success(f"Extracted {len(G.nodes)} entities and {len(G.edges)} relationships.")
