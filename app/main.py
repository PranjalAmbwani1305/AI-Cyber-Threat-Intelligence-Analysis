import streamlit as st
import pandas as pd
from nlp_logic import extract_text, extract_entities, extract_keywords, topic_modeling, build_cti_graph, plot_cti_graph

st.set_page_config(page_title="Cyber Threat Intelligence Analyzer", layout="wide")

st.title("🧠 Cyber Threat Intelligence (CTI) Analyzer")
st.markdown("Analyze PDF or CSV reports to extract **threat entities**, **topics**, and build a **knowledge graph**.")

uploaded_file = st.file_uploader("📂 Upload CTI Report (PDF or CSV)", type=["pdf", "csv"])

if uploaded_file:
    with st.spinner("Extracting text..."):
        text = extract_text(uploaded_file)

    if text.strip() == "":
        st.error("⚠️ Could not extract text from the file.")
    else:
        st.success("✅ Text extracted successfully.")
        st.write("---")

        tab1, tab2, tab3, tab4 = st.tabs(["Entities", "Keywords", "Topics", "Knowledge Graph"])

        with tab1:
            st.subheader("Named Entity Recognition (NER)")
            df_entities = extract_entities(text)
            st.dataframe(df_entities, use_container_width=True)

        with tab2:
            st.subheader("Keyword Extraction")
            df_keywords = extract_keywords(text)
            st.dataframe(df_keywords, use_container_width=True)

        with tab3:
            st.subheader("Topic Modeling (LDA)")
            df_topics = topic_modeling(text)
            st.dataframe(df_topics, use_container_width=True)

        with tab4:
            st.subheader("Knowledge Graph")
            if not df_entities.empty:
                G = build_cti_graph(df_entities)
                fig = plot_cti_graph(G)
                st.pyplot(fig)
            else:
                st.warning("No entities detected to build a graph.")
