import streamlit as st
import pandas as pd
from nlp_logic import extract_text, extract_entities, build_cti_graph, plot_cti_graph

st.set_page_config(page_title="Cyber Threat Intelligence Analyzer", layout="wide")

st.title("Cyber Threat Intelligence (CTI) Analyzer")
st.markdown("Upload **CSV, TXT, or PDF** reports to extract threat entities and visualize relationships.")

# --- File upload ---
uploaded_file = st.file_uploader("Upload CTI Report", type=["csv", "txt", "pdf"])

if uploaded_file:
    with st.spinner("Extracting text..."):
        text = extract_text(uploaded_file)

    if not text.strip():
        st.error("No readable text found in the uploaded file.")
    else:
        st.success("Text extracted successfully!")

        with st.spinner("Running NLP analysis..."):
            df_entities = extract_entities(text)

        if df_entities.empty:
            st.warning("No cyber entities detected in the report.")
        else:
            st.subheader("Extracted Entities")
            st.dataframe(df_entities, use_container_width=True)

            # Build and visualize knowledge graph
            with st.spinner("Building knowledge graph..."):
                G = build_cti_graph(df_entities["Entity"], df_entities["Type"])
                fig = plot_cti_graph(G)

            st.subheader("Knowledge Graph Visualization")
            st.pyplot(fig)

            st.success(f"Extracted {len(df_entities)} entities and built {len(G.edges)} relationships.")

else:
    st.info("Please upload a file to begin analysis.")
