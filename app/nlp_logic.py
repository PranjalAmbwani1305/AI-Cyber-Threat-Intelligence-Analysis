import streamlit as st
from nlp_module import process_cti_data
import pandas as pd

st.set_page_config(page_title="CTI Knowledge Graph Analyzer", layout="wide")

st.title("🧠 Cyber Threat Intelligence (CTI) Knowledge Graph Analyzer")

uploaded_file = st.file_uploader("Upload CTI Report (PDF, TXT, or CSV)", type=["pdf", "txt", "csv"])
process_button = st.button("Process Report")

if process_button and uploaded_file:
    try:
        result = process_cti_data(uploaded_file)
        st.success("✅ Processing complete!")

        st.header("Extracted Entities")
        st.dataframe(result["entities"])

        st.header("Sentiment Summary")
        st.bar_chart(result["sentiment_summary"].set_index("Sentiment"))

        st.header("Knowledge Graph Visualization")
        st.markdown(f'<iframe src="{result["graph_html"]}" width="100%" height="600"></iframe>', unsafe_allow_html=True)

        st.image(result["graph_img"], caption="Static Knowledge Graph", use_container_width=True)

    except Exception as e:
        st.error(f"Error: {str(e)}")
