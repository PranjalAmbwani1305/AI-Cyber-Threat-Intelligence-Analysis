import streamlit as st
from nlp_logic import process_cti_data

st.set_page_config(page_title="AI Cyber Threat Intelligence Dashboard", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["CTI NLP Analysis", "Knowledge Graph"])

st.title("AI Cyber Threat Intelligence Dashboard")
st.write(
    "Analyze Cyber Threat Intelligence (CTI) data — structured or unstructured — "
    "using advanced NLP, entity recognition, clustering, and knowledge graph analytics."
)

uploaded_file = st.file_uploader(
    "Upload your CTI data file (CSV, TXT, or PDF):", type=["csv", "txt", "pdf"]
)

if uploaded_file is not None:
    with st.spinner("Processing your file... please wait..."):
        try:
            result = process_cti_data(uploaded_file)
            st.session_state["cti_result"] = result
            st.success("Data processed successfully.")
        except Exception as e:
            st.error(f"An error occurred during processing:\n\n{str(e)}")

# Page 1 — NLP Analysis
if page == "CTI NLP Analysis":
    if "cti_result" in st.session_state:
        result = st.session_state["cti_result"]

        st.subheader("Named Entities Extracted")
        st.dataframe(result["entities"], use_container_width=True)

        st.subheader("Sentiment Summary")
        st.dataframe(result["sentiment_summary"], use_container_width=True)
    else:
        st.info("Please upload a CTI file to start analysis.")

# Page 2 — Knowledge Graph
elif page == "Knowledge Graph":
    if "cti_result" in st.session_state:
        result = st.session_state["cti_result"]
        st.subheader("Knowledge Graph Visualization")
        st.components.v1.html(result["graph_html"], height=600)
    else:
        st.warning("Please upload and process a file first on the previous page.")
