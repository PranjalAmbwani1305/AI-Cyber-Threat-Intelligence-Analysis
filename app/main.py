import streamlit as st
import pandas as pd
from nlp_logic import load_model, extract_text, chunk_text, build_cti_graph, plot_cti_graph

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------
st.set_page_config(page_title="Cyber Threat Intelligence Dashboard", layout="wide")

# -------------------------------
# PAGE HEADER
# -------------------------------
st.title("Cyber Threat Intelligence (CTI) Dashboard")

# -------------------------------
# LOAD NLP MODEL
# -------------------------------
@st.cache_resource
def init_model():
    return load_model()

tokenizer, ner_pipeline = init_model()

# -------------------------------
# SIDEBAR - FILE UPLOAD
# -------------------------------
st.sidebar.header("File Upload")
uploaded_file = st.sidebar.file_uploader("Upload Threat Feed (.csv or .pdf)", type=["csv", "pdf"])

# -------------------------------
# TAB NAVIGATION
# -------------------------------
tabs = st.tabs(["Overview", "NER", "Knowledge Graph", "Threat Feed"])

# -------------------------------------
# TAB 1 - OVERVIEW
# -------------------------------------
with tabs[0]:
    st.header("Overview")
    st.markdown("""
    This dashboard performs **Cyber Threat Intelligence (CTI)** analysis using NLP.  
    Upload **CSV or PDF** threat reports, extract entities such as **IPs, domains, and malware**,  
    and visualize connections through a **knowledge graph**.
    """)

# -------------------------------------
# TAB 2 - NER (Entity Extraction)
# -------------------------------------
with tabs[1]:
    st.header("Named Entity Recognition (NER) Analysis")

    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            text = "\n".join(pd.read_csv(uploaded_file, encoding="utf-8", dtype=str).astype(str).fillna("").values.flatten())
        else:
            text = extract_text(uploaded_file)

        if not text.strip():
            st.error("No text could be extracted.")
        else:
            st.success("Text extracted successfully!")

            chunks = chunk_text(text, tokenizer)
            st.write(f"Processing {len(chunks)} text chunks...")

            results = []
            for chunk in chunks:
                results.extend(ner_pipeline(chunk))

            if not results:
                st.warning("No entities found in document.")
            else:
                df_entities = pd.DataFrame(results)
                df_entities = df_entities.rename(columns={"word": "Entity", "entity_group": "Type", "score": "Confidence"})
                df_entities["Confidence"] = df_entities["Confidence"].round(3)

                st.dataframe(df_entities, use_container_width=True)

# -------------------------------------
# TAB 3 - KNOWLEDGE GRAPH
# -------------------------------------
with tabs[2]:
    st.header("Knowledge Graph")

    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            text = "\n".join(pd.read_csv(uploaded_file, encoding="utf-8", dtype=str).astype(str).fillna("").values.flatten())
        else:
            text = extract_text(uploaded_file)

        if text.strip():
            chunks = chunk_text(text, tokenizer)
            results = []
            for chunk in chunks:
                results.extend(ner_pipeline(chunk))

            if results:
                df_graph = pd.DataFrame(results)
                df_graph = df_graph.rename(columns={"word": "Entity", "entity_group": "Type"})
                G = build_cti_graph(df_graph["Entity"].tolist(), df_graph["Type"].tolist())
                fig = plot_cti_graph(G)
                st.pyplot(fig)
            else:
                st.warning("No graph data available.")
        else:
            st.warning("Upload a valid file to generate graph.")
    else:
        st.info("Please upload a file to generate the graph.")

# -------------------------------------
# TAB 4 - THREAT FEED
# -------------------------------------
with tabs[3]:
    st.header("Threat Feed")

    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            try:
                df_feed = pd.read_csv(uploaded_file)
            except pd.errors.EmptyDataError:
                st.error("The CSV file is empty or invalid.")
                df_feed = None

            if df_feed is not None and not df_feed.empty:
                # Filters
                col1, col2, col3 = st.columns(3)
                with col1:
                    indicator_type = st.selectbox("Indicator Type", ["All"] + sorted(df_feed["Type"].unique()))
                with col2:
                    confidence_filter = st.selectbox("Confidence", ["All", "High", "Medium", "Low"])
                with col3:
                    source_filter = st.selectbox("Source", ["All"] + sorted(df_feed["Source"].astype(str).unique()))

                if indicator_type != "All":
                    df_feed = df_feed[df_feed["Type"] == indicator_type]
                if confidence_filter != "All":
                    if confidence_filter == "High":
                        df_feed = df_feed[df_feed["Confidence"] > 80]
                    elif confidence_filter == "Medium":
                        df_feed = df_feed[(df_feed["Confidence"] > 40) & (df_feed["Confidence"] <= 80)]
                    else:
                        df_feed = df_feed[df_feed["Confidence"] <= 40]
                if source_filter != "All":
                    df_feed = df_feed[df_feed["Source"].astype(str) == source_filter]

                st.dataframe(df_feed, use_container_width=True)

                col4, col5 = st.columns(2)
                with col4:
                    st.metric("Indicator Count", len(df_feed))
                with col5:
                    st.metric("Average Confidence", round(df_feed["Confidence"].mean(), 2))
            else:
                st.warning("No valid data found in CSV file.")
        else:
            st.info("Threat feed table is only available for CSV uploads.")
    else:
        st.info("Upload a CSV or PDF to begin analysis.")
