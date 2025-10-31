import streamlit as st
import pandas as pd
import time
from nlp_logic import load_model, extract_text, chunk_text, build_cti_graph, plot_cti_graph

# --- PAGE CONFIG ---
st.set_page_config(page_title="CTI Dashboard", layout="wide")

st.title("🧠 Streamlit Frontend")
tabs = st.tabs(["Overview", "NER", "Knowledge Graph", "Threat Feed"])

# --- CACHING (For Speed) ---
@st.cache_resource(show_spinner=False)
def init_model():
    return load_model()

@st.cache_data(show_spinner=False)
def load_csv(file):
    return pd.read_csv(file)

tokenizer, ner_pipeline = init_model()

# --- SIDEBAR ---
with st.sidebar:
    st.header("File Uploader")
    uploaded_file = st.file_uploader("Upload threat feed (.csv, .pdf, .txt)")
    use_sample = st.checkbox("Use sample data")

# --- OVERVIEW TAB ---
with tabs[0]:
    st.subheader("Overview")
    st.markdown("""
    This dashboard analyzes **Cyber Threat Intelligence (CTI)** feeds,
    extracts indicators, and visualizes relationships between entities.
    """)

# --- NER TAB ---
with tabs[1]:
    st.subheader("Named Entity Recognition")
    if uploaded_file or use_sample:
        with st.spinner("Extracting entities..."):
            if use_sample:
                text = "APT29 used Cobalt Strike to target Windows 10 vulnerability CVE-2021-34527."
            else:
                text = extract_text(uploaded_file)

            # Chunk text quickly (no redundant tokenization)
            chunks = chunk_text(text, tokenizer)
            entities = []
            for chunk in chunks:
                entities.extend(ner_pipeline(chunk))

        if entities:
            df = pd.DataFrame(entities).rename(
                columns={"word": "Entity", "entity_group": "Type", "score": "Confidence"}
            )
            df["Confidence"] = (df["Confidence"] * 100).round(1)
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("No entities detected.")
    else:
        st.info("Upload a file or use sample data to begin.")

# --- KNOWLEDGE GRAPH TAB ---
with tabs[2]:
    st.subheader("Knowledge Graph")
    if uploaded_file or use_sample:
        if use_sample:
            df_entities = pd.DataFrame({
                "Entity": ["APT29", "Cobalt Strike", "Windows 10", "CVE-2021-34527"],
                "Type": ["APT", "Tool", "OS", "CVE"]
            })
        else:
            text = extract_text(uploaded_file)
            chunks = chunk_text(text, tokenizer)
            results = []
            for chunk in chunks:
                results.extend(ner_pipeline(chunk))
            df_entities = pd.DataFrame(results).rename(
                columns={"word": "Entity", "entity_group": "Type"}
            )

        if not df_entities.empty:
            with st.spinner("Building knowledge graph..."):
                G = build_cti_graph(df_entities["Entity"], df_entities["Type"])
                fig = plot_cti_graph(G)
            st.pyplot(fig)
        else:
            st.warning("No entities to visualize.")
    else:
        st.info("Upload a file or use sample data to view the graph.")

# --- THREAT FEED TAB ---
with tabs[3]:
    st.subheader("Threat Feed")

    # Load data
    if uploaded_file:
        with st.spinner("Loading CSV..."):
            df_feed = load_csv(uploaded_file)
    elif use_sample:
        df_feed = pd.DataFrame({
            "Indicator": ["105.4.302.40", "125.8.33.228", "200.200.0.86", "231.407.96.197", "222.0.13.1"],
            "Type": ["Domain", "Domain", "Domain", "Malware", "Domain"],
            "Confidence": [1.8, 1.9, 100, 100, 100],
            "Source": ["True", "False", "True", "False", "True"]
        })
    else:
        df_feed = pd.DataFrame()

    if not df_feed.empty:
        st.markdown("### Filters")

        col1, col2, col3 = st.columns(3)
        with col1:
            type_filter = st.selectbox("Indicator Type", ["All"] + sorted(df_feed["Type"].unique().tolist()))
        with col2:
            confidence_filter = st.slider("Confidence", 0.0, 100.0, (0.0, 100.0))
        with col3:
            source_filter = st.selectbox("Source", ["All"] + sorted(df_feed["Source"].unique().tolist()))

        # Fast filter operations
        mask = (df_feed["Confidence"] >= confidence_filter[0]) & (df_feed["Confidence"] <= confidence_filter[1])
        if type_filter != "All":
            mask &= df_feed["Type"] == type_filter
        if source_filter != "All":
            mask &= df_feed["Source"] == source_filter

        filtered = df_feed[mask]

        st.dataframe(filtered, use_container_width=True)
        st.markdown(f"**Indicator Count:** {len(filtered)}")
        st.markdown(f"**Average Confidence:** {filtered['Confidence'].mean():.2f}")
    else:
        st.info("Upload a CSV or use sample data to view threat feed.")
