import streamlit as st
import pandas as pd
from nlp_logic import process_cti_data, build_cti_graph_pyvis

# ---------------- Streamlit Config ----------------
st.set_page_config(page_title="AI Cyber Threat Intelligence Dashboard", layout="wide")

st.title("AI Cyber Threat Intelligence Dashboard")
st.caption("Transform Cyber Threat Intelligence (CTI) data into actionable insights using NLP, entity recognition, clustering, and graph analytics.")

# ---------------- Tabs ----------------
tab1, tab2 = st.tabs(["Entity Analysis", "Knowledge Graph"])

# ---------------- Tab 1: Entity Analysis ----------------
with tab1:
    st.header("Extracted Entities and Insights")

    try:
        df_entities = process_cti_data()
        if not df_entities.empty:
            col1, col2, col3 = st.columns(3)
            col1.metric("Entities Extracted", len(df_entities))
            col2.metric("Entity Types", df_entities['Type'].nunique())
            col3.metric("Highest Confidence", round(df_entities['Score'].max(), 3))

            st.dataframe(df_entities, use_container_width=True, height=450)

            st.subheader("Entity Type Distribution")
            st.bar_chart(df_entities['Type'].value_counts())
        else:
            st.warning("No entities found in the processed data.")
    except Exception as e:
        st.error("Error processing entity data.")
        st.exception(e)

# ---------------- Tab 2: Knowledge Graph ----------------
with tab2:
    st.header("Cyber Threat Knowledge Graph")

    try:
        graph_html = build_cti_graph_pyvis()
        if graph_html:
            st.components.v1.html(graph_html, height=700, scrolling=True)
        else:
            st.warning("Knowledge graph could not be generated.")
    except Exception as e:
        st.error("Error generating knowledge graph.")
        st.exception(e)

st.markdown("---")
st.caption("© 2025 Cyber Threat Intelligence Dashboard | NLP & Graph Analytics")
