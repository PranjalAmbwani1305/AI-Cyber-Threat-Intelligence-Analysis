import streamlit as st
from nlp_logic import process_cti_pdf, visualize_subgraph, perform_clustering, plot_clusters

st.set_page_config(page_title="CTI Workbench", layout="wide")
st.title("Cyber Threat Intelligence Workbench")

uploaded_file = st.file_uploader(" Upload a  Report (PDF)", type=["pdf"])

if uploaded_file:
    st.info("Processing uploaded report... this may take a moment.")
    with st.spinner("Extracting and analyzing..."):
        df_entities, graph, sentences = process_cti_pdf(uploaded_file)
    st.success("Processing complete!")

    tabs = st.tabs(["Entity Overview", "Knowledge Graph", "Topic Clustering"])

    with tabs[0]:
        st.subheader("Extracted Entities")
        st.dataframe(df_entities)

    with tabs[1]:
        st.subheader("Knowledge Graph Visualization")
        if graph and len(graph.vs) > 0:
            selected = st.selectbox("Select Entity", graph.vs["name"])
            if st.button("Show Subgraph"):
                fig, msg = visualize_subgraph(graph, selected)
                st.pyplot(fig)
                st.caption(msg)
        else:
            st.warning("No entities found for graph visualization.")

    with tabs[2]:
        st.subheader("Topic-Based Sentence Clustering")
        if st.button("Run Semantic Clustering"):
            embeddings, labels, topics = perform_clustering(sentences)
            fig = plot_clusters(embeddings, labels, topics)
            st.pyplot(fig)
else:
    st.write("Upload a PDF CTI report to begin analysis.")
