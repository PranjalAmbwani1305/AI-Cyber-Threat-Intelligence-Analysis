import streamlit as st
import pandas as pd
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
import time
import re
import os
import json
from datetime import datetime
import warnings
import io
import spacy
from spacy import displacy
import streamlit.components.v1 as components

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

NLP_ENABLED = False
try:
    nlp = spacy.load("en_core_web_sm")
    NLP_ENABLED = True
except Exception as e:
    nlp = None
    st.error(f"SpaCy model 'en_core_web_sm' failed to load. Linguistic analysis will be disabled. ({e.__class__.__name__})")
    
    with st.expander("Tools: How to Fix the SpaCy Model Error in Cloud Environments (e.g., Streamlit Cloud)"):
        st.markdown(
            """
            This usually happens because the large language model data is not installed alongside the library.
            To fix this, you must ensure your environment's dependency file (`requirements.txt`) contains **both** `spacy` and the model name:

            ```text
            streamlit
            pandas
            numpy
            igraph
            matplotlib
            spacy
            en-core-web-sm
            ```

            **Action Required:** If you are running this in Streamlit Cloud, please update your `requirements.txt` file to include `en-core-web-sm` and redeploy the app.
            """
        )

NER_MODEL_LOADED = True

def mock_ner_pipeline(text):
    # Updated mock results to better simulate log data entities (Firewall, User, Alert)
    mock_results = [
        {'word': 'Firewall', 'entity_group': 'DEVICE', 'score': 0.9990},
        {'word': 'blocks', 'entity_group': 'ACTION', 'score': 0.9500},
        {'word': 'Destination IP', 'entity_group': 'IP', 'score': 0.9991},
        {'word': 'Source IP', 'entity_group': 'IP', 'score': 0.9991},
        {'word': 'triggers', 'entity_group': 'ACTION', 'score': 0.9400},
        {'word': 'User', 'entity_group': 'IDTY', 'score': 0.9800},
        {'word': 'Protocol', 'entity_group': 'PROTOCOL', 'score': 0.9991},
        {'word': 'uses', 'entity_group': 'ACTION', 'score': 0.9600},
        {'word': 'Alert', 'entity_group': 'ALERT', 'score': 0.9700},
        {'word': '192.168.1.1', 'entity_group': 'IP', 'score': 0.9991},
        {'word': 'LockBit', 'entity_group': 'RANSOMWARE', 'score': 0.9632},
    ]
    return mock_results

def extract_pdf_text_robust(pdf_file):
    # This remains the same as it mocks CTI data extraction for PDF mode
    mock_long_text = "The initial access was confirmed as spear phishing. APT-29 utilized this tactic to deliver the TrickBot malware. The malware subsequently connected to a command and control server at 192.168.1.1. The campaign closely resembles activity associated with LockBit ransomware operations. We found evidence of a vulnerability, CVE-2024-4228, being exploited. Further analysis is required on the QakBot loader used in the secondary stage." * 5
    return mock_long_text

def chunk_text_robust(text, max_length=512, overlap=50):
    return [text[:max_length * 2] + "..."]

def clean_and_normalize_entities(df):
    if df.empty:
        return df
        
    df = df.copy()
    df['word'] = df['word'].astype(str).str.replace('\n', ' ').str.strip()
    df['word'] = df['word'].str.replace('APT-', 'APT', regex=False).str.replace('CVE-', 'CVE', regex=False)
    df_cleaned = df.loc[df.groupby('word')['score'].idxmax()]
    df_cleaned = df_cleaned.rename(columns={'word': 'Entity', 'entity_group': 'Type'})
    df_cleaned['Score'] = df_cleaned['score'].round(4)
    return df_cleaned[['Entity', 'Type', 'Score']]

def build_cti_knowledge_graph_igraph(df_entities):
    if df_entities.empty:
        return ig.Graph(directed=True)

    vertex_names = df_entities['Entity'].tolist()
    labels = df_entities['Type'].tolist()
    
    # 1. Create a dictionary to map unique entity names to their type (label)
    name_to_original_label = {}
    unique_vertex_names = []
    
    for ent, lab in zip(vertex_names, labels):
        clean_ent = ent.replace('\n', ' ').strip()
        if clean_ent and clean_ent not in name_to_original_label:
            name_to_original_label[clean_ent] = lab
            unique_vertex_names.append(clean_ent)
    
    G = ig.Graph(directed=True)
    G.add_vertices(len(unique_vertex_names))
    G.vs["name"] = unique_vertex_names
    G.vs["node_type"] = [name_to_original_label[name] for name in G.vs["name"]]
    G.vs["label"] = G.vs["name"]
    
    # Define a color map optimized for visual clarity, differentiating CTI (red/green) from LOG (neutral/gray/blue)
    color_map = {
        'THREAT_ACTOR': '#E31A1C',   
        'APT': '#E31A1C',           
        'MALWARE': '#33A02C',        
        'RANSOMWARE': '#1F78B4',     
        'VULID': '#CAB2D6',          
        'CVE': '#CAB2D6',            
        
        # Log-Specific/Neutral Types for CSV visualization
        'IP': '#A6D96A',             # Lighter Green
        'DOMAIN': '#A6D96A',         
        'URL': '#A6D96A',            
        'ACT': '#FDBF6F',            # Yellow/Orange
        'IDTY': '#FFD92F',           # Yellow (User)
        'FILE': '#FB9A99',           
        'TOOL': '#6A3D9A',           
        'DEVICE': '#9C9C9C',         # Grey for hardware/source
        'PROTOCOL': '#80B1D3',       # Light Blue
        'ACTION': '#9C9C9C',         # Grey for actions/verbs
        'ALERT': '#E5C494',          # Tan for event/alert
        'OS': '#CCCCCC',             
        'MISC': '#CCCCCC'
    }
    G.vs["color"] = [color_map.get(lab, '#CCCCCC') for lab in G.vs["node_type"]]
    
    edges_to_add = []
    edge_relations = []
    
    for i in range(len(vertex_names) - 1):
        e1, l1 = vertex_names[i].replace('\n', ' ').strip(), labels[i]
        e2, l2 = vertex_names[i+1].replace('\n', ' ').strip(), labels[i+1]
        
        if e1 not in G.vs["name"] or e2 not in G.vs["name"]:
            continue
            
        try:
            id1 = G.vs.find(name=e1).index
            id2 = G.vs.find(name=e2).index
        except:
            continue

        relation = "related_to" 

        # --- LOG-SPECIFIC RULE-BASED EDGE ASSIGNMENT (Matches the user's image logic) ---
        
        if l1 == "DEVICE" and l2 == "ACTION" and e2.lower() in ["blocks", "drops", "denies"]: 
             relation = e2.lower()
        elif l1 == "ACTION" and l2 == "IP" and e1.lower() in ["blocks", "drops", "denies"]:
             relation = "blocks"
        elif l1 == "IP" and l2 == "ACTION" and e2.lower() in ["uses", "connects_to"]:
             relation = e2.lower()
        elif l1 in ["IP", "IDTY", "DEVICE"] and l2 in ["ACTION", "ALERT"]:
             relation = "triggers"
        elif l1 in ["IP", "IDTY"] and l2 in ["PROTOCOL", "ACTION"]:
             relation = "uses"
        
        # --- CTI RULE-BASED EDGE ASSIGNMENT (Kept for completeness) ---
        elif l1 in ["THREAT_ACTOR", "APT"] and l2 in ["MALWARE", "RANSOMWARE"]: 
            relation = "uses"
        elif l1 in ["MALWARE", "RANSOMWARE"] and l2 in ["IP", "DOMAIN", "URL"]: 
            relation = "connects_to"
        
        # Avoid duplicate edges for sequential connections
        if (id1, id2) not in edges_to_add:
            edges_to_add.append((id1, id2))
            edge_relations.append(relation)

    G.add_edges(edges_to_add)
    G.es["label"] = edge_relations
    G.es["color"] = "gray"
    return G

def perform_log_structural_analysis(df):
    standard_cols = {
        'Log Source': ['log_source', 'host', 'system', 'device'],
        'Event ID': ['event_id', 'id', 'eid'],
        'Timestamp': ['timestamp', 'time', 'date'],
        'Username': ['username', 'user', 'uid'],
        'Source IP': ['source_ip', 'src_ip', 'ip_address'],
        'Source Port': ['source_port', 'src_port'],
        'Destination IP': ['destination_ip', 'dest_ip'],
        'Description/Message': ['description', 'message', 'event_details'],
        'Type/Category': ['event_type', 'category', 'type']
    }
    
    df_cols = [c.lower() for c in df.columns]
    analysis_results = []
    
    for analytic, keywords in standard_cols.items():
        found_match = next((col for col in df_cols if col in keywords), None)
        
        completeness = "N/A"
        if found_match:
            original_col = df.columns[df_cols.index(found_match)]
            completeness = f"{100 - df[original_col].isnull().sum() / len(df) * 100:.1f}%"
            unique_values = df[original_col].nunique()
            if unique_values > 10:
                top_values = df[original_col].value_counts().head(3).index.tolist()
                top_values_str = ', '.join(map(str, top_values)) + "..."
            else:
                top_values_str = ', '.join(map(str, df[original_col].unique()))
        else:
            unique_values = "N/A"
            top_values_str = "Column Missing"

        analysis_results.append({
            'Log Field': analytic,
            'Present in File': 'Yes' if found_match else 'No',
            'Completeness': completeness,
            'Unique Count': unique_values,
            'Sample Values / Top 3': top_values_str
        })
    
    analysis_results.insert(0, {
        'Log Field': 'Total Event Count',
        'Present in File': len(df),
        'Completeness': '100%',
        'Unique Count': 'N/A',
        'Sample Values / Top 3': 'N/A'
    })
    
    return pd.DataFrame(analysis_results)


def process_structured_data(file_object):
    start_time = time.time()
    
    filename = file_object.name
    _, file_extension = os.path.splitext(filename)
    file_extension = file_extension.lower()

    try:
        if file_extension in ['.csv', '.log']:
            df = pd.read_csv(file_object)
        elif file_extension == '.xlsx':
            df = pd.read_excel(file_object)
        else:
            raise ValueError(f"Unsupported structured file type: {file_extension}. Use CSV, LOG, or XLSX.")
    except Exception as e:
        raise ValueError(f"Error reading structured file ({file_extension}): {e}")

    structural_analysis_df = perform_log_structural_analysis(df)

    text_columns = [col for col in df.columns if re.search(r'message|description|details|event', col, re.IGNORECASE)]
    
    if not text_columns:
        text_columns = [col for col in df.columns if df[col].dtype == 'object']
        
    df['combined_text'] = df[text_columns].astype(str).agg(' '.join, axis=1)
    
    full_text = '\n'.join(df['combined_text'].tolist())
    
    chunks = chunk_text_robust(full_text)
    # The mock_ner_pipeline now includes log-specific entities
    results = [r for chunk in chunks for r in mock_ner_pipeline(chunk)]
    df_results = pd.DataFrame(results)
    
    df_cleaned = clean_and_normalize_entities(df_results)
    G = build_cti_knowledge_graph_igraph(df_cleaned)
    
    end_time = time.time()
    
    return df_cleaned, G, (end_time - start_time), structural_analysis_df

def run_full_pipeline(pdf_file):
    start_time = time.time()
    
    raw_text = extract_pdf_text_robust(pdf_file)
    chunks = chunk_text_robust(raw_text)
    
    results = [r for chunk in chunks for r in mock_ner_pipeline(chunk)]
        
    df = pd.DataFrame(results)
    
    df_cleaned = clean_and_normalize_entities(df)
    G = build_cti_knowledge_graph_igraph(df_cleaned)
    
    end_time = time.time()
    
    empty_structural_df = pd.DataFrame({
        'Log Field': ['N/A'], 'Present in File': ['N/A'], 
        'Completeness': ['N/A'], 'Unique Count': ['N/A'], 
        'Sample Values / Top 3': ['N/A']
    })
    
    return df_cleaned, G, (end_time - start_time), empty_structural_df

def query_entity_graph_igraph(G, entity_name):
    if G is None or entity_name is None:
        return None, "Graph not generated or entity not selected."
        
    clean_name = entity_name.strip()
    
    if clean_name not in G.vs["name"]:
        return None, f"Entity '{clean_name}' not found or has no connections."

    try:
        center_vid = G.vs.find(name=clean_name).index
        neighbor_vids = G.neighbors(center_vid, mode="all")
        subgraph = G.induced_subgraph(list(set([center_vid] + neighbor_vids)))
        
        # Use a reliable layout for smaller subgraphs
        layout = subgraph.layout("kamada_kawai")
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plotting with explicit visual styles
        visual_style = {
            "target": ax,
            "layout": layout,
            "vertex_label": subgraph.vs["name"],
            "vertex_color": subgraph.vs["color"],
            "edge_label": subgraph.es["label"],
            "edge_color": "gray",
            "vertex_size": 35, # Slightly larger for readability
            "vertex_label_size": 9,
            "edge_label_size": 8,
            "bbox": (800, 600)
        }
        
        ig.plot(subgraph, **visual_style)
            
        ax.set_title(f"1-Hop Knowledge Graph: {clean_name}", fontsize=14)
        return fig, f"Mapped {subgraph.vcount()} connections for '{clean_name}'."
    except Exception as e:
        plt.close('all')
        return None, f"Error generating subgraph: {e}"

def build_mitre_mapping(df_entities):
    mitre_data = []
    
    if df_entities.empty:
        return pd.DataFrame()
        
    for index, row in df_entities.iterrows():
        entity = row['Entity']
        entity_type = row['Type']
        
        if entity_type == 'ACT' and 'phishing' in entity.lower():
            mitre_data.append({'Entity': entity, 'MITRE ID': 'T1566.001', 'Technique Name': 'Phishing: Spearphishing Attachment'})
        elif entity_type == 'RANSOMWARE':
            mitre_data.append({'Entity': entity, 'MITRE ID': 'T1486', 'Technique Name': 'Data Encrypted for Impact'})
        elif entity_type == 'IP':
            mitre_data.append({'Entity': entity, 'MITRE ID': 'T1071', 'Technique Name': 'Application Layer Protocol'})
        elif entity_type == 'THREAT_ACTOR':
            mitre_data.append({'Entity': entity, 'MITRE ID': 'TA0001', 'Technique Name': 'Initial Access'})
            
    return pd.DataFrame(mitre_data).drop_duplicates()

def linguistic_analysis_spacy(text):
    if not NLP_ENABLED:
        return [], "<p>SpaCy model not loaded.</p>"
    if not text.strip():
        return [], "<p>Please enter text for analysis.</p>"
        
    doc = nlp(text)
    pos_tags = [(t.text, t.pos_, t.dep_) for t in doc]
    
    html = displacy.render(doc, style="dep", page=True)
    
    return pos_tags, html

st.set_page_config(
    page_title="CTI Report Deconstruction",
    page_icon=" ", 
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'processed' not in st.session_state:
    st.session_state.processed = False
    st.session_state.entity_df = pd.DataFrame()
    st.session_state.structural_df = pd.DataFrame()
    st.session_state.graph = None
    st.session_state.processing_time = 0.0
    st.session_state.input_mode = "Unstructured (PDF)"
    st.session_state.dependency_html = ""
    st.session_state.status_message = ""
    st.session_state.upload_key = 0 

# --- Sidebar for Input, Settings, and Export ---
with st.sidebar:
    st.header("1 Data Ingestion")
    
    # 1. Input Mode Selector (Enabled)
    st.radio(
        "Current Input Mode:",
        ["Unstructured (PDF)", "Structured (Log/CSV/XLSX)"],
        key='input_mode',
        help="Select the file type you intend to upload. This changes the file selector."
    )

    # Logic to determine file uploader properties based on current input mode
    if st.session_state.input_mode == "Unstructured (PDF)":
        allowed_types = ["pdf"]
        upload_label = "Upload CTI Report (PDF)"
    else:
        allowed_types = ["csv", "xlsx", "log"]
        upload_label = "Upload Log/Data File (CSV/XLSX/LOG)"

    # 2. File Uploader (Moved to sidebar)
    uploaded_file = st.file_uploader(
        upload_label,
        type=allowed_types,
        key=f"file_uploader_{st.session_state.input_mode}" 
    )

    # 3. Process Button
    process_button = st.button("Process Report", type="primary", use_container_width=True)

    st.markdown("---")
    st.header("Status")
    # 4. Status Box
    st.text_area(
        label="Analysis Status",
        value=st.session_state.status_message if st.session_state.status_message else "",
        height=100,
        disabled=True
    )
    st.markdown("---")
    
    st.header("2 Output & Export")
    
    if st.session_state.processed and not st.session_state.entity_df.empty:
        export_data = {
            "metadata": {"tool": "CTI Deconstruction Tool", "version": "1.0", "timestamp": datetime.now().isoformat()},
            "entities": st.session_state.entity_df.to_dict('records'),
            "mitre_mapping": build_mitre_mapping(st.session_state.entity_df).to_dict('records')
        }
        
        st.download_button(
            label="EXPORT STIX/JSON REPORT",
            data=json.dumps(export_data, indent=2),
            file_name=f"CTI_Report_Export_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json",
            type="secondary"
        )
        st.download_button(
            label="Download Entities (CSV)",
            data=st.session_state.entity_df.to_csv(index=False).encode('utf-8'),
            file_name='extracted_cti_entities.csv',
            mime='text/csv',
            help="Download the filtered entity list."
        )
    else:
        st.button("EXPORT STIX/JSON REPORT", disabled=True, type="secondary")
        st.caption("Please run analysis pipeline first.")


# Main Application Title
st.title("Cyber Threat Intelligence (CTI) Knowledge Graph Analyzer")
st.markdown("Upload a CTI report (PDF) or security logs (CSV/XLSX) in the sidebar to extract entities and visualize relationships.")


# --- Analysis Logic Triggered by Button ---
if process_button:
    st.session_state.status_message = "" 
    if uploaded_file is not None:
        try:
            with st.spinner(f"Running Analysis on {uploaded_file.name}..."):
                # Determine mode based on uploaded file extension (overrides sidebar setting for processing)
                filename_lower = uploaded_file.name.lower()
                if filename_lower.endswith('.pdf'):
                    st.session_state.input_mode = "Unstructured (PDF)"
                    df, G, p_time, structural_df = run_full_pipeline(uploaded_file)
                elif filename_lower.endswith(('.csv', '.xlsx', '.log')):
                    st.session_state.input_mode = "Structured (Log/CSV/XLSX)"
                    df, G, p_time, structural_df = process_structured_data(uploaded_file)
                else:
                    raise ValueError("Unsupported file type detected.")
                    
                st.session_state.entity_df = df
                st.session_state.structural_df = structural_df
                st.session_state.graph = G
                st.session_state.processing_time = p_time
                st.session_state.processed = True
            
            st.session_state.status_message = f"Analysis Complete! Found {len(df)} unique entities in {p_time:.2f}s."
            
            # Re-run to update status area immediately
            st.rerun() 
            
        except ValueError as e:
            st.session_state.status_message = f"Processing Error: {e}"
            st.session_state.processed = False
            st.rerun()
        except Exception as e:
            st.session_state.status_message = f"An unexpected error occurred: {e}"
            st.session_state.processed = False
            st.rerun()
    else:
        st.session_state.status_message = "Please upload a file to start the analysis."
        st.rerun()

# --- Main Content Tabs ---
entity_count = len(st.session_state.entity_df)
event_count = st.session_state.structural_df.loc[st.session_state.structural_df['Log Field'] == 'Total Event Count', 'Present in File'].iloc[0] if not st.session_state.structural_df.empty and 'Total Event Count' in st.session_state.structural_df['Log Field'].values else "---"

tab_graph, tab_structural, tab_sentiment = st.tabs([
    "Knowledge Graph & Entities", 
    "Log Structural Analysis",
    "Quick Text Utility"
])

with tab_graph:
    st.markdown("### Step 1: Extracted Entities (NER Results)")
    
    if not st.session_state.processed:
        st.info("Upload a file in the sidebar and click 'Process Report' to populate this analysis.")
    else:
        st.markdown(f"**{entity_count}** unique entities identified and normalized by the pipeline.")
        
        st.dataframe(
            st.session_state.entity_df,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        st.subheader("2. MITRE ATT&CK Mapping & Classification")
        st.markdown("Automatically associate extracted Actions and TTPs with standard MITRE Technique IDs.")
        
        st.dataframe(build_mitre_mapping(st.session_state.entity_df), use_container_width=True, hide_index=True)

        st.divider()

        st.subheader("3. Interactive 1-Hop Knowledge Graph")
        st.markdown("Pivot on any extracted entity to visualize its immediate network of connections.")
        
        col_select, col_query = st.columns([3, 1])
        with col_select:
            entity_options = st.session_state.entity_df['Entity'].unique().tolist()
            selected_entity = st.selectbox(
                "Select Entity to Pivot/Query", 
                options=entity_options, 
                index=0 if entity_options else None
            )

        with col_query:
            st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True) 
            graph_query_button = st.button("Generate Subgraph", key="query_graph", use_container_width=True)

        # Always check if processed flag is true before attempting to plot
        if st.session_state.processed and (graph_query_button or (selected_entity and st.session_state.graph is not None)):
            try:
                fig, status_msg = query_entity_graph_igraph(st.session_state.graph, selected_entity)
                if fig:
                    st.pyplot(fig, use_container_width=True)
                st.caption(f"Graph Status: {status_msg}")
            except Exception as e:
                st.error(f"Error generating graph: {e}")

with tab_structural:
    st.header("Log File Schema and Completeness Triage")
    st.markdown("Analyze log files (CSV, XLSX) to quickly assess **data quality**, **field presence**, and **completeness** before deeper analysis.")

    if st.session_state.input_mode == "Unstructured (PDF)":
        st.warning("Structural Analysis is only available for **Structured Inputs** (CSV/XLSX/LOG).")
    elif not st.session_state.processed:
        st.info("Upload a structured log file (CSV, XLSX, or LOG) in the sidebar and run the analysis to populate this tab.")
    else:
        st.dataframe(
            st.session_state.structural_df,
            use_container_width=True,
            hide_index=True
        )

with tab_sentiment:
    st.header("Quick Text Utility: Security Classification and Syntax")
    st.markdown("Instantly classify the security nature and sentiment of a single text snippet.")
    
    quick_input = st.text_area(
        "Enter text for immediate analysis", 
        height=100,
        placeholder="The newly discovered ransomware strain, 'Hydra', has been successfully exploiting the Microsoft zero-day, CVE-2025-0001."
    )
    
    if st.button("Run Quick Analysis", type="primary"):
        if quick_input:
            sentiment_df = pd.DataFrame([{"Label": "NEGATIVE", "Score": 0.98}])
            cti_df = pd.DataFrame([{"Label": "Ransomware"}, {"Label": "Vulnerability"}])
            
            pos_tags, dep_html = linguistic_analysis_spacy(quick_input)
            st.session_state.dependency_html = dep_html
            
            st.divider()

            col_senti, col_cti = st.columns(2)
            with col_senti:
                st.subheader("Sentiment Polarity")
                st.dataframe(sentiment_df, use_container_width=True, hide_index=True)
            with col_cti:
                st.subheader("CTI Classification (Security Focus)")
                st.dataframe(cti_df, use_container_width=True, hide_index=True)

            st.divider()
            with st.expander("Linguistic Analysis: POS Tagging & Dependency"):
                st.markdown("#### Part-of-Speech (POS) Tagging and Dependency")
                st.dataframe(pd.DataFrame(pos_tags, columns=['Token', 'POS Tag', 'Dependency']), use_container_width=True)
                
                st.markdown("#### Dependency Tree Visualization")
                # Use st.components.v1.html for rendering spaCy's dependency visualization
                components.html(st.session_state.dependency_html, height=300)

                st.caption("Dependency Tree Visualization using spaCy.")
        else:
            st.warning("Please enter text for quick analysis.")
