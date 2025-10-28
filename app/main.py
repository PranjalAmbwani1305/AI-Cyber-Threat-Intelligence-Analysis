import streamlit as st
import pandas as pd
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
import io
import warnings
import json
import time
import re

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

NER_MODEL_LOADED = True

def mock_ner_pipeline(text):
    mock_results = [
        {'word': 'APT29', 'entity_group': 'THREAT_ACTOR', 'score': 0.9921},
        {'word': 'TrickBot', 'entity_group': 'MALWARE', 'score': 0.9785},
        {'word': '192.168.1.1', 'entity_group': 'IP', 'score': 0.9991},
        {'word': 'spear phishing', 'entity_group': 'ACT', 'score': 0.9540},
        {'word': 'QakBot', 'entity_group': 'MALWARE', 'score': 0.9810},
        {'word': 'CVE-2024-4228', 'entity_group': 'VULID', 'score': 0.9950},
        {'word': 'LockBit', 'entity_group': 'RANSOMWARE', 'score': 0.9632},
        {'word': 'APT-29', 'entity_group': 'THREAT_ACTOR', 'score': 0.9800},
        {'word': 'phishing', 'entity_group': 'ACT', 'score': 0.9300},
        {'word': 'user_a', 'entity_group': 'IDTY', 'score': 0.8800},
        {'word': 'C:\\temp\\mal.exe', 'entity_group': 'FILE', 'score': 0.9600},
    ]
    return mock_results

def extract_pdf_text_robust(pdf_file):
    mock_long_text = "The initial access was confirmed as spear phishing. APT-29 utilized this tactic to deliver the TrickBot malware. The malware subsequently connected to a command and control server at 192.168.1.1. The campaign closely resembles activity associated with LockBit ransomware operations. We found evidence of a vulnerability, CVE-2024-4228, being exploited. Further analysis is required on the QakBot loader used in the secondary stage." * 5
    return mock_long_text

def process_structured_data(file_object):
    start_time = time.time()
    
    try:
        df = pd.read_csv(file_object)
    except Exception as e:
        raise ValueError(f"Error reading structured file: {e}")

    text_columns = [col for col in df.columns if re.search(r'message|description|details|event', col, re.IGNORECASE)]
    
    if not text_columns:
        text_columns = df.columns.tolist() 
        st.warning(f"No descriptive columns found. Analyzing all {len(text_columns)} columns for entities.")

    df['combined_text'] = df[text_columns].astype(str).agg(' '.join, axis=1)
    
    full_text = '\n'.join(df['combined_text'].tolist())
    
    chunks = chunk_text_robust(full_text)
    results = [r for chunk in chunks for r in mock_ner_pipeline(chunk)]
    df_results = pd.DataFrame(results)
    
    df_cleaned = clean_and_normalize_entities(df_results)
    G = build_cti_knowledge_graph_igraph(df_cleaned)
    
    end_time = time.time()
    
    return df_cleaned, G, (end_time - start_time)

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
    name_to_label = dict(zip(df_entities['Entity'], df_entities['Type']))
    
    G = ig.Graph(directed=True)
    G.add_vertices(len(vertex_names))
    G.vs["name"] = vertex_names
    G.vs["node_type"] = [name_to_label[name] for name in G.vs["name"]]
    G.vs["label"] = G.vs["name"]
    
    color_map = {
        'THREAT_ACTOR': '#e31a1c', 'MALWARE': '#33a02c', 'RANSOMWARE': '#a6cee3', 
        'IP': '#fdbf6f', 'VULID': '#ffff99', 'ACT': '#1f78b4', 'DOMAIN': '#b2df8a', 'IDTY': '#ff7f00', 'FILE': '#fb9a99'
    }
    G.vs["color"] = [color_map.get(lab, '#a6cee3') for lab in G.vs["node_type"]]
    
    edges_to_add = []
    edge_relations = []
    
    for i in range(len(vertex_names) - 1):
        e1, l1 = vertex_names[i], name_to_label[vertex_names[i]]
        e2, l2 = vertex_names[i+1], name_to_label[vertex_names[i+1]]
        
        relation = "related_to"
        if l1 in ["THREAT_ACTOR", "RANSOMWARE"] and l2 == "MALWARE": relation = "uses"
        elif l1 in ["MALWARE", "RANSOMWARE"] and l2 in ["IP", "DOMAIN"]: relation = "connects_to"
        elif l1 == "ACT" and l2 in ["MALWARE", "RANSOMWARE"]: relation = "delivers"
        elif l1 in ["MALWARE", "RANSOMWARE"] and l2 == "VULID": relation = "exploits"
        
        try:
            id1 = G.vs.find(name=e1).index
            id2 = G.vs.find(name=e2).index
            edges_to_add.append((id1, id2))
            edge_relations.append(relation)
        except:
            continue
        
    G.add_edges(edges_to_add)
    G.es["label"] = edge_relations
    return G

def run_full_pipeline(pdf_file):
    start_time = time.time()
    
    raw_text = extract_pdf_text_robust(pdf_file)
    chunks = chunk_text_robust(raw_text)
    
    results = [r for chunk in chunks for r in mock_ner_pipeline(chunk)]
        
    df = pd.DataFrame(results)
    
    df_cleaned = clean_and_normalize_entities(df)
    G = build_cti_knowledge_graph_igraph(df_cleaned)
    
    end_time = time.time()
    
    return df_cleaned, G, (end_time - start_time)

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
        
        layout = subgraph.layout("kamada_kawai")
        fig, ax = plt.subplots(figsize=(10, 8))

        ig.plot(subgraph, target=ax, layout=layout,
            vertex_label=subgraph.vs["name"],
            vertex_color=subgraph.vs["color"],
            edge_label=subgraph.es["label"],
            edge_color="gray",
            vertex_size=25,
            bbox=(800, 600))
            
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
