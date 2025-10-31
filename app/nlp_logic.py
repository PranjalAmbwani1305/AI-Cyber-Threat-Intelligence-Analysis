import pandas as pd
from pyvis.network import Network
import networkx as nx
import tempfile, os

def process_cti_data():
    """
    Placeholder for real CTI NLP logic.
    Replace this with integration to your models or preprocessing pipeline.
    Should return a DataFrame with columns: Entity, Type, Score.
    """
    data = {
        "Entity": ["APT29", "Phishing", "CVE-2025-10001", "Exchange Server", "10.0.0.15"],
        "Type": ["APT", "Attack", "Vulnerability", "Tool", "IP"],
        "Score": [0.98, 0.95, 0.93, 0.89, 0.87]
    }
    df = pd.DataFrame(data)
    return df


def build_cti_graph_pyvis():
    """
    Builds an interactive Cyber Threat Intelligence graph using Pyvis.
    """
    df = process_cti_data()
    if df.empty:
        return None

    G = nx.DiGraph()

    color_map = {
        "Attack": "#1f77b4",
        "Tool": "#2ca02c",
        "APT": "#d62728",
        "Vulnerability": "#ff7f0e",
        "IP": "#17becf",
        "Domain": "#9467bd",
    }

    for _, row in df.iterrows():
        ent, typ = row["Entity"], row["Type"]
        G.add_node(ent, color=color_map.get(typ, "#9fa8da"), title=f"{ent} ({typ})")

    for i in range(len(df) - 1):
        G.add_edge(df.iloc[i]["Entity"], df.iloc[i + 1]["Entity"], title="related_to")

    net = Network(height="650px", bgcolor="#0e1117", font_color="white", directed=True)
    net.from_nx(G)
    net.repulsion(node_distance=180, spring_length=200)

    temp_path = os.path.join(tempfile.gettempdir(), "cti_graph.html")
    net.save_graph(temp_path)

    with open(temp_path, "r", encoding="utf-8") as f:
        html = f.read()
    return html
