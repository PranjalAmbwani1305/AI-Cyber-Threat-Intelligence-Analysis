# Cyber Threat Intelligence Dashboard

### Overview
This Streamlit application provides a unified dashboard for analyzing cybersecurity data from heterogeneous sources such as **firewall logs**, **EDR events**, **SIEM alerts**, and **threat intelligence feeds**.  
It integrates multiple analytical techniques (NER, Knowledge Graph, Clustering, Topic Modeling, Sentiment, CTI classification) into one web interface.

---

## 🚀 Features
| Module | Description |
|---------|--------------|
| **Named Entity Recognition (NER)** | Extracts IP addresses, domains, CVEs, and usernames from log data or free text. |
| **Knowledge Graph** | Builds an interactive graph of relationships between indicators — e.g., `Source_IP → Destination_IP`, `Username → Source_IP`. |
| **Sentence Clustering** | Groups similar log messages or descriptions using embeddings and DBSCAN/KMeans. |
| **Topic Modeling** | Summarizes high-volume textual logs into coherent topics using BERTopic or TF-IDF fallback. |
| **Sentiment Analysis** | Measures positive/negative polarity of reports to indicate threat criticality. |
| **CTI Classification** | Classifies text using CTI keyword mapping (Phishing, Ransomware, Exploit, Vulnerability, etc.). |

---

## ⚙️ How It Works
1. Upload any CSV file (`.csv`) — examples:  
   - `firewall_logs_complete_1.csv`  
   - `siem_alerts_sparse_2.csv`  
   - `edr_events_variant_3.csv`  
   - `threat_feed_indicators_4.csv`

2. The app **automatically detects the best text column** or lets you manually select one.

3. The system combines structured columns (Source_IP, Destination_IP, Username, etc.) to build meaningful context even for non-textual logs.

4. Select an analysis task from the sidebar and click **“Run Analysis”**.

---

## 🧠 Tech Stack
- **Streamlit** for the web UI  
- **pandas / numpy** for preprocessing  
- **transformers** (optional) for advanced NER/Sentiment  
- **sentence-transformers / scikit-learn** for embeddings & clustering  
- **BERTopic** (optional) for topic modeling  
- **networkx / matplotlib** for graph visualization  

All heavy libraries are optional — the app will gracefully fall back to lightweight regex and TF-IDF methods if unavailable.

---

## 🖥️ Deployment
Deployed Streamlit app:  
👉 [ai-cyber-threat-intelligence-analysis.streamlit.app](https://ai-cyber-threat-intelligence-analysis.streamlit.app/)

To run locally:

```bash
pip install -r requirements.txt
streamlit run cti_dashboard_auto_column.py
