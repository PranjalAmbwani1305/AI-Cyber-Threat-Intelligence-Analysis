# 🛡️ AI-Powered Cyber Threat Intelligence (CTI) Analysis Tool

**LIVE APP:** [https://ai-cyber-threat-intelligence-analysis.streamlit.app/](https://ai-cyber-threat-intelligence-analysis.streamlit.app/)

---

This project uses a specialized **AI model** to automatically analyze **Cyber Threat Intelligence (CTI) reports** in **PDF format**. It extracts key entities such as malware names, threat actors, IP addresses, and attack techniques, and then visualizes their relationships as an interactive **Knowledge Graph**. This helps security analysts quickly understand complex threat narratives.

---

## ✨ Features

* **Automated PDF Processing**: Ingests unstructured CTI reports directly in PDF format using **PyPDF2**.
* **AI-Powered Entity Recognition**: Utilizes the state-of-the-art **CyberPeace-Institute/SecureBERT-NER** transformer model, specialized in identifying cybersecurity-related entities.
* **Knowledge Graph Generation**: Automatically constructs a graph using **igraph** where entities are nodes and CTI-logic rules define the relationships (edges), mapping out the attack chain.
* **Interactive Web Interface**: A user-friendly, no-code application built with **Gradio** allows for easy report uploading and interaction.
* **Subgraph Visualization**: Users can select a specific entity (like a malware name) to instantly generate a focused 1-hop visualization of its direct connections and relationships.

---

## 🛠️ Technology Stack

| Category | Technology / Library | Purpose |
| :--- | :--- | :--- |
| **Primary Language** | Python 3.8+ | Backend logic, data processing, and AI workflows. |
| **Web Interface** | **Gradio** | Creates the interactive, shareable web application UI. |
| **AI/ML Core** | **Hugging Face Transformers**, **PyTorch** | Hosts and runs the **SecureBERT-NER** model pipeline. |
| **PDF Handling** | **PyPDF2** | Extracts raw text from PDF documents. |
| **Graph Analysis** | **python-igraph**, **cairocffi** | Constructs the knowledge graph structure and handles graph layout/rendering. |
| **Data Handling** | **Pandas** | Manages and displays extracted entities in tabular form. |
| **Visualization** | **Matplotlib** | Used for general plotting and displaying the interactive subgraphs. |

---

## ⚙️ How It Works (Workflow)

The application follows a clear, multi-step workflow to process and analyze the reports:

1.  **PDF Upload & Text Extraction**: The user uploads a report via the Gradio UI. The system extracts the raw text using `PyPDF2`.
2.  **Text Chunking**: The extracted text is split into smaller, overlapping chunks to ensure the entire report can be processed by the Transformer model.
3.  **Named Entity Recognition (NER)**: Each chunk is fed into the `SecureBERT-NER` pipeline. The model identifies and tags entities (e.g., `MALWARE`, `THREAT_ACTOR`, `IP`).
4.  **Knowledge Graph Construction**: The extracted entities are used to build a directed graph using the `igraph` library. Custom rules are applied to create meaningful relationships (edges) between the entities (nodes).
5.  **Interactive Visualization**: The application displays extracted entities and generates a plot of the subgraph based on the user's entity selection.

---

## 💻 Getting Started

### Prerequisites

* Python 3.8 or higher
* Jupyter Notebook or JupyterLab

### Installation

1.  Clone the repository to your local machine.
    ```bash
    git clone [your-repository-link]
    cd ai-cti-analysis
    ```
2.  (Recommended) Create and activate a virtual environment.
3.  Install the required libraries:

    ```bash
    # Create requirements.txt (if necessary) with content:
    # transformers, torch, PyPDF2, python-igraph, cairocffi, gradio, pandas, matplotlib

    pip install -r requirements.txt
    ```

### Running the Application

The primary application is contained in the `Knowledge_graph_gradio_app.ipynb` notebook.

1.  Open the **`Knowledge_graph_gradio_app.ipynb`** file in Jupyter.
2.  Run all the cells in the notebook.
3.  A public URL will be generated in the output of the last cell. Open this link in your web browser to use the application.

---

## 📂 Project Structure (Development Notebooks)

This project includes several notebooks that represent different stages of development and functionality:

* **`Knowledge_graph_gradio_app.ipynb`**: **(Main Application)** The final, polished web application with an interactive Gradio UI.
* **`CTI_LLM.ipynb`**: **(Core Prototype)** The initial development notebook demonstrating the end-to-end workflow from PDF processing to graph visualization.
* **`NER_Transformer.ipynb`**: **(Simplified Extractor)** A basic script focused solely on extracting text and running the NER model.
* **`CTI.ipynb`**: **(Experimental)** An alternative approach showing how to train a custom CTI entity recognition model using the `spaCy` library (not used by the main application).

---

### 🧑‍💻 Authors
* **Pranjal Ambwani**
* **Dhruv Jain** — [GitHub](https://github.com/DhruvJain7)
* **Amruta Poojary** — [GitHub](https://github.com/Amruta08)
* **Aaseem Mhaskar** — [GitHub](https://github.com/aaseem22)
* **Arib Qureshi** — [GitHub](https://github.com/AribQureshi)
