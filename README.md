# AI-Powered Cyber Threat Intelligence (CTI) Analysis Tool

This project uses a specialized AI model to automatically analyze Cyber Threat Intelligence (CTI) reports in PDF format. It extracts key entities such as malware names, threat actors, IP addresses, and attack techniques, and then visualizes their relationships as an interactive knowledge graph. This helps security analysts quickly understand complex threat narratives.

---

## Features

-   **Automated PDF Processing**: Ingests unstructured CTI reports directly in PDF format.
-   **AI-Powered Entity Recognition**: Utilizes the `CyberPeace-Institute/SecureBERT-NER` model, a transformer specialized in identifying cybersecurity-related entities.
-   **Knowledge Graph Generation**: Automatically constructs a graph where each entity is a node and their relationships are edges, mapping out the attack chain.
-   **Interactive Web Interface**: A user-friendly web application built with Gradio allows for easy report uploading and interaction without writing any code.
-   **Subgraph Visualization**: Users can select a specific entity (like a malware name) to generate a focused 1-hop visualization of its direct connections and relationships.

---

## How It Works

The application follows a clear, multi-step workflow to process and analyze the reports:

1.  **PDF Upload & Text Extraction**: A user uploads a CTI report through the Gradio web interface. The application uses the `PyPDF2` library to extract the raw text from the document.
2.  **Text Chunking**: Since transformer models have a maximum input length, the extracted text is split into smaller, overlapping chunks to ensure the entire report can be processed without losing context.
3.  **Named Entity Recognition (NER)**: Each chunk is fed into the `SecureBERT-NER` pipeline. The model identifies and tags entities with labels such as `MALWARE`, `THREAT_ACTOR`, `TOOL`, `IP`, and `ACT` (Action).
4.  **Knowledge Graph Construction**: The list of extracted entities is used to build a directed graph using the `igraph` library. Custom rules based on CTI logic (e.g., a `THREAT_ACTOR` *performs* an `ACT`) are used to create meaningful relationships (edges) between the entities (nodes).
5.  **Interactive Visualization**: The application displays a table of all extracted entities and populates a dropdown menu. When a user selects an entity, the backend generates and displays a plot of its subgraph, showing all its immediate connections.

---

## Getting Started

To run the interactive web application on your local machine, follow these steps.

### Prerequisites

-   Python 3.8+
-   Jupyter Notebook or JupyterLab

### Installation

1.  Clone the repository to your local machine.
2.  It is highly recommended to create a virtual environment to manage dependencies.
3.  Install the required libraries using pip. You can create a `requirements.txt` file with the content below and run `pip install -r requirements.txt`.

    ```
    # requirements.txt
    transformers
    torch
    PyPDF2
    python-igraph
    cairocffi
    gradio
    pandas
    matplotlib
    ```

### Running the Application

The primary application is contained in the `Knowledge_graph_gradio_app.ipynb` notebook.

1.  Open the `Knowledge_graph_gradio_app.ipynb` file in Jupyter.
2.  Run all the cells in the notebook.
3.  A public URL will be generated in the output of the last cell. Open this link in your web browser to use the application.

---

## Project Structure

This project includes several notebooks that represent different stages of development and functionality:

-   `Knowledge_graph_gradio_app.ipynb`: **(Main Application)** The final, polished web application with an interactive UI for analyzing PDF reports.
-   `CTI_LLM.ipynb`: **(Core Prototype)** The initial development notebook demonstrating the end-to-end workflow from PDF processing to graph visualization using `networkx`.
-   `NER_Transformer.ipynb`: **(Simplified Extractor)** A basic script focused solely on extracting text and running the NER model. Useful for quick tests.
-   `CTI.ipynb`: **(Experimental)** An alternative approach showing how to train a custom CTI entity recognition model from scratch using the `spaCy` library. This is not used by the main application but serves as a proof-of-concept for custom model training.

---

## Technology Stack

-   **Backend**: Python
-   **AI/ML**: Hugging Face Transformers, PyTorch
-   **Web UI**: Gradio
-   **PDF Processing**: PyPDF2
-   **Graph Analysis & Visualization**: igraph, Matplotlib
-   **Data Handling**: Pandas
