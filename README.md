# AI-Powered Cyber Threat Intelligence (CTI) Analysis Tool

This project is a multi-feature web application built with Python and Gradio to help cybersecurity analysts analyze Cyber Threat Intelligence (CTI) reports.

Instead of manually reading long PDF reports, this tool allows an analyst to upload a PDF and instantly get a high-level, interactive summary. It extracts key entities, visualizes their relationships, clusters sentences into semantic topics, summarizes document sentiment, and discovers underlying themes.

![CTI Analysis Tool Screenshot](https://i.imgur.com/83p1X2O.png)

## ✨ Features

This application is organized into several tabs, each performing a different analysis:

1.  **Knowledge Graph Analyzer:**
    * Upload a PDF report (e.g., from a threat intel provider).
    * Uses a `SecureBERT-NER` model to automatically extract CTI entities (like `APT`, `MALWARE`, `CVE`, `IP`, `HASH`, etc.).
    * Builds an interactive knowledge graph (`igraph`) showing the relationships between these entities.
    * Allows you to select any entity from a dropdown to see a 1-hop graph of its immediate neighbors.

2.  **Document Summary:**
    * **CTI Keyword Summary:** Scans all cleaned sentences for a custom list of keywords (e.g., "phishing," "ransomware," "cve-") and provides a summary table of what was found and where.
    * **Sentiment Analysis:** Runs batch sentiment analysis on all sentences and displays the Top 5 most positive and negative sentences from the document.

3.  **Semantic Topic Clustering (DBSCAN):**
    * Cleans and filters all sentences from the PDF to remove noise (e.g., headers, footers, page numbers).
    * Displays the full list of cleaned sentences used for the analysis (with a scrollbar).
    * Uses `SentenceTransformer` embeddings and `DBSCAN` to group all sentences into semantic clusters.
    * Visualizes the clusters on a 2D plot (using `PCA` for dimensionality reduction).
    * Provides a dropdown to select any cluster and view all sentences within it, allowing for deep exploration.

4.  **Topic Modeling (BERTopic):**
    * Provides an advanced alternative to clustering.
    * Runs `BERTopic` on the cleaned PDF sentences to discover and visualize the main themes and topics discussed in the report.

5.  **Linguistic Analysis (spaCy):**
    * A utility tab to analyze the grammatical structure of any single sentence.
    * This is useful for examining a specific, complex sentence found in the clustering or summary tabs.
    * It provides a Part-of-Speech (POS) table and a high-contrast, scrollable Dependency Plot.

---

## 🛠️ Technology Stack

| Category | Technology / Library | Purpose |
| :--- | :--- | :--- |
| **Primary Language** | Python 3.8+ | Backend logic, data processing, and AI workflows. |
| **Web Interface** | **Gradio** | Creates the interactive, shareable web application UI. |
| **AI/ML Core** | **Hugging Face Transformers**, **PyTorch** | Hosts and runs the `SecureBERT-NER` and sentiment models. |
| **NLP Models** | **Sentence-Transformers** | Generates embeddings for semantic clustering. |
| **NLP Models** | **BERTopic** | Powers the advanced Topic Modeling tab. |
| **NLP Models** | **spaCy** | Powers the Linguistic Analysis tab (POS, Dependency). |
| **Text Handling** | **PyPDF2** | Extracts raw text from PDF documents. |
| **Text Handling** | **NLTK** | Used for robust sentence splitting and cleaning. |
| **Data Analysis** | **Pandas** | Manages and displays data in tables. |
| **Data Analysis** | **scikit-learn** | Used for DBSCAN clustering, PCA, and TfidfVectorizer. |
| **Graph/Viz** | **python-igraph**, **cairocffi** | Constructs and renders the knowledge graph structure. |
| **Graph/Viz** | **Matplotlib** | Used for displaying the interactive subgraphs. |

---

## ⚙️ How It Works (Workflow)

The application follows two parallel data pipelines after a user uploads a PDF:

1.  **PDF Upload & Text Extraction**: The user uploads a report via the Gradio UI. The system extracts the raw text using `PyPDF2`.

2.  **Pipeline A: Knowledge Graph (NER)**
    * The raw text is split into overlapping `chunks` to fit the NER model's context window.
    * `SecureBERT-NER` processes these chunks to find and tag CTI-specific entities.
    * An `igraph` knowledge graph is built from these entities, with edges defined by CTI-logic rules.
    * This pipeline populates the **"Knowledge Graph Analyzer"** tab.

3.  **Pipeline B: Document Analysis (Sentences)**
    * The raw text is *also* passed through a `clean_and_split_sentences` function (using `NLTK` and regex) to create a high-quality list of clean sentences.
    * This "clean sentence list" is stored and used as the common input for multiple analysis tabs.

4.  **On-Demand Analysis**:
    * The clean sentence list is used to power the **"Document Summary"**, **"Semantic Topic Clustering"**, and **"Topic Modeling"** tabs when the user clicks the respective buttons.
    * The **"Linguistic Analysis"** tab runs on-demand on any text the user types in.

---

## 💻 Getting Started

### Prerequisites

* Python 3.8 or higher
* Jupyter Notebook or JupyterLab

### Running the Application

The entire application is contained in the `integrated-final_app.ipynb` notebook.

1.  **Clone the Repository:**
    ```bash
    git clone [https://your-repo-url.git](https://your-repo-url.git)
    cd your-project-directory
    ```

2.  **Create and Activate a Virtual Environment:**
    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source vNenv/bin/activate
    
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install gradio transformers sentence-transformers torch PyPDF2 python-igraph spacy bertopic scikit-learn pandas matplotlib
    ```

4.  **Download the spaCy Model:**
    ```bash
    python -m spacy download en_core_web_sm
    ```

5.  **Run the Application:**
    * Open the **`integrated-final_app.ipynb`** file in Jupyter or VS Code.
    * Run all the cells in the notebook from top to bottom.
    * The final cell will load all the models and print a local URL. Open this link in your web browser to use the application.

    ```
    Running on local URL:  [http://127.0.0.1:7860](http://127.0.0.1:7860)
    ```

---

### 🧑‍💻 Authors
* **Pranjal Ambwani**
* **Dhruv Jain** — [GitHub](https://github.com/DhruvJain7)
* **Amruta Poojary** — [GitHub](https://github.com/Amruta08)
* **Aaseem Mhaskar** — [GitHub](https://github.com/aaseem22)
* **Arib Qureshi** — [GitHub](https://github.com/AribQureshi)