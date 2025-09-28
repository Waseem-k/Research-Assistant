# RAG-based Research Assistant

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project provides a sophisticated Research Assistant that leverages a Retrieval-Augmented Generation (RAG) pipeline to deliver insightful answers from a vast collection of research documents. It features an interactive web interface built with Streamlit, a hybrid retrieval system that queries both a local Qdrant database and the live Arxiv repository, and is powered entirely by local models via Ollama to ensure complete data privacy.

## ✨ Key Features

-   **Hybrid Information Retrieval**: Utilizes an `EnsembleRetriever` to combine results from two powerful sources: a `Qdrant` vector database for fast, semantic search over ingested documents and an `ArxivRetriever` to fetch the latest research directly from Arxiv in real-time.
-   **Local & Private**: Core functionality, including embeddings and text generation, is driven by the `llama3` model running locally via Ollama. This enables offline usage and guarantees that your data remains private.
-   **Automated Data Ingestion**: A dedicated script (`ingestion.py`) processes research papers from a CSV file, generates embeddings, and indexes them into the Qdrant vector store, creating a robust, searchable knowledge base.
-   **RAG Performance Evaluation**: Deeply integrated with TruLens for comprehensive evaluation of the RAG pipeline. It tracks key metrics such as **Answer Relevance**, **Context Relevance**, and **Groundedness** to ensure the quality and factual accuracy of the generated responses.
-   **Interactive Web Interface**: A clean and user-friendly UI built with Streamlit (`rag.py`) allows for intuitive interaction—simply upload your papers and start asking questions.

## ⚙️ Tech Stack & Architecture

The project is built on a modern RAG stack orchestrated with LangChain. The data flows from the user interface through the retrieval and generation pipeline to produce an answer, with continuous evaluation provided by TruLens.

-   **Frontend**: Streamlit
-   **LLM & Embeddings**: Ollama (`llama3` model)
-   **Vector Store**: Qdrant
-   **Retrieval**: LangChain's `EnsembleRetriever` (Qdrant + Arxiv)
-   **Orchestration**: LangChain Expression Language (LCEL)
-   **Evaluation & Observability**: TruLens

---
**Architectural Flow:**
`User Query (Streamlit UI)` -> `LangChain (EnsembleRetriever)` -> `[Qdrant DB | Arxiv API]` -> `Retrieved Context` -> `LLM (Ollama)` -> `Generated Answer` -> `UI`
---

## 🚀 Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

Before you begin, ensure you have the following installed and configured:
-   Python 3.9+
-   **Ollama**: Install [Ollama](https://ollama.com/) and pull the `llama3` model.
    ```
    ollama pull llama3
    ```
-   **Qdrant**: A running instance of [Qdrant](https://qdrant.tech/documentation/guides/installation/). Using Docker is recommended for a quick setup:
    ```
    docker run -p 6333:6333 qdrant/qdrant
    ```

### Installation and Setup

1.  **Clone the Repository**
    ```
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Install Dependencies**
    Create a `requirements.txt` file in the root of the project with the following content:
    ```
    qdrant-client
    jproperties
    ollama
    langchain
    langchain-ollama
    langchain-qdrant
    langchain-community
    numpy
    trulens-eval
    streamlit
    pandas
    nest-asyncio
    arxiv
    litellm
    ```
    Then, install all the required packages using pip:
    ```
    pip install -r requirements.txt
    ```

3.  **Configure the Application**
    Create an `appconfig.properties` file in the root directory. This file holds the configuration for connecting to Qdrant and locating your data file.
    ```
    QDRANT_URL=http://localhost:6333
    QDRANT_API_KEY=your-qdrant-api-key # Optional, if you have one configured
    QDRANT_COLLECTION_NAME=research_papers
    FILE_PATH=path/to/your/papers.csv
    ```

4.  **Ingest Your Data**
    Run the ingestion script to process your research papers and load them into Qdrant. Make sure the CSV file path in `appconfig.properties` is correct.
    ```
    python ingestion.py
    ```

## Usage

Once the setup is complete, you can start the Research Assistant.

### Launch the Web Interface

Launch the Streamlit application to use the interactive UI.

```
streamlit run rag.py
```
Navigate to the local URL provided by Streamlit (usually `http://localhost:8501`) in your browser to start interacting with the assistant.

## 📊 Evaluation with TruLens

This project is instrumented with TruLens to provide deep insights into the RAG pipeline's performance.

-   When you run the Streamlit app, a TruLens dashboard is automatically initiated on a separate port (typically `http://localhost:8502`).
-   The dashboard allows you to inspect each query, the retrieved context from both Qdrant and Arxiv, the generated response, and the scores for feedback functions like **Groundedness**, **Answer Relevance**, and **Context Relevance**.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more information.
