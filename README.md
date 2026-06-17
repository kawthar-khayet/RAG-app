# RAG-app

A comprehensive Retrieval-Augmented Generation (RAG) application combining document retrieval with large language models for intelligent question answering and information extraction.

##  Project Overview

This project implements a complete RAG pipeline that:
- **Indexes Documents** - Efficiently stores and retrieves documents
- **Semantic Search** - Uses embeddings for intelligent document retrieval
- **LLM Integration** - Combines retrieved context with LLMs for accurate answers
- **Multi-format Support** - Handles PDFs, text files, web content
- **Web Interface** - User-friendly Streamlit/Gradio interface

## Features

 **Multi-Document Support**
- PDF documents
- Text files
- Web URLs
- Markdown files

 **Advanced Retrieval**
- Semantic similarity search
- Hybrid search (BM25 + embeddings)
- Metadata filtering
- Reranking

 **LLM Integration**
- OpenAI GPT-3.5/GPT-4
- Llama 2
- Local models via Ollama
- Custom LLM support

 **Web Interface**
- Streamlit dashboard
- Chat interface
- Document management
- Query history

## Project Structure
 
```
RAG-app/
├── app.py                  # Main Streamlit application entry point
├── config.py               # Centralized configuration (models, paths, parameters)
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variables template
│
├── src/
│   ├── embeddings.py       # Embedding model wrappers
│   ├── vectorstore.py      # Vector database setup and management
│   ├── ingest.py        #
│   └── utils.py            # Logging, helpers, shared utilities
│
├── data/
│   ├── documents/          # Raw input documents (place your files here)
│   ├── processed/          # Chunked and cleaned document objects
│   └── vectorstore/        # Persisted FAISS / Chroma index
│
├── models/
│   ├── embeddings/         # Local embedding model weights
│   └── llm_models/         # Local LLM weights (GGUF, etc.)
│
├── notebooks/
│   ├── 01_setup.ipynb          # Environment setup and sanity checks
│   ├── 02_rag_pipeline.ipynb   # End-to-end pipeline walkthrough
│   └── 03_evaluation.ipynb     # Retrieval and generation evaluation
│
├── tests/
│   ├── test_retriever.py       # Unit tests for retrieval logic
│   ├── test_chain.py           # Unit tests for RAG chain
│   └── test_integration.py     # End-to-end integration tests
│
├── logs/                   # Application and error logs
└── results/
    ├── metrics.json        # Evaluation scores (precision, recall, RAGAS)
    └── query_logs.json     # Full query and response history

```

---
## Installation
 
**Prerequisites:** Python 3.10+, pip, (optional) Node.js for local LLM tooling.
 
```bash
# 1. Clone the repository
git clone https://github.com/your-username/RAG-app.git
cd RAG-app
 
# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
 
# 3. Install dependencies
pip install -r requirements.txt
 
# 4. Set up environment variables
cp .env.example .env
# Edit .env and fill in your API keys
```
 
---
---
 ## Usage
 
### 1. Ingest documents
 
Place your documents in `data/documents/`, then run:
 
```bash
python -c "from src.preprocessing import ingest; ingest()"
```
 
This loads, splits, embeds, and saves the vector index to `data/vectorstore/`.
---
## Running Tests
 
```bash
# Run all tests
pytest tests/
 
# Run a specific test file
pytest tests/test_retriever.py -v
 
# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```
 
---
