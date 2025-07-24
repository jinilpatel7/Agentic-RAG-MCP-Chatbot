# Agentic-RAG-MCP-Chatbot
An agent-based RAG chatbot that uses Langchain, ChromaDB, and FastAPI with Model Context Protocol (MCP) messaging. Supports PDF, DOCX, CSV, PPTX, and TXT/Markdown documents, and features a Streamlit UI for multi-turn QA with source context.

An advanced, agent-based RAG (Retrieval-Augmented Generation) chatbot that can intelligently ingest and chat with multiple documents. This system uses a team of specialized AI agents that communicate using a Model Context Protocol (MCP)-like messaging system to provide accurate, source-grounded answers.

---

## Features

- **Multi-Document Support**: Upload and chat with PDF, DOCX, CSV, PPTX, and TXT/Markdown files simultaneously.
- **Agentic Architecture**: Specialized agents for ingestion, retrieval, and response generation making system modular and scalable.
- **Retrieval-Augmented Generation (RAG)**: Retrieves relevant document snippets to reduce hallucinations and improve response accuracy.
- **Source Citations**: Responses include document source names for transparency.
- **Interactive UI**: Built with Streamlit for intuitive multi-turn chat and file uploads.
- **Robust Backend**: FastAPI powers a stable backend API.
- **Clear & Reset**: Start new sessions by clearing chat history and vector store.

---

## Project Architecture

This system uses a divide-and-conquer strategy with a backend API coordinating a team of agents that serve a specific purpose.

### Workflows

- **Ingestion Workflow**: For learning from documents
- **Query Workflow**: For responding to user queries using retrieved context

### Architecture Diagram

To view the full architecture visually, open the following file:

> [`Architecture.html`](./Architecture.html)

---

## How It Works: The Agentic Flow

### Ingestion Flow

1. User uploads files via the **Streamlit UI**.
2. The **IngestionAgent** is triggered.
3. The **TextExtractor** reads the content from supported formats.
4. The **TextProcessor** chunks long text into smaller parts.
5. The **EmbeddingAgent** converts chunks into dense vector embeddings.
6. The **ChromaDB Vector Store** stores these vectors with metadata.

### Query Flow (Answering Questions)

1. User inputs a question.
2. The **CoordinatorAgent** receives the query.
3. It delegates to the **RetrievalAgent** to find matching text chunks.
4. The **RetrievalAgent**:
   - Sends query to the **EmbeddingAgent** to get query embedding
   - Queries **ChromaDB** with that embedding
   - Gets **Top-K** matching chunks
5. The **CoordinatorAgent** sends query + retrieved context to the **LLMResponseAgent**.
6. The **LLMResponseAgent** crafts a prompt and queries **Mistral via OpenRouter**.
7. Response is returned through the agents to the UI.

---

## File & Directory Breakdown

| Path                           | Purpose                                                                   |
| ------------------------------ | ------------------------------------------------------------------------- |
| `api/main.py`                  | FastAPI backend with endpoints: `/upload-and-process`, `/query`, `/clear` |
| `ui/app.py`                    | Streamlit frontend for UI, chat, and file upload                          |
| `src/`                         | Core logic directory                                                      |
| ┣ `agents/`                    | Specialized AI agents                                                     |
| ┃ ┣ `ingestion_agent.py`       | Handles file intake and text extraction coordination                      |
| ┃ ┣ `textextraction.py`        | Reads and extracts text from .pdf, .docx, .csv, etc.                      |
| ┃ ┣ `processing.py`            | Chunks text for effective embedding                                       |
| ┃ ┣ `embedding_agent.py`       | Generates vector embeddings from chunks                                   |
| ┃ ┣ `retrieval_agent.py`       | Searches vector DB for relevant content                                   |
| ┃ ┣ `llm_response_agent.py`    | Formats query + context for LLM response                                  |
| ┃ ┗ `coordinator_agent.py`     | Orchestrates agent communication                                          |
| ┣ `vector_store/chroma_db.py`  | Interfaces with ChromaDB for vector storage/search                        |
| ┣ `mcp/mcp_like_msg.py`        | Structured message passing between agents                                 |
| ┣ `logger.py` / `exception.py` | Logging and custom exception handling, Making debugging easier            |
| `data/`                        | Temporary upload directory for raw documents                              |
| `vectorstore/`                 | Directory where ChromaDB data is stored                                   |
| `.env`                         | API keys and environment config                                           |
| `requirements.txt`             | Python dependency list                                                    |
| `setup.py`                     | Makes project pip-installable                                             |

---

## Setup and Installation

### 1. Prerequisites

- Python 3.9 or higher
- [OpenRouter API Key](https://openrouter.ai/keys)

### 2. Clone the Repository

```bash
git clone https://github.com/jinilpatel7/Agentic-RAG-MCP-Chatbot.git
cd Agentic-RAG-MCP-Chatbot
```

### 3. Set Up Virtual Environment

```bash
# Windows
python -m venv myenv
myenv\Scripts\activate
```
### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

Create a `.env` file in the root:

```env
# .env
MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
CHROMA_DIR=./vectorstore/chroma_db
UPLOAD_DIR=./data
OPENROUTER_API_KEY="sk-or-v1-..."
MISTRAL_MODEL_NAME="mistralai/mistral-7b-instruct"
```

---

## How to Run the Application

### Terminal 1: Start the FastAPI Backend

```bash
uvicorn api.main:app --reload
```

### Terminal 2: Start the Streamlit Frontend

```bash
streamlit run ui/app.py
```

- This opens the chat interface in your browser.

---

## Usage Guide

- **Upload Documents** via sidebar.
- Click **"Process"** to ingest documents.
- Ask **questions** in chat.
- Click **"Clear All Data & Chat"** to reset.

---

- Built with: LangChain, ChromaDB, Streamlit, FastAPI, OpenRouter, HuggingFace Embeddings
- Author: Jinil Patel

---
