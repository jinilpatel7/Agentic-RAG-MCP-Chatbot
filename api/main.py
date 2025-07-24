# This file defines the main FastAPI application for the Multi-Document RAG API.

import os
import sys
import shutil
from typing import List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from src.agents.ingestion_agent import IngestionAgent
from src.agents.processing import TextProcessing
from src.agents.embedding_agent import EmbeddingAgent
from src.agents.coordinator_agent import CoordinatorAgent
from src.vector_store.chroma_db import ChromaDBHandler
from src.exception import CustomException

# --- Load environment variables from .env file ---
load_dotenv()

# --- Initialize the FastAPI application ---
app = FastAPI(
    title="Multi-Document RAG API",
    description="An API for chatting with multiple documents using a RAG pipeline.",
    version="1.0.0"
)

# --- Setup file and vectorstore directories ---
UPLOAD_DIRECTORY = os.getenv("UPLOAD_DIR", "./data")
PERSIST_DIRECTORY = os.getenv("CHROMA_DIR", "./vectorstore/chroma_db")

# Create necessary directories if they don't exist
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

# --- Initialize core components globally for reuse ---
try:
    # Load the embedding model
    embedding_agent = EmbeddingAgent()

    # Load or create ChromaDB vector store
    vector_store = ChromaDBHandler(persist_directory=PERSIST_DIRECTORY)
    vector_store.create_or_load(embeddings=embedding_agent.embedding_model)

    # Coordinator agent orchestrates retrieval and generation
    coordinator_agent = CoordinatorAgent()
except Exception as e:
    # If something fails during startup, don't allow the app to run
    raise RuntimeError(f"Failed to initialize core components: {e}") from e

# --- Pydantic model to validate query request payload ---
class QueryRequest(BaseModel):
    query: str

# --- Upload and process documents ---
@app.post("/upload-and-process")
async def upload_and_process_files(files: List[UploadFile] = File(...)):
    """
    Upload documents, extract text, chunk them, embed them, and store in vector database.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    saved_file_paths = []        # To keep track of saved files
    all_extracted_text = ""      # To store extracted raw text for response

    # Save uploaded files to disk
    for file in files:
        file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_file_paths.append(file_path)
        finally:
            file.file.close()

    try:
        # Step 1: Extract raw text using the ingestion agent
        ingestion_agent = IngestionAgent()
        extracted_data = ingestion_agent.ingest_files(saved_file_paths)

        # Step 2: Process text into chunked documents
        all_docs = []
        text_processor = TextProcessing(chunk_size=500, chunk_overlap=50)

        for filename, text in extracted_data.items():
            all_extracted_text += f"--- {filename} ---\n{text}\n\n"
            docs = text_processor.process(text, metadata={"source": filename})
            all_docs.extend(docs)

        if not all_docs:
            return JSONResponse(
                status_code=200,
                content={
                    "message": "Files were uploaded, but no text could be extracted or processed.",
                    "extracted_text": all_extracted_text
                }
            )

        # Step 3: Embed and store in vector database
        vector_store.add_documents(all_docs)

        return {
            "message": f"Successfully processed {len(saved_file_paths)} files.",
            "filenames": [os.path.basename(p) for p in saved_file_paths],
            "extracted_text": all_extracted_text
        }

    except Exception as e:
        raise CustomException(e, sys)


# --- Query previously processed documents ---
@app.post("/query")
async def handle_query(request: QueryRequest):
    """
    Submit a question and retrieve an answer based on uploaded documents.
    """
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        # Delegate the query to the CoordinatorAgent, which handles retrieval + LLM response
        result = coordinator_agent.handle_query(query=request.query)

        # Respond with both the generated answer and the source documents
        return {
            "answer": result["payload"]["answer"],
            "sources": result["payload"]["sources"]
        }

    except Exception as e:
        raise CustomException(e, sys)


# --- Clear all stored data (embeddings and uploaded files) ---
@app.post("/clear")
async def clear_data():
    """
    Delete all files and vector embeddings. Use cautiously!
    """
    try:
        # Step 1: Clear vector store collection
        vector_store.clear_collection()

        # Step 2: Delete all uploaded files from disk
        for filename in os.listdir(UPLOAD_DIRECTORY):
            file_path = os.path.join(UPLOAD_DIRECTORY, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

        return {"message": "All data has been cleared successfully."}

    except Exception as e:
        raise CustomException(e, sys)
