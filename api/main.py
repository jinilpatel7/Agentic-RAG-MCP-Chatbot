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

# Load environment variables from .env file
load_dotenv()

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Multi-Document RAG API",
    description="An API for chatting with multiple documents using a RAG pipeline.",
    version="1.0.0"
)

# --- Configuration ---
UPLOAD_DIRECTORY = os.getenv("UPLOAD_DIR", "./data")
PERSIST_DIRECTORY = os.getenv("CHROMA_DIR", "./vectorstore/chroma_db")

# Create directories if they don't exist
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

try:
    embedding_agent = EmbeddingAgent()
    vector_store = ChromaDBHandler(persist_directory=PERSIST_DIRECTORY)
    vector_store.create_or_load(embeddings=embedding_agent.embedding_model)
    
    # CoordinatorAgent will be used for querying
    coordinator_agent = CoordinatorAgent()
except Exception as e:
    # If basic setup fails, the app shouldn't start.
    raise RuntimeError(f"Failed to initialize core components: {e}") from e

# --- Pydantic Models for Request/Response ---
class QueryRequest(BaseModel):
    query: str

# --- API Endpoints ---

@app.post("/upload-and-process")
async def upload_and_process_files(files: List[UploadFile] = File(...)):
    """
    Endpoint to upload files, extract text, process, and store embeddings.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
        
    saved_file_paths = []
    all_extracted_text = ""

    for file in files:
        file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_file_paths.append(file_path)
        finally:
            file.file.close()
    
    try:
        # 1. Ingestion Agent: Extract text from files
        ingestion_agent = IngestionAgent()
        extracted_data = ingestion_agent.ingest_files(saved_file_paths)
        
        all_docs = []
        text_processor = TextProcessing(chunk_size=500, chunk_overlap=50)

        # 2. Processing Agent: Chunk text into documents
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

        # 3. Embedding Agent: Embed and store documents in ChromaDB
        # We use the globally initialized agent and vector store
        vector_store.add_documents(all_docs)

        return {
            "message": f"Successfully processed {len(saved_file_paths)} files.",
            "filenames": [os.path.basename(p) for p in saved_file_paths],
            "extracted_text": all_extracted_text
        }
    except Exception as e:
        raise CustomException(e, sys)


@app.post("/query")
async def handle_query(request: QueryRequest):
    """
    Endpoint to handle a user's query against the processed documents.
    """
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
    try:
        # Use the globally initialized CoordinatorAgent
        result = coordinator_agent.handle_query(query=request.query)
        
        # THE FIX IS HERE:
        # We now directly use the 'sources' key from the coordinator's payload
        return {
            "answer": result["payload"]["answer"],
            "sources": result["payload"]["sources"]
        }
    except Exception as e:
        raise CustomException(e, sys)
    
@app.post("/clear")
async def clear_data():
    """
    Endpoint to clear the vector store and delete all uploaded files.
    """
    try:
        # Clear ChromaDB collection
        vector_store.clear_collection()
        
        # Delete files from the data directory
        for filename in os.listdir(UPLOAD_DIRECTORY):
            file_path = os.path.join(UPLOAD_DIRECTORY, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                
        return {"message": "All data has been cleared successfully."}
    except Exception as e:
        raise CustomException(e, sys)