import sys
from langchain_huggingface import HuggingFaceEmbeddings
from src.vector_store.chroma_db import ChromaDBHandler
from src.exception import CustomException
from src.logger import logging


class EmbeddingAgent:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            logging.info(f"Loading HuggingFace embedding model: {model_name}")
            self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)
        except Exception as e:
            raise CustomException("embedding_agent.py", 13, str(e))

    def embed_and_store(self, documents, persist_directory: str):
        try:
            logging.info("Embedding documents and storing to Chroma DB...")
            vector_store = ChromaDBHandler(persist_directory)
            vector_store.create_or_load(self.embedding_model)
            vector_store.add_documents(documents)
        except Exception as e:
            raise CustomException("embedding_agent.py", 25, str(e))
