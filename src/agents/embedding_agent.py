# This Agent is responsible for embedding documents and storing them in a vector database.

import sys
from langchain_huggingface import HuggingFaceEmbeddings
from src.vector_store.chroma_db import ChromaDBHandler
from src.exception import CustomException
from src.logger import logging

class EmbeddingAgent:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initializes the embedding agent using a HuggingFace sentence transformer model.

        Args:
            model_name (str): Name of the embedding model to load from HuggingFace.
        """
        try:
            logging.info(f"Loading HuggingFace embedding model: {model_name}")
            
            # Load the embedding model (e.g., MiniLM) from HuggingFace
            self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)
        
        except Exception as e:
            # Raise a CustomException if model loading fails
            raise CustomException(e, sys)

    def embed_and_store(self, documents, persist_directory: str):
        """
        Embeds the provided documents and stores them in a Chroma vector database.

        Args:
            documents (List[Document]): List of LangChain Document objects to embed.
            persist_directory (str): Directory path where ChromaDB should persist the vectors.
        """
        try:
            logging.info("Embedding documents and storing to Chroma DB...")

            # Initialize or load an existing Chroma vector store
            vector_store = ChromaDBHandler(persist_directory)

            # Create ChromaDB collection (if not exists) using the loaded embedding model
            vector_store.create_or_load(self.embedding_model)

            # Add embedded documents to the vector store for future retrieval
            vector_store.add_documents(documents)

        except Exception as e:
            logging.error("Failed to embed and store documents in ChromaDB.")
            raise CustomException(e, sys)
