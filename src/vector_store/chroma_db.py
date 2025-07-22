import sys
import os
from typing import List
from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.exception import CustomException
from src.logger import logging


class ChromaDBHandler:
    def __init__(self, persist_directory: str):
        try:
            self.persist_directory = persist_directory
            self.db = None
            # Create directory if it doesn't exist
            os.makedirs(persist_directory, exist_ok=True)
            logging.info(f"Initializing Chroma vectorstore at: {persist_directory}")
        except Exception as e:
            raise CustomException(e, sys)

    def create_or_load(self, embeddings):
        try:
            self.db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=embeddings,
                collection_name="rag_collection"
            )
            logging.info("Chroma vectorstore initialized successfully.")
        except Exception as e:
            raise CustomException(e, sys)

    def add_documents(self, documents: List[Document]):
        try:
            if not self.db:
                raise Exception("Chroma DB not initialized. Call create_or_load first.")
            
            # Add unique IDs to prevent duplicates
            for i, doc in enumerate(documents):
                doc.metadata["doc_id"] = f"{doc.metadata.get('source', 'unknown')}_{i}"
            
            logging.info(f"Adding {len(documents)} document chunks to Chroma...")
            self.db.add_documents(documents)
            logging.info("Documents added to Chroma DB successfully.")
        except Exception as e:
            raise CustomException(e, sys)

    def similarity_search(self, query: str, k: int = 5):
        try:
            if not self.db:
                raise Exception("Chroma DB not initialized. Call create_or_load first.")
            
            logging.info(f"Searching for: {query}")
            results = self.db.similarity_search(query, k=k)
            
            # Log sources found
            sources = set([doc.metadata.get("source", "unknown") for doc in results])
            logging.info(f"Found results from sources: {sources}")
            
            return results
        except Exception as e:
            raise CustomException(e, sys)

    def clear_collection(self):
        """Clear all documents from the collection"""
        try:
            if self.db:
                # Get all document IDs and delete them
                collection = self.db._collection
                all_ids = collection.get()["ids"]
                if all_ids:
                    collection.delete(ids=all_ids)
                    logging.info(f"Cleared {len(all_ids)} documents from collection")
        except Exception as e:
            logging.error(f"Error clearing collection: {str(e)}")