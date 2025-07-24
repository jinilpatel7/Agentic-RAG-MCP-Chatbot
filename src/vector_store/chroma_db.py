# This file defines a handler for interacting with the Chroma vector database.

import sys
import os
from typing import List
from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.exception import CustomException
from src.logger import logging


class ChromaDBHandler:
    """
    Handler class to manage interaction with the Chroma vector database.
    This includes creating/loading the DB, adding documents, performing similarity searches, and clearing the DB.
    """

    def __init__(self, persist_directory: str):
        """
        Initializes the handler and sets up the directory to persist the Chroma DB.

        Args:
            persist_directory (str): Path to save and reload vectorstore data.
        """
        try:
            self.persist_directory = persist_directory
            self.db = None
            os.makedirs(persist_directory, exist_ok=True)  # Ensure directory exists
            logging.info(f"Initializing Chroma vectorstore at: {persist_directory}")
        except Exception as e:
            raise CustomException(e, sys)

    def create_or_load(self, embeddings):
        """
        Creates a new or loads an existing Chroma vectorstore using the given embedding model.

        Args:
            embeddings: The embedding function/model used for storing and retrieving vectors.
        """
        try:
            self.db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=embeddings,
                collection_name="rag_collection"  # Use a consistent collection name
            )
            logging.info("Chroma vectorstore initialized successfully.")
        except Exception as e:
            raise CustomException(e, sys)

    def add_documents(self, documents: List[Document]):
        """
        Adds a list of LangChain Document chunks to the Chroma DB.

        Args:
            documents (List[Document]): The documents (with metadata) to be stored.
        """
        try:
            if not self.db:
                raise Exception("Chroma DB not initialized. Call create_or_load first.")
            
            # Assign unique doc_id using source and index to help prevent duplicates
            for i, doc in enumerate(documents):
                doc.metadata["doc_id"] = f"{doc.metadata.get('source', 'unknown')}_{i}"
            
            logging.info(f"Adding {len(documents)} document chunks to Chroma...")
            self.db.add_documents(documents)
            logging.info("Documents added to Chroma DB successfully.")
        except Exception as e:
            raise CustomException(e, sys)

    def similarity_search(self, query: str, k: int = 5):
        """
        Performs a vector-based similarity search in the Chroma DB.

        Args:
            query (str): The user query to search for relevant documents.
            k (int): Number of top results to return.

        Output:
            List[Document]: List of top-k documents relevant to the query.
        """
        try:
            if not self.db:
                raise Exception("Chroma DB not initialized. Call create_or_load first.")
            
            logging.info(f"Searching for: {query}")
            results = self.db.similarity_search(query, k=k)

            # Log where results came from
            sources = set([doc.metadata.get("source", "unknown") for doc in results])
            logging.info(f"Found results from sources: {sources}")
            
            return results
        except Exception as e:
            raise CustomException(e, sys)

    def clear_collection(self):
        """
        Clears all documents from the Chroma collection.
        Useful for testing or re-ingesting new data.
        """
        try:
            if self.db:
                # Access the low-level collection object
                collection = self.db._collection
                all_ids = collection.get()["ids"]
                if all_ids:
                    collection.delete(ids=all_ids)
                    logging.info(f"Cleared {len(all_ids)} documents from collection")
        except Exception as e:
            logging.error(f"Error clearing collection: {str(e)}")
