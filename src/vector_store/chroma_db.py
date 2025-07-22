import sys
from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.exception import CustomException
from src.logger import logging


class ChromaDBHandler:
    def __init__(self, persist_directory: str):
        try:
            self.persist_directory = persist_directory
            self.db = None
            logging.info(f"Initializing Chroma vectorstore at: {persist_directory}")
        except Exception as e:
            raise CustomException("chroma_db.py", 12, str(e))

    def create_or_load(self, embeddings):
        try:
            self.db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=embeddings,
            )
            logging.info("Chroma vectorstore initialized successfully.")
        except Exception as e:
            raise CustomException("chroma_db.py", 19, str(e))

    def add_documents(self, documents: list[Document]):
        try:
            if not self.db:
                raise Exception("Chroma DB not initialized. Call create_or_load first.")
            logging.info(f"Adding {len(documents)} document chunks to Chroma...")
            self.db.add_documents(documents)
            # self.db.persist()  <-- REMOVE THIS LINE
            logging.info("Documents added to Chroma DB successfully.")
        except Exception as e:
            raise CustomException("chroma_db.py", 28, str(e))
