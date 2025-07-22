import sys
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.exception import CustomException
from src.logger import logging


class TextProcessing:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def process(self, text: str, metadata: dict = None) -> List[Document]:
        """
        Splits text into chunks and returns list of LangChain Document objects with metadata.
        """
        try:
            logging.info("Splitting extracted text into chunks...")

            chunks = self.splitter.split_text(text)

            docs = [
                Document(
                    page_content=chunk,
                    metadata=metadata if metadata else {}
                )
                for chunk in chunks
            ]

            logging.info(f"Text split into {len(docs)} chunks.")
            return docs

        except Exception as e:
            logging.error("Failed to process text into chunks.")
            raise CustomException(e, sys)
