# This Agent is responsible for chunking text into smaller pieces.
import sys
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.exception import CustomException
from src.logger import logging

class TextProcessing:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initializes the text processing utility with a recursive character-based splitter.
        
        Args:
            chunk_size (int): Maximum number of characters per chunk.
            chunk_overlap (int): Number of overlapping characters between chunks to preserve context.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Recursive splitter breaks text into chunks based on logical boundaries (like sentences or paragraphs)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def process(self, text: str, metadata: dict = None) -> List[Document]:
        """
        Splits a large text string into smaller chunks and wraps them as LangChain Document objects.

        Args:
            text (str): The full text to split.
            metadata (dict, optional): Metadata to attach to each chunk (e.g., file name or type).

        Output:
            List[Document]: A list of LangChain Document objects, each containing a chunk of the original text.
        """
        try:
            logging.info("Splitting extracted text into chunks...")

            # Split the input text into smaller, overlapping chunks
            chunks = self.splitter.split_text(text)

            # Wrap each chunk in a LangChain Document, attaching metadata if provided
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
