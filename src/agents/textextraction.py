# agents/textextraction.py

import os
import sys
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
    TextLoader
)

from src.exception import CustomException
from src.logger import logging

from dataclasses import dataclass
@dataclass
class TextExtractor:
    def __init__(self):
        logging.info("Initializing TextExtractor with supported file types...")
        self.supported_loaders = {
            ".pdf": PyPDFLoader,
            ".docx": UnstructuredWordDocumentLoader,
            ".pptx": UnstructuredPowerPointLoader,
            ".csv": CSVLoader,
            ".md": UnstructuredMarkdownLoader,
            ".markdown": UnstructuredMarkdownLoader,
            ".txt": TextLoader
        }

    def extract(self, file_path: str) -> str:
        """
        Extracts text based on file extension using appropriate loader.
        """
        try:
            ext = os.path.splitext(file_path)[1].lower()
            loader_cls = self.supported_loaders.get(ext)

            if not loader_cls:
                raise ValueError(f"Unsupported file type: {ext}")

            loader = loader_cls(file_path)
            docs = loader.load()
            return "\n".join([doc.page_content for doc in docs])

        except Exception as e:
            logging.error(f"Failed to extract text from {file_path}")
            raise CustomException(e, sys)
