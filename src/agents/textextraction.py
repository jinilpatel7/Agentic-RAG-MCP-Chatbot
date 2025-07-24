# This Agent is responsible to extraxt text from various file types using LangChain loaders.
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
        # Initialize and log supported file types
        logging.info("Initializing TextExtractor with supported file types...")

        # Mapping of file extensions to corresponding LangChain loaders
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
        Extracts text from a given file using the appropriate loader based on file extension.

        Args:
            file_path (str): The path to the file to be extracted.

        Output:
            The combined text content extracted from the file.

        Raises:
            CustomException: If the file type is unsupported or extraction fails.
        """
        try:
            # Get the file extension in lowercase
            ext = os.path.splitext(file_path)[1].lower()

            # Select the correct loader class for the file type
            loader_cls = self.supported_loaders.get(ext)

            if not loader_cls:
                # Raise an error if the file type is not supported
                raise ValueError(f"Unsupported file type: {ext}")

            # Instantiate the loader and load the file content
            loader = loader_cls(file_path)
            docs = loader.load()

            # Join content from all pages/documents into one string
            return "\n".join([doc.page_content for doc in docs])

        except Exception as e:
            # Log and raise a custom exception if anything fails
            logging.error(f"Failed to extract text from {file_path}")
            raise CustomException(e, sys)
