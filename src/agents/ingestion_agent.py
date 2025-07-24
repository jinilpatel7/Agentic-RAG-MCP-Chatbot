# This Agent is responsible to take the uploaded files and call the TextExtractor to extract text from the files.
import os
import sys

from src.agents.textextraction import TextExtractor
from src.logger import logging
from src.exception import CustomException

from dataclasses import dataclass
@dataclass
class IngestionAgent:
    def __init__(self):
        # Initialize the ingestion agent and load the text extractor
        logging.info("Initializing IngestionAgent...")
        self.extractor = TextExtractor()

    def ingest_files(self, file_paths: list) -> dict:
        """
        Ingests a list of file paths and returns a dictionary mapping each file name
        to its extracted text content.

        
        file_paths (list): A list of file paths to be ingested.

        Output:
            A dictionary where keys are file names and values are extracted text.
        """
        logging.info(f"Starting ingestion for {len(file_paths)} file(s).")
        extracted = {}

        for file_path in file_paths:
            logging.info(f"Processing file: {file_path}")
            try:
                # Extract text from the file using the TextExtractor
                text = self.extractor.extract(file_path)

                # Use the file name (not full path) as the key in the output
                extracted[os.path.basename(file_path)] = text
                logging.info(f"Successfully extracted text from: {file_path}")

            except Exception as e:
                # Log the error and raise a custom exception with context
                logging.error(f"Error processing {file_path}")
                
                # Still include the file in the result with an empty string
                extracted[os.path.basename(file_path)] = ""
                raise CustomException(e, sys)

        logging.info("Completed ingestion of all files.")
        return extracted

# Example usage
if __name__ == "__main__":
    test_files = [
        "C:\\Users\\jinil\\Downloads\\Jinil_Patel_Resume.pdf",
        "C:\\Users\\jinil\\Desktop\\Docker Notes.docx",
        "C:\\Users\\jinil\\Downloads\\Final_leads_comb.csv"
    ]

    logging.info("Running IngestionAgent from main block.")
    agent = IngestionAgent()
    results = agent.ingest_files(test_files)

    for fname, text in results.items():
        print(f"\n--- {fname} ---\n{text[:500]}...\n")
