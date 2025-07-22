# agents/ingestion_agent.py

import os
import sys

from src.agents.textextraction import TextExtractor
from src.logger import logging
from src.exception import CustomException

from dataclasses import dataclass
@dataclass
class IngestionAgent:
    def __init__(self):
        logging.info("Initializing IngestionAgent...")
        self.extractor = TextExtractor()

    def ingest_files(self, file_paths: list) -> dict:
        """
        Takes list of file paths and returns extracted text per file.
        """
        logging.info(f"Starting ingestion for {len(file_paths)} file(s).")
        extracted = {}
        for file_path in file_paths:
            logging.info(f"Processing file: {file_path}")
            try:
                text = self.extractor.extract(file_path)
                extracted[os.path.basename(file_path)] = text
                logging.info(f"Successfully extracted text from: {file_path}")
            except Exception as e:
                logging.error(f"Error processing {file_path}")
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
