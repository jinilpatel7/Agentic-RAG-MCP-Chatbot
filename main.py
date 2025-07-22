import os
from src.agents.ingestion_agent import IngestionAgent
from src.agents.processing import TextProcessing
from src.agents.embedding_agent import EmbeddingAgent
from src.logger import logging

CHROMA_DIR = "./vectorstore/chroma_db"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def run_end_to_end_test():
    test_files = [
        "C:\\Users\\jinil\\Downloads\\Jinil_Patel_Resume.pdf",
        "C:\\Users\\jinil\\Desktop\\Docker Notes.docx"
    ]

    logging.info("Starting full pipeline test...")

    # Step 1: Ingest
    ingestion_agent = IngestionAgent()
    extracted = ingestion_agent.ingest_files(test_files)

    # Step 2: Process
    processor = TextProcessing(chunk_size=500, chunk_overlap=50)
    all_docs = []

    for fname, text in extracted.items():
        if not text.strip():
            print(f"No content extracted from {fname}")
            continue
        docs = processor.process(text, metadata={"source": fname})
        all_docs.extend(docs)

    # Step 3: Embed and store
    embedder = EmbeddingAgent(model_name=MODEL_NAME)
    embedder.embed_and_store(all_docs, persist_directory=CHROMA_DIR)

    logging.info("Pipeline test completed successfully.")

if __name__ == "__main__":
    run_end_to_end_test()
