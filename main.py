import os
import sys
import shutil
from src.agents.ingestion_agent import IngestionAgent
from src.agents.processing import TextProcessing
from src.agents.embedding_agent import EmbeddingAgent
from src.agents.coordinator_agent import CoordinatorAgent
from src.vector_store.chroma_db import ChromaDBHandler
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.logger import logging
from src.exception import CustomException
from dotenv import load_dotenv

# Load environment
load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_DIR", "./vectorstore/chroma_db")
MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")


def clear_vector_store():
    """Clear existing vector store"""
    if os.path.exists(CHROMA_DIR):
        try:
            shutil.rmtree(CHROMA_DIR)
            logging.info(f"Cleared existing vector store at {CHROMA_DIR}")
        except Exception as e:
            logging.error(f"Error clearing vector store: {str(e)}")


def run_end_to_end_test():
    try:
        # Clear existing vector store to avoid duplicates
        clear_vector_store()
        
        test_files = [
            "C:\\Users\\jinil\\Downloads\\Jinil_Patel_Resume.pdf",
            "C:\\Users\\jinil\\Desktop\\Docker Notes.docx"
        ]

        logging.info("Starting full agentic pipeline test...")

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
            
            # Use full file path as source
            docs = processor.process(text, metadata={"source": fname})
            all_docs.extend(docs)
            logging.info(f"Processed {len(docs)} chunks from {fname}")

        logging.info(f"Total documents to embed: {len(all_docs)}")

        # Step 3: Embed + store in Chroma
        embedder = EmbeddingAgent(model_name=MODEL_NAME)
        embedder.embed_and_store(all_docs, persist_directory=CHROMA_DIR)

        # Step 4: Initialize coordinator with the same vector store
        embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
        vector_db = ChromaDBHandler(persist_directory=CHROMA_DIR)
        vector_db.create_or_load(embeddings=embeddings)
        
        from src.agents.retrieval_agent import RetrievalAgent
        from src.agents.llm_response_agent import LLMResponseAgent
        
        retrieval_agent = RetrievalAgent(vector_db=vector_db)
        llm_agent = LLMResponseAgent()
        coordinator = CoordinatorAgent(retrieval_agent=retrieval_agent, llm_agent=llm_agent)

        # Step 5: Test multiple queries
        queries = [
            "What are the skills mentioned in the resume?",
            "What is Docker and how does it work?",
            "Who's resume is it and what are their qualifications?"
            ]

        for query in queries:
            print(f"\n{'='*60}")
            print(f"QUERY: {query}")
            print('='*60)
            
            response = coordinator.handle_query(query)
            
            print("\n--- ANSWER ---")
            print(response["payload"]["answer"])
            
            print("\n--- SOURCE CHUNKS ---")
            chunks = response["payload"]["source_chunks"]
            unique_chunks = []
            seen = set()
            
            # Remove duplicate chunks
            for chunk in chunks:
                chunk_preview = chunk[:100]
                if chunk_preview not in seen:
                    seen.add(chunk_preview)
                    unique_chunks.append(chunk)
            
            for i, chunk in enumerate(unique_chunks, 1):
                print(f"\n[Chunk {i}]")
                print(chunk[:300] + "..." if len(chunk) > 300 else chunk)

        logging.info("Pipeline test completed successfully.")

    except Exception as e:
        logging.error(f"Error in pipeline: {str(e)}")
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_end_to_end_test()