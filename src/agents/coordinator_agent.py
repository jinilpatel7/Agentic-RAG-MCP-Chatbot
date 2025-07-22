import uuid
import sys
from src.logger import logging
from src.exception import CustomException
from src.agents.retrieval_agent import RetrievalAgent
from src.agents.llm_response_agent import LLMResponseAgent
from src.vector_store.chroma_db import ChromaDBHandler
from langchain_community.embeddings import HuggingFaceEmbeddings
import os


class CoordinatorAgent:
    def __init__(self, retrieval_agent=None, llm_agent=None):
        try:
            if retrieval_agent and llm_agent:
                self.retriever = retrieval_agent
                self.llm_agent = llm_agent
            else:
                # Initialize with defaults
                chroma_dir = os.getenv("CHROMA_DIR", "./vectorstore/chroma_db")
                model_name = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
                
                # Create vector DB
                embeddings = HuggingFaceEmbeddings(model_name=model_name)
                vector_db = ChromaDBHandler(persist_directory=chroma_dir)
                vector_db.create_or_load(embeddings=embeddings)
                
                self.retriever = RetrievalAgent(vector_db=vector_db)
                self.llm_agent = LLMResponseAgent()
                
            logging.info("CoordinatorAgent initialized successfully")
        except Exception as e:
            raise CustomException(e, sys)

    def handle_query(self, query: str, documents: list = None) -> dict:
        trace_id = str(uuid.uuid4())
        try:
            logging.info(f"Coordinator started processing query with trace_id: {trace_id}")

            # Step 1: Call RetrievalAgent
            retrieval_msg = self.retriever.retrieve_context(query, documents or [], trace_id)

            # Step 2: Extract top chunks and forward to LLM
            top_chunks = retrieval_msg["payload"]["top_chunks"]

            llm_msg = self.llm_agent.generate_response(
                query=query,
                retrieved_chunks=top_chunks,
                trace_id=trace_id
            )

            # Return final LLM message
            return {
                "type": "FINAL_RESPONSE",
                "sender": "CoordinatorAgent",
                "receiver": "UI",
                "trace_id": trace_id,
                "payload": {
                    "answer": llm_msg["payload"]["answer"],
                    "source_chunks": top_chunks
                }
            }

        except Exception as e:
            logging.error(f"Error in coordinator: {str(e)}")
            raise CustomException(e, sys)