import uuid
import sys
import os
from src.logger import logging
from src.exception import CustomException
from src.agents.embedding_agent import EmbeddingAgent
from src.agents.retrieval_agent import RetrievalAgent
from src.agents.llm_response_agent import LLMResponseAgent
from src.vector_store.chroma_db import ChromaDBHandler


class CoordinatorAgent:
    def __init__(self, retrieval_agent=None, llm_agent=None):
        try:
            if retrieval_agent and llm_agent:
                self.retriever = retrieval_agent
                self.llm_agent = llm_agent
            else:
                chroma_dir = os.getenv("CHROMA_DIR", "./vectorstore/chroma_db")
                model_name = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

                embedding_agent = EmbeddingAgent(model_name=model_name)
                embeddings = embedding_agent.embedding_model

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

            # Step 2: Extract payload and forward to LLM
            top_docs = retrieval_msg["payload"]["top_docs"]
            sources = retrieval_msg["payload"]["sources"]

            llm_msg = self.llm_agent.generate_response(
                query=query,
                retrieved_docs=top_docs,
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
                    "sources": sources
                }
            }

        except Exception as e:
            logging.error(f"Error in coordinator: {str(e)}")
            raise CustomException(e, sys)