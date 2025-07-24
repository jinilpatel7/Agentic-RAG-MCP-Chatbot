# This Agent is responsible for coordinating the flow of data between the retrieval and LLM response agents.
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
        """
        Initializes the CoordinatorAgent, which orchestrates the full flow:
        - Load or create the vector database.
        - Initialize the retrieval and LLM agents.
        """
        try:
            # If external agent instances are passed, use them directly
            if retrieval_agent and llm_agent:
                self.retriever = retrieval_agent
                self.llm_agent = llm_agent
            else:
                # Load environment variables or use defaults
                chroma_dir = os.getenv("CHROMA_DIR", "./vectorstore/chroma_db")
                model_name = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

                # Step 1: Load the embedding model
                embedding_agent = EmbeddingAgent(model_name=model_name)
                embeddings = embedding_agent.embedding_model

                # Step 2: Create or load vector store with these embeddings
                vector_db = ChromaDBHandler(persist_directory=chroma_dir)
                vector_db.create_or_load(embeddings=embeddings)

                # Step 3: Initialize retrieval and LLM agents
                self.retriever = RetrievalAgent(vector_db=vector_db)
                self.llm_agent = LLMResponseAgent()

            logging.info("CoordinatorAgent initialized successfully")
        except Exception as e:
            raise CustomException(e, sys)

    def handle_query(self, query: str, documents: list = None) -> dict:
        """
        Main method that handles a user query by:
        1. Retrieving relevant context from stored documents.
        2. Generating a response using the LLM.
        3. Returning the final result to be displayed in the UI.

        Args:
            query (str): The user's question or input.
            documents (list): Optional additional documents (currently unused).

        Output:
            dict: Final structured response containing the LLM answer and the source files.
        """
        trace_id = str(uuid.uuid4())  # Generate unique trace ID for tracking
        try:
            logging.info(f"Coordinator started processing query with trace_id: {trace_id}")

            # Step 1: Retrieve relevant document chunks from the vector store
            retrieval_msg = self.retriever.retrieve_context(query, documents or [], trace_id)

            # Step 2: Pass the top documents to the LLM agent for answer generation
            top_docs = retrieval_msg["payload"]["top_docs"]
            sources = retrieval_msg["payload"]["sources"]

            llm_msg = self.llm_agent.generate_response(
                query=query,
                retrieved_docs=top_docs,
                trace_id=trace_id
            )

            # Step 3: Format and return final response to the UI or API
            return {
                "type": "FINAL_RESPONSE",
                "sender": "CoordinatorAgent",
                "receiver": "UI",
                "trace_id": trace_id,
                "payload": {
                    "answer": llm_msg["payload"]["answer"],
                    "sources": sources  # Show which documents were used
                }
            }

        except Exception as e:
            logging.error(f"Error in coordinator: {str(e)}")
            raise CustomException(e, sys)
