# This Agent is responsible for retrieving relevant document chunks from a vector database based on user queries.

import sys
import uuid
from typing import List, Dict
from src.exception import CustomException
from src.logger import logging
from src.vector_store.chroma_db import ChromaDBHandler
from src.mcp.mcp_like_msg import MCPMessage
from langchain_core.documents import Document


class RetrievalAgent:
    def __init__(self, vector_db=None, persist_directory: str = "vectorstore"):
        """
        Initializes the RetrievalAgent with a vector database.

        Args:
            vector_db: Optional pre-initialized vector store instance.
            persist_directory (str): Directory where ChromaDB persists data
        """
        try:
            if vector_db:
                self.vector_db = vector_db  # Use provided vector store
            else:
                self.vector_db = ChromaDBHandler(persist_directory)  # Load or create ChromaDB

            logging.info("RetrievalAgent initialized successfully")
        except Exception as e:
            raise CustomException(e, sys)

    def retrieve_context(self, query: str, documents: list, trace_id: str) -> dict:
        """
        Retrieves the top relevant document chunks from the vector store based on the input query.

        Args:
            query (str): The user's input or question.
            documents (list): (Unused in current logic) Placeholder for future use.
            trace_id (str): A unique ID to trace this request across agents.

        Output:
            dict: A structured MCP-like dictionary containing the top documents and their sources.
        """
        try:
            logging.info(f"Starting document retrieval for query: {query}")

            # Search the vector store for top-K most relevant document chunks
            top_docs: List[Document] = self.vector_db.similarity_search(query, k=7)

            if not top_docs:
                logging.warning("No relevant documents found for the query.")
                return {
                    "sender": "RetrievalAgent",
                    "receiver": "LLMResponseAgent",
                    "type": "RETRIEVAL_RESULT",
                    "trace_id": trace_id,
                    "payload": {
                        "top_docs": [],
                        "sources": []
                    }
                }

            # Extract the actual content and the sources (e.g., file names)
            top_chunks = [doc.page_content for doc in top_docs]
            sources_used = list(set(doc.metadata.get("source", "Unknown") for doc in top_docs))

            logging.info(f"Retrieved {len(top_docs)} chunks from sources: {sources_used}")

            return {
                "sender": "RetrievalAgent",
                "receiver": "LLMResponseAgent",
                "type": "RETRIEVAL_RESULT",
                "trace_id": trace_id,
                "payload": {
                    "top_docs": top_docs,     # Pass full Document objects with metadata
                    "sources": sources_used   # Source files used in retrieval
                }
            }
        except Exception as e:
            raise CustomException(e, sys)

    def retrieve(self, query: str) -> MCPMessage:
        """
        Alternate simplified method to retrieve relevant chunks for a query.
        Returns a structured MCPMessage object for compatibility with message-passing workflows.

        Args:
            query (str): The user's question or prompt.

        Output:
            MCPMessage: A message object containing retrieved chunks and their metadata.
        """
        try:
            # Generate a unique trace ID for tracking this request
            trace_id = str(uuid.uuid4())

            # Use main logic to retrieve top relevant documents
            result = self.retrieve_context(query, [], trace_id)

            # Prepare payload for the MCPMessage (only include content, query, and sources)
            payload = {
                "top_chunks": [doc.page_content for doc in result["payload"]["top_docs"]],
                "query": query,
                "sources": result["payload"]["sources"]
            }

            # Wrap everything into a structured message for downstream agents
            return MCPMessage(
                sender=result["sender"],
                receiver=result["receiver"],
                msg_type=result["type"],
                trace_id=trace_id,
                payload=payload
            )
        except Exception as e:
            raise CustomException(e, sys)
