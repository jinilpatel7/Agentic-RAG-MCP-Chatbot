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
        try:
            if vector_db:
                self.vector_db = vector_db
            else:
                self.vector_db = ChromaDBHandler(persist_directory)
            logging.info("RetrievalAgent initialized successfully")
        except Exception as e:
            raise CustomException(e, sys)

    def retrieve_context(self, query: str, documents: list, trace_id: str) -> dict:
        """
        Retrieve the most relevant document chunks for the query.
        This new logic prioritizes relevance above all.
        """
        try:
            logging.info(f"Starting document retrieval for query: {query}")
            
            # --- IMPROVEMENT: Retrieve the top K most relevant docs directly ---
            # We fetch more (e.g., 7) to give the LLM a richer context.
            # The similarity search is already sorted by relevance.
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

            # Extract the page content (the text chunks) and a unique list of sources
            top_chunks = [doc.page_content for doc in top_docs]
            sources_used = list(set(doc.metadata.get("source", "Unknown") for doc in top_docs))
            
            logging.info(f"Retrieved {len(top_docs)} chunks from sources: {sources_used}")

            return {
                "sender": "RetrievalAgent",
                "receiver": "LLMResponseAgent",
                "type": "RETRIEVAL_RESULT",
                "trace_id": trace_id,
                "payload": {
                    # We now pass the full Document objects to preserve metadata
                    "top_docs": top_docs,
                    "sources": sources_used
                }
            }
        except Exception as e:
            raise CustomException(e, sys)

    def retrieve(self, query: str) -> MCPMessage:
        """Alternative method for backward compatibility if needed."""
        try:
            trace_id = str(uuid.uuid4())
            result = self.retrieve_context(query, [], trace_id)
            
            # Note: This method might need adjustment if downstream code relies on its specific payload.
            # For our current flow, retrieve_context is used directly.
            payload = {
                "top_chunks": [doc.page_content for doc in result["payload"]["top_docs"]],
                "query": query,
                "sources": result["payload"]["sources"]
            }
            
            return MCPMessage(
                sender=result["sender"],
                receiver=result["receiver"],
                msg_type=result["type"],
                trace_id=trace_id,
                payload=payload
            )
        except Exception as e:
            raise CustomException(e, sys)