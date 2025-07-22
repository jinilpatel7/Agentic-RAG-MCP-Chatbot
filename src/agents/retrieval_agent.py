import sys
import uuid
from typing import List, Dict
from src.exception import CustomException
from src.logger import logging
from src.vector_store.chroma_db import ChromaDBHandler
from src.mcp.mcp_like_msg import MCPMessage


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
        """Retrieve relevant context for the query"""
        try:
            logging.info(f"Starting document retrieval for query: {query}")
            
            # Search with more results to ensure diversity
            docs = self.vector_db.similarity_search(query, k=10)
            
            # Group by source and select top chunks from each source
            chunks_by_source = {}
            for doc in docs:
                source = doc.metadata.get("source", "Unknown")
                if source not in chunks_by_source:
                    chunks_by_source[source] = []
                chunks_by_source[source].append(doc.page_content)
            
            # Get top chunks ensuring diversity across sources
            top_chunks = []
            sources_used = []
            
            # First, get at least one chunk from each source
            for source, chunks in chunks_by_source.items():
                if chunks:
                    top_chunks.append(chunks[0])
                    sources_used.append(source)
            
            # Then fill remaining slots with best chunks
            remaining_slots = 5 - len(top_chunks)
            if remaining_slots > 0:
                for doc in docs[:remaining_slots]:
                    if doc.page_content not in top_chunks:
                        top_chunks.append(doc.page_content)
                        sources_used.append(doc.metadata.get("source", "Unknown"))
            
            # Limit to 5 chunks
            top_chunks = top_chunks[:5]
            
            logging.info(f"Retrieved {len(top_chunks)} chunks from sources: {set(sources_used)}")

            return {
                "sender": "RetrievalAgent",
                "receiver": "LLMResponseAgent",
                "type": "RETRIEVAL_RESULT",
                "trace_id": trace_id,
                "payload": {
                    "top_chunks": top_chunks,
                    "query": query,
                    "sources": list(set(sources_used))
                }
            }
        except Exception as e:
            raise CustomException(e, sys)

    def retrieve(self, query: str) -> MCPMessage:
        """Alternative method for backward compatibility"""
        try:
            trace_id = str(uuid.uuid4())
            result = self.retrieve_context(query, [], trace_id)
            return MCPMessage(
                sender=result["sender"],
                receiver=result["receiver"],
                msg_type=result["type"],
                trace_id=result["trace_id"],
                payload=result["payload"]
            )
        except Exception as e:
            raise CustomException(e, sys)