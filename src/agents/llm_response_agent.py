import os
import sys
from dotenv import load_dotenv
from src.logger import logging
from src.exception import CustomException
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from typing import List
from langchain_core.documents import Document

# Load environment variables from .env file at the module level
load_dotenv()

class LLMResponseAgent:
    def __init__(self):
        try:
            self.model_name = os.getenv("MISTRAL_MODEL_NAME")
            self.api_key = os.getenv("OPENROUTER_API_KEY")

            if not self.api_key:
                raise ValueError(
                    "OPENROUTER_API_KEY not found. "
                    "Please make sure it is set in your .env file."
                )

            self.llm = ChatOpenAI(
                model=self.model_name,
                openai_api_key=self.api_key,
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=0.3,
                max_tokens=2000
            )

            logging.info(f"LLMResponseAgent initialized with model: {self.model_name}")
        except Exception as e:
            raise CustomException(e, sys)

    def generate_response(self, query: str, retrieved_docs: List[Document], trace_id: str) -> dict:
        """Generate response using LLM with retrieved context."""
        try:
            # --- IMPROVEMENT: Create a context string with labeled sources ---
            context_parts = []
            for doc in retrieved_docs:
                source = doc.metadata.get('source', 'Unknown Document')
                context_parts.append(f"--- Context from: {source} ---\n{doc.page_content}")
            
            context = "\n\n".join(context_parts)
            
            if not context:
                logging.warning("No context provided to LLM, generating response based on query alone.")
                answer = "I could not find any relevant information in the uploaded documents to answer your question."
            else:
                answer = self.generate_answer(context, query)
            
            return {
                "sender": "LLMResponseAgent",
                "receiver": "CoordinatorAgent",
                "type": "LLM_RESPONSE",
                "trace_id": trace_id,
                "payload": {
                    "answer": answer,
                    "query": query
                }
            }
        except Exception as e:
            raise CustomException(e, sys)

    def generate_answer(self, context: str, query: str) -> str:
        try:
            prompt = f"""You are a helpful and precise assistant. Use the following context, which is composed of sections from different documents, to answer the question. 
Your answer should be comprehensive and synthesize information from all relevant sources provided. 
Explicitly mention the source document (e.g., 'According to Jinil_Patel_Resume.pdf...') when the information is specific to one file.

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
1. Base your answer *only* on the provided context.
2. If the context contains information from multiple documents, synthesize it into a single, coherent answer.
3. Be specific and mention concrete details, skills, or concepts from the context.
4. If the context does not contain enough information to answer the question, clearly state that the information is not available in the provided documents.
5. Do not make up information.

ANSWER:"""

            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            return response.content.strip()

        except Exception as e:
            logging.error(f"Error generating answer: {str(e)}")
            raise CustomException(e, sys)