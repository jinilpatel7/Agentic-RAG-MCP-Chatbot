import os
import sys
from src.logger import logging
from src.exception import CustomException
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

load_dotenv()


class LLMResponseAgent:
    def __init__(self):
        try:
            self.model_name = os.getenv("MISTRAL_MODEL_NAME")
            self.api_key = os.getenv("OPENROUTER_API_KEY")

            self.llm = ChatOpenAI(
                model=self.model_name,
                openai_api_key=self.api_key,
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=0.3,  # Lower temperature for more focused answers
                max_tokens=1000   # More tokens for comprehensive answers
            )

            logging.info(f"LLMResponseAgent initialized with model: {self.model_name}")
        except Exception as e:
            raise CustomException(e, sys)

    def generate_response(self, query: str, retrieved_chunks: list, trace_id: str) -> dict:
        """Generate response using LLM with retrieved context"""
        try:
            # Join chunks with clear separation
            context = "\n\n---\n\n".join(retrieved_chunks)
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
            prompt = f"""You are a helpful assistant. Use the following context to answer the question. 
Be specific and comprehensive in your answer. If the context contains information from multiple documents, 
synthesize the information appropriately.

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
1. Answer based on the provided context
2. Be specific and mention concrete details from the context
3. If the context mentions specific skills, technologies, or concepts, list them clearly
4. If the context doesn't contain relevant information, say so
5. Do not make up information not present in the context

ANSWER:"""

            messages = [HumanMessage(content=prompt)]
            response = self.llm(messages)
            return response.content.strip()

        except Exception as e:
            logging.error(f"Error generating answer: {str(e)}")
            raise CustomException(e, sys)