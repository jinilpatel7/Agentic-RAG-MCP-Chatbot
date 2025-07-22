from setuptools import find_packages, setup
from typing import List

HYPHON_E_DOT = "-e ."
def get_requirements(filepath: str) -> List[str]:
    requirements = []

    with open(filepath) as file_obj:
        requirements = file_obj.readlines()
        requirements = [i.replace("\n", "") for i in requirements]

        if HYPHON_E_DOT in requirements:
            requirements.remove(HYPHON_E_DOT)

setup(name='Agentic_RAG_MCP_Chatbot',
      version='0.0.1',
      description='An agent-based RAG chatbot that uses Langchain, ChromaDB, and FastAPI with Model Context Protocol (MCP) messaging. Supports PDF, DOCX, CSV, PPTX, and TXT/Markdown documents, and features a Streamlit UI for multi-turn QA with source context.',
      author='Jinil Patel',
      author_email='ajinilpatel@gmail.com',
      packages=find_packages(),
      install_requires = get_requirements("requirements.txt")
     )