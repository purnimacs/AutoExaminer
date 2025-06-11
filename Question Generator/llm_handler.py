# llm_handler.py
from typing import Optional
from rate_limited_llm import get_rate_limited_llm
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
 
logger = logging.getLogger(__name__)  # Fixed the __name__ formatting

def initialize_llm(google_api_key: str):
    """Initialize the rate-limited Gemini LLM"""
    try:
        llm = get_rate_limited_llm(google_api_key)
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        raise

def setup_vector_store(texts, persist_directory="./chroma_persist"):
    """Set up the vector store using HuggingFace embeddings"""
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            length_function=len,
        )
        
        if isinstance(texts, str):
            texts = text_splitter.split_text(texts)
        
        vectordb = Chroma.from_texts(
            texts=texts,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        
        return vectordb
    except Exception as e:
        logger.error(f"Error setting up vector store: {e}")
        raise

def create_qa_chain(llm, vector_store):
    """Create a question-answering chain with optimized retrieval"""
    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(
                search_kwargs={  
                    "k": 3,  # Increased from 2 for more context
                }
            ),
            return_source_documents=True,
        )
        return qa_chain
    except Exception as e:
        logger.error(f"Error creating QA chain: {e}")
        raise

def get_response(qa_chain, question: str) -> Optional[dict]:
    """Get response from the QA chain with error handling"""
    try:
        response = qa_chain.invoke({"query": question})
        return {
            "answer": response["result"],
            "source_documents": response["source_documents"]
        }
    except Exception as e:
        logger.error(f"Error getting response: {e}")
        return None
