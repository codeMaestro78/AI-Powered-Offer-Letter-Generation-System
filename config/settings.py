import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Gemini API Key - Prioritize Streamlit secrets, then .env, with a fallback for local use
    try:
        # This will work in the Streamlit Cloud environment
        import streamlit as st
        GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
    except (ImportError, AttributeError):
        # This will work for local development if a .env file is present
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    # Fallback if the key is still not found
    if not GEMINI_API_KEY:
        print("Warning: GEMINI_API_KEY not found. Please set it in your environment variables or Streamlit secrets.")

    LLM_MODEL = 'gemini-1.5-pro'
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Vector database path
    VECTOR_DB_PATH = os.getenv('VECTOR_DB_PATH', 'data/embeddings')
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '1000'))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    CACHE_DIR = os.getenv("CACHE_DIR", "data/cache")
    
    # Document processing
    SUPPORTED_FORMATS = ['.pdf', '.txt', '.docx', '.csv']
    COMPANY_NAME = os.getenv("COMPANY_NAME", "Company ABC")
    COMPANY_EMAIL = os.getenv("COMPANY_EMAIL", "peopleops@companyabc.com")
    COMPANY_WEBSITE = os.getenv("COMPANY_WEBSITE", "www.companyabc.com")