import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # gemini api key
    GEMINI_API_KEY = os.getenv("")
    LLM_MODEL='gemini-1.5-pro'
    EMBEDDING_MODEL=os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")

    # vector data base path
    VECTOR_DB_PATH=os.getenv('VECTOR_DB_PATH','data/embeddings')
    CHUNK_SIZE=int(os.getenv('CHUNK_SIZE','1000'))
    CHUNK_OVERLAP=int(os.getenv("CHUNK_OVERLAP", "200"))
    CACHE_DIR = os.getenv("CACHE_DIR", "data/cache")
    
    # document processing

    SUPPORTED_FORMATS=['.pdf', '.txt', '.csv']
    COMPANY_NAME = os.getenv("COMPANY_NAME", "Company ABC")
    COMPANY_EMAIL = os.getenv("COMPANY_EMAIL", "peopleops@companyabc.com")
    COMPANY_WEBSITE = os.getenv("COMPANY_WEBSITE", "www.companyabc.com")