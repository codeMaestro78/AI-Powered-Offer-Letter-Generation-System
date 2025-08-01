import streamlit as st
from sentence_transformers import SentenceTransformer
import torch
import logging

from src.vector_store import VectorStore

logger = logging.getLogger(__name__)

@st.cache_resource
def load_sentence_transformer(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Loads the SentenceTransformer model and caches it."""
    logger.info(f"Loading sentence transformer model: {model_name}")
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    encoder = SentenceTransformer(model_name, device=device)
    encoder.eval() # Set to evaluation mode
    logger.info(f"Sentence transformer model loaded on device: {device}")
    return encoder

@st.cache_resource
def get_vector_store() -> VectorStore:
    """Initializes and caches the VectorStore object."""
    logger.info("Initializing VectorStore...")
    encoder = load_sentence_transformer()
    vector_store = VectorStore(encoder=encoder)
    logger.info("VectorStore initialized.")
    return vector_store
