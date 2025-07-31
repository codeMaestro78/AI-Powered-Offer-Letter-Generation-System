import os
import pickle
import json
import hashlib
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import threading
from pathlib import Path
import logging

# Suppress FAISS GPU warnings and handle import gracefully
import os
os.environ['FAISS_NO_AVX2'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'  # Prevent threading issues

import warnings
import sys

def _import_faiss_safely():
    """Import FAISS with proper error handling and warning suppression"""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Suppress specific FAISS GPU warnings
            warnings.filterwarnings("ignore", message=".*GPU.*")
            warnings.filterwarnings("ignore", message=".*CUDA.*")
            warnings.filterwarnings("ignore", message=".*GpuIndexIVFFlat.*")
            import faiss
            return faiss
    except ImportError as e:
        print(f"Error importing FAISS: {e}")
        print("Please install FAISS with: pip install faiss-cpu")
        sys.exit(1)
    except Exception as e:
        print(f"Warning during FAISS import: {e}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import faiss
            return faiss

faiss = _import_faiss_safely()

import numpy as np
from sentence_transformers import SentenceTransformer
import torch

from config.settings import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Enhanced search result with metadata"""
    chunk: Dict[str, Any]
    similarity_score: float
    rank: int
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class IndexStats:
    """Statistics for the vector index"""
    total_chunks: int = 0
    index_size_mb: float = 0.0
    embedding_dimension: int = 0
    last_updated: Optional[str] = None
    index_type: str = ""
    creation_time: float = 0.0
    metadata_keys: List[str] = field(default_factory=list)

class VectorStore:
    """High-performance vector store with advanced features"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_gpu: bool = None):
        self.config = Config()
        
        # GPU detection and setup
        self.device = self._setup_device(use_gpu)
        logger.info(f"🔧 Using device: {self.device}")
        
        # Model initialization with optimizations
        self.encoder = self._initialize_encoder(model_name)
        
        # Index management
        self.index = None
        self.chunks = []
        self.metadata_index = {}  # Fast metadata lookup
        self.chunk_hash_map = {}  # Deduplication mapping
        
        # File paths
        self.embeddings_path = Path(self.config.VECTOR_DB_PATH) / "embeddings.pkl"
        self.index_path = Path(self.config.VECTOR_DB_PATH) / "faiss_index.bin"
        self.metadata_path = Path(self.config.VECTOR_DB_PATH) / "metadata.json"
        self.stats_path = Path(self.config.VECTOR_DB_PATH) / "index_stats.json"
        
        # Performance tracking
        self.stats = IndexStats()
        self.search_cache = {}
        self.cache_lock = threading.Lock()
        self.max_cache_size = 1000
        self.cache_ttl = 3600  # 1 hour
        
        # Thread safety
        self.index_lock = threading.RLock()
        
        # Create directory structure
        Path(self.config.VECTOR_DB_PATH).mkdir(parents=True, exist_ok=True)
        
        # Load existing data
        self._load_existing_data()

    def _setup_device(self, use_gpu: Optional[bool]) -> str:
        """Setup computation device with automatic detection"""
        if use_gpu is None:
            # Auto-detect
            if torch.cuda.is_available():
                device = 'cuda'
                logger.info(f"🚀 CUDA detected: {torch.cuda.get_device_name()}")
            elif torch.backends.mps.is_available():
                device = 'mps'
                logger.info("🚀 MPS (Apple Silicon) detected")
            else:
                device = 'cpu'
                logger.info("💻 Using CPU")
        else:
            device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        
        return device

    def _initialize_encoder(self, model_name: str) -> SentenceTransformer:
        """Initialize sentence transformer with optimizations"""
        try:
            encoder = SentenceTransformer(model_name, device=self.device)
            
            # Optimize for inference
            if hasattr(encoder, 'eval'):
                encoder.eval()
            
            # Test encoding to warm up
            test_embedding = encoder.encode(["test"], show_progress_bar=False)
            self.embedding_dim = test_embedding.shape[1]
            
            logger.info(f"✅ Encoder initialized: {model_name} (dim: {self.embedding_dim})")
            return encoder
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize encoder: {str(e)}")
            raise

    def _load_existing_data(self):
        """Load existing embeddings and metadata"""
        if self.load_embeddings():
            logger.info("📁 Loaded existing vector store data")
        else:
            logger.info("🆕 Starting with empty vector store")

    def create_embeddings(self, chunks: List[Dict[str, Any]], 
                         batch_size: int = 32, 
                         use_deduplication: bool = True,
                         save_intermediate: bool = True) -> None:
        """Create embeddings with advanced features and optimizations"""
        
        start_time = time.time()
        logger.info(f"🚀 Starting embedding creation for {len(chunks)} chunks")
        
        try:
            # Input validation
            if not chunks:
                logger.error("❌ No chunks provided for embedding creation")
                raise ValueError("Empty chunks list provided")
            
            # Preprocessing and deduplication
            if use_deduplication:
                chunks = self._deduplicate_chunks(chunks)
                logger.info(f"📊 After deduplication: {len(chunks)} unique chunks")
            
            # Final validation after deduplication
            if not chunks:
                logger.error("❌ No chunks remaining after deduplication")
                raise ValueError("No valid chunks after deduplication")
            
            # Store chunks
            self.chunks = chunks
            self._build_metadata_index()
            
            # Extract texts for embedding
            texts = self._extract_and_validate_texts(chunks)
            
            # Create embeddings in batches
            embeddings = self._create_embeddings_batched(texts, batch_size)
            
            # Validate embeddings
            if embeddings is None or embeddings.size == 0:
                raise ValueError("Failed to create any embeddings")
            
            # Create optimized FAISS index
            self._create_optimized_index(embeddings)
            
            # Save to disk
            if save_intermediate:
                self._save_to_disk(embeddings)
            
            # Update statistics
            self._update_stats(embeddings, start_time)
            
            logger.info(f"✅ Embedding creation completed in {time.time() - start_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Error creating embeddings: {str(e)}")
            raise

    def _extract_and_validate_texts(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Extract and validate texts from chunks"""
        texts = []
        
        for i, chunk in enumerate(chunks):
            content = chunk.get("content", "")
            
            # Validate content
            if not isinstance(content, str):
                logger.warning(f"⚠️ Non-string content in chunk {i}: {type(content)}")
                content = str(content) if content is not None else ""
            
            # Handle empty content
            if not content.strip():
                logger.warning(f"⚠️ Empty content in chunk {i}, using placeholder")
                content = f"[Empty chunk {i}]"
            
            texts.append(content)
        
        logger.info(f"📝 Extracted {len(texts)} texts for embedding")
        return texts

    def _deduplicate_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate chunks based on content hash"""
        if not chunks:
            return chunks
            
        unique_chunks = []
        seen_hashes = set()
        
        for chunk in chunks:
            content = chunk.get('content', '')
            
            # Handle non-string content
            if not isinstance(content, str):
                content = str(content) if content is not None else ""
            
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                self.chunk_hash_map[len(unique_chunks)] = content_hash
                unique_chunks.append(chunk)
        
        return unique_chunks

    def _build_metadata_index(self):
        """Build fast metadata lookup index"""
        self.metadata_index = {}
        
        for i, chunk in enumerate(self.chunks):
            metadata = chunk.get('metadata', {})
            if not isinstance(metadata, dict):
                continue
                
            for key, value in metadata.items():
                if key not in self.metadata_index:
                    self.metadata_index[key] = {}
                
                # Convert value to string for consistent indexing
                value_str = str(value)
                if value_str not in self.metadata_index[key]:
                    self.metadata_index[key][value_str] = []
                self.metadata_index[key][value_str].append(i)

    def _create_embeddings_batched(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Create embeddings in optimized batches with better error handling"""
        
        if not texts:
            logger.error("❌ No texts provided for embedding creation")
            raise ValueError("Empty texts list")
        
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        successful_batches = 0
        
        logger.info(f"📦 Processing {total_batches} batches of size {batch_size}")
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            logger.info(f"🔄 Processing batch {batch_num}/{total_batches} ({len(batch_texts)} texts)")
            
            try:
                # Validate batch texts
                valid_batch_texts = []
                for text in batch_texts:
                    if isinstance(text, str) and text.strip():
                        valid_batch_texts.append(text)
                    else:
                        valid_batch_texts.append("empty")
                
                if not valid_batch_texts:
                    logger.warning(f"⚠️ No valid texts in batch {batch_num}, skipping")
                    continue
                
                # Create embeddings for batch
                batch_embeddings = self.encoder.encode(
                    valid_batch_texts,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=False  # We'll normalize later
                )
                
                # Validate batch embeddings
                if batch_embeddings is None or batch_embeddings.size == 0:
                    logger.error(f"❌ Failed to create embeddings for batch {batch_num}")
                    continue
                
                # Ensure correct shape and type
                batch_embeddings = np.array(batch_embeddings, dtype=np.float32)
                if len(batch_embeddings.shape) != 2:
                    logger.error(f"❌ Invalid embedding shape in batch {batch_num}: {batch_embeddings.shape}")
                    continue
                
                all_embeddings.append(batch_embeddings)
                successful_batches += 1
                logger.info(f"✅ Batch {batch_num} completed: shape {batch_embeddings.shape}")
                
            except Exception as e:
                logger.error(f"❌ Error processing batch {batch_num}: {str(e)}")
                # Create zero embeddings for failed batch as fallback
                try:
                    zero_embeddings = np.zeros((len(batch_texts), self.embedding_dim), dtype=np.float32)
                    all_embeddings.append(zero_embeddings)
                    logger.warning(f"⚠️ Using zero embeddings for failed batch {batch_num}")
                except Exception as fallback_error:
                    logger.error(f"❌ Failed to create fallback embeddings: {fallback_error}")
        
        # Check if we have any embeddings
        if not all_embeddings:
            logger.error("❌ No embeddings were created successfully")
            raise RuntimeError("Failed to create any embeddings from the provided texts")
        
        # Concatenate all embeddings
        try:
            embeddings = np.vstack(all_embeddings).astype(np.float32)
            logger.info(f"📊 Created embeddings shape: {embeddings.shape}")
            logger.info(f"✅ Successfully processed {successful_batches}/{total_batches} batches")
            
            # Validate final embeddings
            if embeddings.shape[0] != len(texts):
                logger.warning(f"⚠️ Embedding count mismatch: expected {len(texts)}, got {embeddings.shape[0]}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Error concatenating embeddings: {str(e)}")
            logger.error(f"All embeddings shapes: {[emb.shape for emb in all_embeddings]}")
            raise

    def _create_optimized_index(self, embeddings: np.ndarray):
        """Create optimized FAISS index based on data size"""
        
        if embeddings is None or embeddings.size == 0:
            raise ValueError("Cannot create index with empty embeddings")
        
        with self.index_lock:
            dimension = embeddings.shape[1]
            n_vectors = embeddings.shape[0]
            
            logger.info(f"🏗️ Creating FAISS index for {n_vectors} vectors, dim={dimension}")
            
            # Choose index type based on dataset size
            if n_vectors < 1000:
                # Small dataset: Use flat index
                self.index = faiss.IndexFlatIP(dimension)
                index_type = "IndexFlatIP"
                
            elif n_vectors < 10000:
                # Medium dataset: Use IVF with small number of centroids
                nlist = min(100, max(1, n_vectors // 10))
                quantizer = faiss.IndexFlatIP(dimension)
                self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
                index_type = f"IndexIVFFlat_nlist{nlist}"
                
            else:
                # Large dataset: Use IVF with more centroids and PQ
                nlist = min(1000, max(1, n_vectors // 100))
                m = min(16, max(1, dimension // 4))  # PQ subquantizers
                quantizer = faiss.IndexFlatIP(dimension)
                self.index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, 8)
                index_type = f"IndexIVFPQ_nlist{nlist}_m{m}"
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Train index if needed
            if hasattr(self.index, 'train') and not self.index.is_trained:
                logger.info("🎯 Training FAISS index...")
                self.index.train(embeddings)
            
            # Add vectors to index
            self.index.add(embeddings)
            
            # Set search parameters for IVF indices
            if hasattr(self.index, 'nprobe'):
                self.index.nprobe = min(32, max(1, self.index.nlist // 4))
            
            self.stats.index_type = index_type
            logger.info(f"✅ Created {index_type} index with {self.index.ntotal} vectors")

    def _save_to_disk(self, embeddings: np.ndarray):
        """Save all data to disk with compression"""
        
        try:
            logger.info("💾 Saving vector store to disk...")
            
            # Save chunks and metadata with compression
            save_data = {
                'chunks': self.chunks,
                'metadata_index': self.metadata_index,
                'chunk_hash_map': self.chunk_hash_map,
                'embedding_dimension': self.embedding_dim,
                'model_name': getattr(self.encoder, '_target_device', 'unknown'),
                'created_at': time.time()
            }
            
            with open(self.embeddings_path, 'wb') as f:
                pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Save FAISS index
            if self.index is not None:
                faiss.write_index(self.index, str(self.index_path))
            
            # Save statistics
            self._save_stats()
            
            logger.info("✅ Vector store saved successfully")
            
        except Exception as e:
            logger.error(f"❌ Error saving to disk: {str(e)}")
            raise

    def load_embeddings(self) -> bool:
        """Load embeddings and index from disk with validation"""
        
        try:
            if not (self.embeddings_path.exists() and self.index_path.exists()):
                logger.info("📁 No existing vector store found")
                return False
            
            logger.info("📂 Loading vector store from disk...")
            
            # Load chunks and metadata
            with open(self.embeddings_path, 'rb') as f:
                data = pickle.load(f)
                self.chunks = data.get('chunks', [])
                self.metadata_index = data.get('metadata_index', {})
                self.chunk_hash_map = data.get('chunk_hash_map', {})
                stored_dim = data.get('embedding_dimension', self.embedding_dim)
            
            # Validate dimension compatibility
            if stored_dim != self.embedding_dim:
                logger.warning(f"⚠️ Dimension mismatch: stored={stored_dim}, current={self.embedding_dim}")
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(str(self.index_path))
            
            # Load statistics
            self._load_stats()
            
            # Validate consistency
            if len(self.chunks) != self.index.ntotal:
                logger.warning(f"⚠️ Inconsistency: chunks={len(self.chunks)}, index={self.index.ntotal}")
                return False
            
            logger.info(f"✅ Loaded vector store: {len(self.chunks)} chunks, {self.stats.index_type}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading embeddings: {str(e)}")
            return False

    def search(self, query: str, k: int = 5, 
               score_threshold: float = 0.0,
               use_cache: bool = True,
               rerank: bool = False) -> List[SearchResult]:
        """Advanced search with caching and reranking"""
        
        if self.index is None or len(self.chunks) == 0:
            logger.warning("⚠️ No index available for search")
            return []
        
        # Input validation
        if not query or not query.strip():
            logger.warning("⚠️ Empty query provided")
            return []
        
        # Check cache first
        if use_cache:
            cache_key = self._get_cache_key(query, k, score_threshold)
            cached_result = self._get_cached_result(cache_key)
            if cached_result is not None:
                logger.info("📦 Using cached search result")
                return cached_result
        
        try:
            start_time = time.time()
            
            # Create query embedding
            query_embedding = self.encoder.encode([query.strip()], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            
            # Search with increased k for potential reranking
            search_k = min(k * 2 if rerank else k, self.index.ntotal)
            scores, indices = self.index.search(query_embedding, search_k)
            
            # Process results
            results = self._process_search_results(scores[0], indices[0], score_threshold, k)
            
            # Optional reranking
            if rerank and len(results) > k:
                results = self._rerank_results(query, results, k)
            
            # Cache results
            if use_cache:
                self._cache_result(cache_key, results)
            
            search_time = time.time() - start_time
            logger.info(f"🔍 Search completed in {search_time:.3f}s, found {len(results)} results")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error during search: {str(e)}")
            return []

    # ... (rest of the methods remain the same as in your original code)
    
    def _process_search_results(self, scores: np.ndarray, indices: np.ndarray, 
                              score_threshold: float, k: int) -> List[SearchResult]:
        """Process raw search results into SearchResult objects"""
        
        results = []
        for rank, (score, idx) in enumerate(zip(scores, indices)):
            if idx >= len(self.chunks) or score < score_threshold:
                continue
            
            chunk = self.chunks[idx].copy()
            
            # Add search metadata
            search_metadata = {
                'search_score': float(score),
                'search_rank': rank,
                'chunk_index': int(idx)
            }
            
            result = SearchResult(
                chunk=chunk,
                similarity_score=float(score),
                rank=rank,
                metadata=search_metadata
            )
            
            results.append(result)
            
            if len(results) >= k:
                break
        
        return results

    def _rerank_results(self, query: str, results: List[SearchResult], k: int) -> List[SearchResult]:
        """Rerank results using additional scoring"""
        
        try:
            # Simple reranking based on content length and keyword matching
            query_words = set(query.lower().split())
            
            for result in results:
                content = result.chunk.get('content', '').lower()
                
                # Keyword matching bonus
                content_words = set(content.split())
                keyword_overlap = len(query_words.intersection(content_words))
                keyword_bonus = keyword_overlap / max(len(query_words), 1) * 0.1
                
                # Content length penalty for very short/long content
                content_length = len(content)
                if content_length < 50:
                    length_penalty = 0.05
                elif content_length > 2000:
                    length_penalty = 0.02
                else:
                    length_penalty = 0
                
                # Update score
                result.similarity_score += keyword_bonus - length_penalty
                result.metadata['rerank_bonus'] = keyword_bonus
                result.metadata['length_penalty'] = length_penalty
            
            # Sort by updated scores
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            # Update ranks
            for i, result in enumerate(results[:k]):
                result.rank = i
            
            return results[:k]
            
        except Exception as e:
            logger.warning(f"⚠️ Reranking failed: {str(e)}")
            return results[:k]

    def _get_cache_key(self, query: str, k: int, score_threshold: float) -> str:
        """Generate cache key for search query"""
        key_string = f"{query}_{k}_{score_threshold}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_cached_result(self, cache_key: str) -> Optional[List[SearchResult]]:
        """Get cached search result if available and not expired"""
        with self.cache_lock:
            if cache_key in self.search_cache:
                cached_data = self.search_cache[cache_key]
                if time.time() - cached_data['timestamp'] < self.cache_ttl:
                    return cached_data['results']
                else:
                    del self.search_cache[cache_key]
        return None

    def _cache_result(self, cache_key: str, results: List[SearchResult]):
        """Cache search results with timestamp"""
        with self.cache_lock:
            # Limit cache size
            if len(self.search_cache) >= self.max_cache_size:
                # Remove oldest entry
                oldest_key = min(self.search_cache.keys(), 
                               key=lambda k: self.search_cache[k]['timestamp'])
                del self.search_cache[oldest_key]
            
            self.search_cache[cache_key] = {
                'results': results,
                'timestamp': time.time()
            }

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the vector store"""
        
        self.stats.metadata_keys = list(self.metadata_index.keys())  # Update stats object
        
        stats = {
            'total_chunks': len(self.chunks),
            'index_type': self.stats.index_type,
            'embedding_dimension': self.embedding_dim,
            'index_size_vectors': self.index.ntotal if self.index else 0,
            'cache_size': len(self.search_cache),
            'metadata_keys': self.stats.metadata_keys,  # Use the updated field
            'device': self.device,
            'model_name': getattr(self.encoder, '_target_device', 'unknown')
        }
        
        # Add memory usage estimate
        if self.index:
            vector_memory_mb = (self.index.ntotal * self.embedding_dim * 4) / (1024 * 1024)  # 4 bytes per float32
            stats['estimated_memory_mb'] = vector_memory_mb
        
        # Add file sizes
        if self.embeddings_path.exists():
            stats['embeddings_file_size_mb'] = self.embeddings_path.stat().st_size / (1024 * 1024)
        if self.index_path.exists():
            stats['index_file_size_mb'] = self.index_path.stat().st_size / (1024 * 1024)
        
        return stats

    def _update_stats(self, embeddings: np.ndarray, start_time: float):
        """Update internal statistics"""
        self.stats.total_chunks = len(self.chunks)
        self.stats.embedding_dimension = embeddings.shape[1]
        self.stats.last_updated = time.strftime('%Y-%m-%d %H:%M:%S')
        self.stats.creation_time = time.time() - start_time
        self.stats.metadata_keys = list(self.metadata_index.keys())  

    def _save_stats(self):
        """Save statistics to disk"""
        try:
            stats_dict = {
                'total_chunks': self.stats.total_chunks,
                'embedding_dimension': self.stats.embedding_dimension,
                'index_type': self.stats.index_type,
                'last_updated': self.stats.last_updated,
                'creation_time': self.stats.creation_time,
                'metadata_keys': self.stats.metadata_keys  # Add this
            }
            
            with open(self.stats_path, 'w') as f:
                json.dump(stats_dict, f, indent=2)
                
        except Exception as e:
            logger.warning(f"⚠️ Could not save stats: {str(e)}")

    def _load_stats(self):
        """Load statistics from disk"""
        try:
            if self.stats_path.exists():
                with open(self.stats_path, 'r') as f:
                    stats_dict = json.load(f)
                    self.stats.total_chunks = stats_dict.get('total_chunks', 0)
                    self.stats.embedding_dimension = stats_dict.get('embedding_dimension', 0)
                    self.stats.index_type = stats_dict.get('index_type', '')
                    self.stats.last_updated = stats_dict.get('last_updated')
                    self.stats.creation_time = stats_dict.get('creation_time', 0.0)
                    self.stats.metadata_keys = stats_dict.get('metadata_keys', [])  # Add this
                        
        except Exception as e:
            logger.warning(f"⚠️ Could not load stats: {str(e)}")

    def clear_cache(self):
        """Clear all caches"""
        with self.cache_lock:
            self.search_cache.clear()
        logger.info("🧹 Search cache cleared")

    def __len__(self) -> int:
        """Return number of chunks in the store"""
        return len(self.chunks)

    def __repr__(self) -> str:
        """String representation of the vector store"""
        return (f"VectorStore(chunks={len(self.chunks)}, "
                f"index_type={self.stats.index_type}, "
                f"device={self.device})")

    def debug_embedding_creation(self, chunks: List[Dict[str, Any]], batch_size: int = 32) -> Dict[str, Any]:
        """Debug version of embedding creation with detailed logging"""
        
        debug_info = {
            'input_chunks_count': len(chunks),
            'batch_size': batch_size,
            'processing_steps': [],
            'errors': [],
            'warnings': []
        }
        
        try:
            # Step 1: Input validation
            debug_info['processing_steps'].append('input_validation')
            if not chunks:
                debug_info['errors'].append('No chunks provided')
                return debug_info
            
            # Step 2: Content extraction
            debug_info['processing_steps'].append('content_extraction')
            texts = []
            empty_content_count = 0
            
            for i, chunk in enumerate(chunks):
                content = chunk.get("content", "")
                if not isinstance(content, str):
                    content = str(content) if content is not None else ""
                    debug_info['warnings'].append(f'Non-string content in chunk {i}')
                
                if not content.strip():
                    empty_content_count += 1
                    content = f"[Empty chunk {i}]"
                
                texts.append(content)
            
            debug_info['extracted_texts_count'] = len(texts)
            debug_info['empty_content_count'] = empty_content_count
            
            # Step 3: Batch processing simulation
            debug_info['processing_steps'].append('batch_processing')
            total_batches = (len(texts) + batch_size - 1) // batch_size
            debug_info['total_batches'] = total_batches
            
            batch_info = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                batch_data = {
                    'batch_number': batch_num,
                    'batch_size': len(batch_texts),
                    'sample_text': batch_texts[0][:100] + '...' if batch_texts[0] else 'empty'
                }
                
                # Test encoding for this batch
                try:
                    test_embedding = self.encoder.encode([batch_texts[0]], show_progress_bar=False)
                    batch_data['encoding_test'] = 'success'
                    batch_data['embedding_shape'] = test_embedding.shape
                except Exception as e:
                    batch_data['encoding_test'] = 'failed'
                    batch_data['encoding_error'] = str(e)
                    debug_info['errors'].append(f'Batch {batch_num} encoding test failed: {str(e)}')
                
                batch_info.append(batch_data)
            
            debug_info['batch_info'] = batch_info
            debug_info['processing_steps'].append('debug_complete')
            
            return debug_info
            
        except Exception as e:
            debug_info['errors'].append(f'Debug process failed: {str(e)}')
            return debug_info