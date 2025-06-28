import os
import pickle
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import chromadb
from chromadb.config import Settings
import re
from config import (
    EMBEDDING_MODEL, SIMILARITY_THRESHOLD, TOP_K_RESULTS, 
    INITIAL_SEARCH_MULTIPLIER, MIN_CONTENT_LENGTH, MIN_WORDS_FOR_QUALITY,
    NOISE_PATTERNS, BATCH_SIZE, SHOW_PROGRESS_BAR
)

logger = logging.getLogger(__name__)

class VectorStore:
    """Manages vector embeddings and similarity search"""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL, storage_path: str = "./vector_store"):
        """
        Initialize the vector store
        
        Args:
            model_name: Name of the sentence transformer model to use
            storage_path: Path to store vector embeddings
        """
        self.model_name = model_name
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Initialize sentence transformer model
        logger.info(f"Loading sentence transformer model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        
        # Initialize FAISS index
        self.faiss_index = None
        self.chunk_documents = []
        self.document_metadata = {}
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.storage_path / "chroma_db"),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="document_chunks",
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info("Vector store initialized successfully")
    
    def add_documents(self, chunks: List[Dict[str, Any]]) -> bool:
        """
        Add document chunks to the vector store
        
        Args:
            chunks: List of chunk documents with content and metadata
            
        Returns:
            True if successful, False otherwise
        """
        if not chunks:
            logger.warning("No chunks provided to add to vector store")
            return False
        
        try:
            # Extract content and metadata
            contents = [chunk['content'] for chunk in chunks]
            metadatas = [chunk['metadata'] for chunk in chunks]
            ids = [f"chunk_{chunk['chunk_id']}_{chunk['metadata'].get('filename', 'unknown')}" 
                   for chunk in chunks]
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(contents)} chunks")
            embeddings = self.embedding_model.encode(
                contents, 
                show_progress_bar=SHOW_PROGRESS_BAR,
                batch_size=BATCH_SIZE
            )
            
            # Add to ChromaDB
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=contents,
                metadatas=metadatas,
                ids=ids
            )
            
            # Update FAISS index
            self._update_faiss_index(embeddings, chunks)
            
            logger.info(f"Successfully added {len(chunks)} chunks to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            return False
    
    def _update_faiss_index(self, embeddings: np.ndarray, chunks: List[Dict[str, Any]]):
        """Update FAISS index with new embeddings"""
        if self.faiss_index is None:
            # Initialize FAISS index
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Add embeddings to FAISS index
        self.faiss_index.add(embeddings.astype('float32'))
        
        # Store chunk documents and metadata
        self.chunk_documents.extend(chunks)
        for chunk in chunks:
            self.document_metadata[chunk['chunk_id']] = chunk['metadata']
    
    def search(self, query: str, top_k: int = TOP_K_RESULTS, similarity_threshold: float = SIMILARITY_THRESHOLD) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score threshold
            
        Returns:
            List of similar documents with scores
        """
        if not query.strip():
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Search in ChromaDB with higher initial results for filtering
            initial_results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=min(top_k * INITIAL_SEARCH_MULTIPLIER, 20),  # Get more results for filtering
                include=['documents', 'metadatas', 'distances']
            )
            
            # Process and filter results
            search_results = []
            logger.info(f"Initial results from ChromaDB: {len(initial_results['documents'][0])}")
            
            for i in range(len(initial_results['documents'][0])):
                distance = initial_results['distances'][0][i]
                # Convert distance to similarity score (1 - distance for cosine similarity)
                similarity_score = 1 - distance
                
                content = initial_results['documents'][0][i]
                metadata = initial_results['metadatas'][0][i]
                
                logger.debug(f"Result {i}: similarity={similarity_score:.3f}, threshold={similarity_threshold}")
                
                # Apply quality filters
                if (similarity_score >= similarity_threshold and 
                    self._is_quality_content(content, query)):
                    
                    search_results.append({
                        'content': content,
                        'metadata': metadata,
                        'similarity_score': similarity_score,
                        'distance': distance
                    })
                else:
                    logger.debug(f"Filtered out result {i}: similarity={similarity_score:.3f} < {similarity_threshold} or quality check failed")
            
            # Sort by similarity score (descending)
            search_results.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            # Limit to top_k results
            search_results = search_results[:top_k]
            
            logger.info(f"Found {len(search_results)} relevant chunks for query: {query}")
            if search_results:
                logger.info(f"Top similarity score: {search_results[0]['similarity_score']:.3f}")
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []
    
    def _is_quality_content(self, content: str, query: str) -> bool:
        """Check if content is relevant to the query"""
        if not content or len(content.strip()) < MIN_CONTENT_LENGTH:
            return False
        
        # Check for meaningful content (not just noise)
        words = content.split()
        if len(words) < MIN_WORDS_FOR_QUALITY:
            return False
        
        # Check for common noise patterns
        for pattern in NOISE_PATTERNS:
            if re.match(pattern, content.strip()):
                return False
        
        # Note: Removed exact query term matching requirement
        # Semantic similarity from embeddings should be sufficient
        # This allows for better handling of synonyms and related concepts
        
        return True
    
    def search_faiss(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search using FAISS index (alternative method)
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of similar documents with scores
        """
        if self.faiss_index is None or not self.chunk_documents:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Search in FAISS
            scores, indices = self.faiss_index.search(
                query_embedding.astype('float32'), 
                min(top_k, len(self.chunk_documents))
            )
            
            # Process results
            search_results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.chunk_documents):
                    chunk = self.chunk_documents[idx]
                    search_results.append({
                        'content': chunk['content'],
                        'metadata': chunk['metadata'],
                        'similarity_score': float(score),
                        'chunk_id': chunk['chunk_id']
                    })
            
            # Sort by similarity score (descending)
            search_results.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching FAISS index: {e}")
            return []
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about stored documents"""
        try:
            collection_count = self.collection.count()
            faiss_count = self.faiss_index.ntotal if self.faiss_index else 0
            
            return {
                'chroma_documents': collection_count,
                'faiss_vectors': faiss_count,
                'total_chunks': len(self.chunk_documents),
                'model_name': self.model_name,
                'storage_path': str(self.storage_path)
            }
        except Exception as e:
            logger.error(f"Error getting document stats: {e}")
            return {}
    
    def clear_all(self):
        """Clear all stored documents and embeddings"""
        try:
            # Clear ChromaDB collection
            self.chroma_client.delete_collection("document_chunks")
            self.collection = self.chroma_client.create_collection(
                name="document_chunks",
                metadata={"hnsw:space": "cosine"}
            )
            
            # Clear FAISS index
            self.faiss_index = None
            self.chunk_documents = []
            self.document_metadata = {}
            
            logger.info("Vector store cleared successfully")
            
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
    
    def save_index(self, filepath: str):
        """Save FAISS index to file"""
        if self.faiss_index is not None:
            try:
                faiss.write_index(self.faiss_index, filepath)
                logger.info(f"FAISS index saved to {filepath}")
            except Exception as e:
                logger.error(f"Error saving FAISS index: {e}")
    
    def load_index(self, filepath: str):
        """Load FAISS index from file"""
        try:
            self.faiss_index = faiss.read_index(filepath)
            logger.info(f"FAISS index loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}") 