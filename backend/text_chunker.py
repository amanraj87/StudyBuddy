import re
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
import logging
from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE, MIN_WORDS_PER_CHUNK, NOISE_PATTERNS

logger = logging.getLogger(__name__)

class TextChunker:
    """Handles text chunking for document processing"""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        """
        Initialize the text chunker
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize text splitters
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        
        self.token_splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split text into chunks with metadata
        
        Args:
            text: The text to chunk
            metadata: Additional metadata to include with each chunk
            
        Returns:
            List of dictionaries containing chunks and their metadata
        """
        if not text or not text.strip():
            return []
        
        try:
            # Pre-process text to improve chunking
            text = self._preprocess_text(text)
            
            # Use recursive splitter for better semantic boundaries
            chunks = self.recursive_splitter.split_text(text)
            
            # Filter and quality-check chunks
            quality_chunks = []
            for i, chunk in enumerate(chunks):
                if self._is_quality_chunk(chunk):
                    chunk_doc = {
                        'content': chunk.strip(),
                        'chunk_id': i,
                        'chunk_size': len(chunk),
                        'metadata': metadata.copy() if metadata else {}
                    }
                    
                    # Add chunk-specific metadata
                    chunk_doc['metadata'].update({
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'chunk_start_char': self._find_chunk_start(text, chunk, i, chunks),
                        'quality_score': self._calculate_chunk_quality(chunk)
                    })
                    
                    quality_chunks.append(chunk_doc)
            
            logger.info(f"Created {len(quality_chunks)} quality chunks from {len(chunks)} total chunks")
            return quality_chunks
            
        except Exception as e:
            logger.error(f"Error chunking text: {e}")
            # Fallback to simple splitting
            return self._fallback_chunking(text, metadata)
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text to improve chunking quality"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Ensure proper sentence endings
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        # Add line breaks before common section headers
        text = re.sub(r'([.!?])\s*([A-Z][a-z]+:)', r'\1\n\n\2', text)
        
        return text.strip()
    
    def _is_quality_chunk(self, chunk: str) -> bool:
        """Check if a chunk meets quality criteria"""
        if not chunk or len(chunk.strip()) < MIN_CHUNK_SIZE:
            return False
        
        # Check for meaningful content (not just noise)
        words = chunk.split()
        if len(words) < MIN_WORDS_PER_CHUNK:
            return False
        
        # Check for common noise patterns
        for pattern in NOISE_PATTERNS:
            if re.match(pattern, chunk.strip()):
                return False
        
        # Check for meaningful sentence structure
        sentences = chunk.split('. ')
        meaningful_sentences = 0
        for sentence in sentences:
            if len(sentence.split()) >= 5:  # At least 5 words per sentence
                meaningful_sentences += 1
        
        return meaningful_sentences >= 1
    
    def _calculate_chunk_quality(self, chunk: str) -> float:
        """Calculate a quality score for a chunk (0.0 to 1.0)"""
        if not chunk:
            return 0.0
        
        score = 0.0
        
        # Length score (prefer chunks with reasonable length)
        length = len(chunk)
        if 100 <= length <= 2000:
            score += 0.3
        elif 50 <= length < 100 or 2000 < length <= 3000:
            score += 0.2
        else:
            score += 0.1
        
        # Word count score
        words = chunk.split()
        if 20 <= len(words) <= 300:
            score += 0.3
        elif 10 <= len(words) < 20 or 300 < len(words) <= 500:
            score += 0.2
        else:
            score += 0.1
        
        # Sentence structure score
        sentences = chunk.split('. ')
        if len(sentences) >= 2:
            score += 0.2
        elif len(sentences) == 1:
            score += 0.1
        
        # Content diversity score (avoid repetitive content)
        unique_words = set(word.lower() for word in words if len(word) > 2)
        if len(unique_words) >= len(words) * 0.6:  # At least 60% unique words
            score += 0.2
        else:
            score += 0.1
        
        return min(score, 1.0)
    
    def _find_chunk_start(self, full_text: str, chunk: str, chunk_index: int, all_chunks: List[str]) -> int:
        """Find the starting character position of a chunk in the original text"""
        if chunk_index == 0:
            return 0
        
        # Find the position of this chunk in the original text
        start_pos = full_text.find(chunk)
        if start_pos != -1:
            return start_pos
        
        # If not found, estimate based on previous chunks
        estimated_start = sum(len(c) for c in all_chunks[:chunk_index])
        return estimated_start
    
    def _fallback_chunking(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Fallback chunking method using simple sentence splitting"""
        # Split by sentences first
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        chunk_id = 0
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append({
                        'content': current_chunk.strip(),
                        'chunk_id': chunk_id,
                        'chunk_size': len(current_chunk),
                        'metadata': metadata.copy() if metadata else {}
                    })
                    chunk_id += 1
                current_chunk = sentence + ". "
        
        # Add the last chunk
        if current_chunk:
            chunks.append({
                'content': current_chunk.strip(),
                'chunk_id': chunk_id,
                'chunk_size': len(current_chunk),
                'metadata': metadata.copy() if metadata else {}
            })
        
        return chunks
    
    def chunk_document(self, document_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a processed document
        
        Args:
            document_data: Dictionary containing document text and metadata
            
        Returns:
            List of chunked documents
        """
        text = document_data.get('text', '')
        metadata = document_data.get('metadata', {})
        
        # Add document-level metadata
        metadata.update({
            'file_path': document_data.get('file_path', ''),
            'file_extension': document_data.get('file_extension', ''),
            'file_size': document_data.get('file_size', 0)
        })
        
        return self.chunk_text(text, metadata)
    
    def merge_small_chunks(self, chunks: List[Dict[str, Any]], min_size: int = 100) -> List[Dict[str, Any]]:
        """
        Merge chunks that are too small
        
        Args:
            chunks: List of chunk documents
            min_size: Minimum chunk size in characters
            
        Returns:
            List of merged chunks
        """
        if not chunks:
            return []
        
        merged_chunks = []
        current_chunk = chunks[0].copy()
        
        for chunk in chunks[1:]:
            if len(current_chunk['content']) < min_size:
                # Merge with next chunk
                current_chunk['content'] += "\n\n" + chunk['content']
                current_chunk['chunk_size'] = len(current_chunk['content'])
                current_chunk['metadata']['total_chunks'] = current_chunk['metadata'].get('total_chunks', 1) + 1
            else:
                # Current chunk is big enough, save it and start new one
                merged_chunks.append(current_chunk)
                current_chunk = chunk.copy()
        
        # Add the last chunk
        merged_chunks.append(current_chunk)
        
        # Update chunk IDs
        for i, chunk in enumerate(merged_chunks):
            chunk['chunk_id'] = i
        
        return merged_chunks 