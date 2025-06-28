import os
import logging
from typing import List, Dict, Any
from pathlib import Path
import pdfplumber
from docx import Document
import pytesseract
from PIL import Image
from unstructured.partition.auto import partition
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document processing for various file formats"""
    
    def __init__(self):
        self.supported_formats = {
            '.pdf': self._process_pdf,
            '.docx': self._process_docx,
            '.doc': self._process_docx,
            '.txt': self._process_txt,
            '.md': self._process_txt,
            '.jpg': self._process_image,
            '.jpeg': self._process_image,
            '.png': self._process_image,
            '.tiff': self._process_image,
        }
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Process a document and extract its content
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            file_extension = file_path.suffix.lower()
            if file_extension not in self.supported_formats:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            logger.info(f"Processing document: {file_path}")
            
            # Extract text using appropriate method
            text = self.supported_formats[file_extension](file_path)
            
            # Clean and preprocess text
            cleaned_text = self._clean_text(text)
            
            # Extract metadata
            metadata = self._extract_metadata(file_path, cleaned_text)
            
            return {
                'text': cleaned_text,
                'metadata': metadata,
                'file_path': str(file_path),
                'file_size': file_path.stat().st_size,
                'file_extension': file_extension
            }
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            raise
    
    def _process_pdf(self, file_path: Path) -> str:
        """Extract text from PDF files"""
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            logger.warning(f"PDF processing failed, trying unstructured: {e}")
            # Fallback to unstructured
            elements = partition(str(file_path))
            text = "\n".join([str(element) for element in elements])
        
        return text
    
    def _process_docx(self, file_path: Path) -> str:
        """Extract text from DOCX files"""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error processing DOCX file: {e}")
            raise
    
    def _process_txt(self, file_path: Path) -> str:
        """Extract text from plain text files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read()
    
    def _process_image(self, file_path: Path) -> str:
        """Extract text from images using OCR"""
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            logger.error(f"Error processing image with OCR: {e}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace but preserve paragraph breaks
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove common PDF artifacts and noise
        text = re.sub(r'https?://[^\s]+', '', text)  # Remove URLs
        text = re.sub(r'www\.[^\s]+', '', text)      # Remove www links
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\n]', '', text)  # Keep more punctuation
        
        # Remove common header/footer patterns
        text = re.sub(r'Page \d+ of \d+', '', text)
        text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)  # Remove standalone page numbers
        
        # Remove author/editor patterns that often appear in academic papers
        text = re.sub(r'[A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+', '', text)  # Remove 3-word name patterns
        text = re.sub(r'[A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+', '', text)      # Remove "First M. Last" patterns
        
        # Normalize line breaks
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove empty lines and normalize spacing
        lines = []
        for line in text.split('\n'):
            line = line.strip()
            if line and len(line) > 3:  # Only keep lines with meaningful content
                lines.append(line)
        
        text = '\n'.join(lines)
        
        # Remove excessive periods and normalize sentence endings
        text = re.sub(r'\.{3,}', '...', text)
        text = re.sub(r'\.{2,}', '.', text)
        
        return text.strip()
    
    def _extract_metadata(self, file_path: Path, text: str) -> Dict[str, Any]:
        """Extract metadata from the document"""
        metadata = {
            'filename': file_path.name,
            'file_size_bytes': file_path.stat().st_size,
            'word_count': len(text.split()),
            'character_count': len(text),
            'line_count': len(text.split('\n')),
        }
        
        # Try to extract title from first few lines
        lines = text.split('\n')
        if lines:
            # Use first non-empty line as potential title
            for line in lines[:5]:
                if line.strip() and len(line.strip()) > 10:
                    metadata['potential_title'] = line.strip()[:100]
                    break
        
        return metadata
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats"""
        return list(self.supported_formats.keys()) 