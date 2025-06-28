"""
Configuration settings for the Document Conversational Agent
"""

# Document Processing Settings
CHUNK_SIZE = 800  # Size of text chunks in characters
CHUNK_OVERLAP = 150  # Overlap between chunks
MIN_CHUNK_SIZE = 50  # Minimum chunk size to keep
MIN_WORDS_PER_CHUNK = 10  # Minimum words per chunk

# Vector Store Settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Sentence transformer model
SIMILARITY_THRESHOLD = 0.5  # Minimum similarity score for search results (lowered from 0.65)
TOP_K_RESULTS = 5  # Number of top results to return
INITIAL_SEARCH_MULTIPLIER = 3  # Get more results initially for filtering

# OpenAI Settings
OPENAI_MODEL = "gpt-4o"  # OpenAI model to use for answer generation
OPENAI_MAX_TOKENS = 1000  # Maximum tokens for OpenAI response
OPENAI_TEMPERATURE = 0.3  # Temperature for OpenAI response (0.0 = deterministic, 1.0 = creative)
OPENAI_TIMEOUT = 30  # Timeout for OpenAI API calls in seconds
USE_OPENAI_BY_DEFAULT = True  # Whether to use OpenAI if API key is available

# Quality Filtering Settings
MIN_CONTENT_LENGTH = 20  # Minimum content length for quality check
MIN_WORDS_FOR_QUALITY = 10  # Minimum words for quality content
MIN_SENTENCE_LENGTH = 5  # Minimum words per sentence
MIN_RELEVANCE_SCORE = 0.3  # Minimum relevance score for sentence inclusion
MIN_SIMILARITY_FOR_ANSWER = 0.7  # Minimum similarity for answer generation

# Text Cleaning Settings
REMOVE_URLS = True  # Remove URLs from text
REMOVE_AUTHOR_PATTERNS = True  # Remove author name patterns
REMOVE_PAGE_NUMBERS = True  # Remove standalone page numbers
PRESERVE_PARAGRAPHS = True  # Preserve paragraph breaks

# Noise Pattern Filters
NOISE_PATTERNS = [
    r'^\s*[A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+\s*$',  # Just names
    r'^\s*https?://',  # Just URLs
    r'^\s*www\.',      # Just www links
    r'^\s*\d+\s*$',    # Just numbers
    r'^\s*[A-Z\s]+\s*$',  # Just uppercase words
]

# Question Classification Keywords
QUESTION_TYPES = {
    'definition': ['what is', 'define', 'definition', 'meaning', 'what are'],
    'process': ['how', 'process', 'steps', 'method', 'procedure'],
    'explanation': ['why', 'reason', 'cause', 'because', 'explain'],
    'temporal': ['when', 'time', 'date', 'period', 'duration'],
    'location': ['where', 'location', 'place', 'site'],
    'person': ['who', 'person', 'author', 'creator', 'developer']
}

# Conversation Settings
MAX_CONVERSATION_HISTORY = 20  # Maximum conversation history entries
MAX_CONTEXT_LENGTH = 4000  # Maximum context length for LLM

# Performance Settings
BATCH_SIZE = 32  # Batch size for embedding generation
SHOW_PROGRESS_BAR = True  # Show progress bar during processing

# Logging Settings
LOG_LEVEL = "INFO"  # Logging level
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# File Processing Settings
SUPPORTED_FORMATS = {
    '.pdf': 'PDF documents',
    '.docx': 'Word documents',
    '.doc': 'Word documents (legacy)',
    '.txt': 'Text files',
    '.md': 'Markdown files',
    '.jpg': 'JPEG images',
    '.jpeg': 'JPEG images',
    '.png': 'PNG images',
    '.tiff': 'TIFF images'
}

# Error Messages
ERROR_MESSAGES = {
    'no_relevant_info': """I couldn't find any relevant information in the documents to answer your question. This could mean:

1. The documents don't contain information about this topic
2. The question might need to be rephrased
3. The documents haven't been uploaded yet

Please try:
- Rephrasing your question with different keywords
- Asking about a different topic that might be covered in your documents
- Checking if your documents have been successfully uploaded""",
    
    'low_quality_content': "I found some content in the documents, but it doesn't seem to directly answer your question. The content might be related but not specific enough to provide a clear answer.",
    
    'processing_error': "An error occurred while processing your question. Please try again or contact support if the problem persists.",
    
    'openai_error': "There was an issue with the AI service. I've fallen back to the local answer generation method."
} 