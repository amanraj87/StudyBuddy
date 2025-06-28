# Document Conversational Agent

A powerful conversational agent that can read, understand, and answer questions about any document you upload. Built with modern AI technologies including RAG (Retrieval-Augmented Generation), vector embeddings, semantic search, and OpenAI GPT-4o integration for enhanced answer generation.

## Features

- **Multi-format Document Support**: PDF, DOCX, TXT, MD, JPG, PNG, TIFF
- **Advanced Text Processing**: OCR for images, intelligent text chunking with comprehensive quality filtering
- **Semantic Search**: Find relevant information using vector embeddings with enhanced similarity matching (threshold: 0.65)
- **Intelligent Q&A**: Natural language interface with OpenAI GPT-4o integration for superior answer generation
- **Hybrid Answer Generation**: Uses GPT-4o when available, falls back to local RAG system
- **Voice Mode**: Speech-to-text input and text-to-speech output for hands-free interaction
- **Modern Web UI**: Responsive design with drag-and-drop upload
- **Real-time Statistics**: Track documents, chunks, and conversations
- **Conversation History**: Maintain context across multiple questions
- **Quality Assurance**: Multi-stage content filtering to ensure relevant and meaningful responses
- **Easy Startup**: Single command to launch both backend and frontend servers

## Voice Mode Features

### Speech Input

- **Click to Speak**: Click the microphone button to start voice input
- **Automatic Processing**: Voice input is automatically converted to text and sent as a question
- **Visual Feedback**: Real-time visual indicators show when the system is listening
- **Error Handling**: Graceful fallback with helpful error messages for unsupported browsers

### Speech Output

- **Text-to-Speech**: Assistant responses are spoken aloud when voice output is enabled
- **Smart Text Cleaning**: Markdown and formatting are removed for natural speech
- **Voice Selection**: Automatically selects the best available voice for your system
- **Adjustable Settings**: Control speech rate, pitch, and volume

### How to Use Voice Mode

1. **Enable Voice Mode**: Click the "Voice Mode" button in the chat controls
2. **Voice Input**: Click the microphone button and speak your question
3. **Voice Output**: Toggle the speaker button to enable/disable spoken responses
4. **Mixed Mode**: Use both voice input and text input as needed

### Browser Compatibility

Voice mode works best in modern browsers:

- **Chrome/Edge**: Full support for speech recognition and synthesis
- **Firefox**: Good support for speech synthesis, limited speech recognition
- **Safari**: Limited support, may require user interaction
- **Mobile Browsers**: Support varies by device and browser

**Note**: Speech recognition requires HTTPS in production environments.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   AI/ML Layer   │
│   (HTML/CSS/JS) │◄──►│   (FastAPI)     │◄──►│   (RAG System)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   Vector Store  │
                       │   (ChromaDB)    │
                       │ + Quality Filter│
                       └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   OpenAI GPT-4o │
                       │   (Optional)    │
                       └─────────────────┘
```

## Prerequisites

- Python 3.8 or higher
- uv (Python package manager) - recommended for faster dependency management
- Modern web browser
- At least 4GB RAM (for AI models)
- OpenAI API key (optional, for enhanced answer generation)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Personal_Project
```

### 2. Install Python Dependencies

```bash
# Using uv (recommended)
uv pip install -r backend/requirements.txt

# Or using pip
pip install -r backend/requirements.txt
```

### 3. Configure OpenAI (Optional but Recommended)

For enhanced answer generation, set up your OpenAI API key:

```bash
# Create a .env file in the backend directory
echo "OPENAI_API_KEY=your_openai_api_key_here" > backend/.env
```

**Benefits of OpenAI Integration:**

- More natural and coherent answers
- Better understanding of complex questions
- Improved context synthesis
- More accurate information extraction

**Cost Considerations:**

- GPT-4o costs approximately $0.005 per 1K input tokens and $0.015 per 1K output tokens
- Typical queries cost $0.01-$0.05 depending on document length
- System automatically falls back to local generation if API is unavailable

### 4. Install Additional Dependencies (Optional)

For better OCR support on Windows:

```bash
# Install Tesseract OCR
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

## Usage

### Quick Start (Recommended)

Use the startup script to launch both backend and frontend simultaneously:

```bash
# From the project root directory
uv run start.py
```

This will:

- Start the FastAPI backend server on `http://localhost:8000`
- Start the frontend HTTP server on `http://localhost:8080`
- Automatically open the frontend in your default web browser
- Display real-time status and URLs

### Manual Startup

If you prefer to start services manually:

#### 1. Start the Backend Server

```bash
cd backend
uv run main.py
```

The server will start on `http://localhost:8000`

#### 2. Open the Frontend

Open `frontend/index.html` in your web browser, or serve it using a local server:

```bash
cd frontend
uv run python -m http.server 8080
```

Then visit `http://localhost:8080`

### 3. Upload Documents

1. Drag and drop files or click "Choose Files"
2. Supported formats: PDF, DOCX, TXT, MD, JPG, PNG, TIFF
3. Wait for processing to complete

### 4. Ask Questions

1. Type your question in the chat input
2. Press Enter or click the send button
3. Get AI-powered answers based on your documents
4. Check the answer method indicator (OpenAI GPT-4o or Local RAG)

## Recent Updates

### Startup Script Improvements (Latest)

- **Fixed Path Resolution**: Resolved issues with frontend startup on Windows systems
- **Absolute Paths**: Improved subprocess handling with proper absolute path resolution
- **Better Error Handling**: Enhanced error messages and graceful fallbacks
- **Cross-Platform Compatibility**: Improved support for different operating systems
- **Automatic Browser Launch**: Frontend automatically opens in default browser

### Quality Improvements

The system includes comprehensive quality filtering and OpenAI integration for superior responses:

### Enhanced Text Processing

- **Smart Cleaning**: Removes noise (URLs, author names, page numbers) while preserving meaningful content and punctuation
- **Quality Chunking**: Uses RecursiveCharacterTextSplitter with semantic boundaries and quality scoring
- **Content Validation**: Validates chunk quality based on length, structure, and content diversity
- **Noise Filtering**: Removes low-quality content using pattern matching and quality thresholds

### Improved Search

- **Higher Similarity Threshold**: Increased from 0.5 to 0.65 for better relevance
- **Multi-stage Filtering**: Content validation against query terms and quality criteria
- **Quality Scoring**: Ranks results based on relevance, content quality, and similarity scores
- **Enhanced Retrieval**: Uses ChromaDB with cosine similarity and quality filtering

### Intelligent Answer Generation

- **OpenAI GPT-4o Integration**: Uses GPT-4o for superior answer generation when available
- **Hybrid System**: Automatically falls back to local RAG if OpenAI is unavailable
- **Question Classification**: Categorizes questions (definition, process, explanation, temporal, location, person)
- **Relevance Scoring**: Extracts and scores relevant information based on question type
- **Structured Responses**: Creates context-aware answers with proper formatting
- **Confidence Calculation**: Provides confidence scores based on similarity and result quality
- **Better Error Handling**: Provides helpful guidance when no relevant content is found

### Configuration Options

All quality settings can be adjusted in `backend/config.py`:

- Similarity thresholds (default: 0.65)
- Chunk size and overlap (default: 800/150)
- Quality filtering criteria
- Text cleaning options
- Question classification keywords
- OpenAI settings (model, temperature, max tokens)

## API Endpoints

### Document Management

- `POST /upload` - Upload a document
- `DELETE /documents` - Clear all documents

### Chat

- `POST /ask` - Ask a question
- `GET /conversation-history` - Get chat history
- `DELETE /conversation-history` - Clear chat history

### Statistics

- `GET /stats` - Get document statistics
- `GET /supported-formats` - Get supported file formats

### Health

- `GET /health` - Health check
- `GET /` - API information

## Configuration

### Environment Variables

Create a `.env` file in the backend directory:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Vector Store Configuration
VECTOR_STORE_PATH=./vector_store
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Document Processing
CHUNK_SIZE=800
CHUNK_OVERLAP=150
SIMILARITY_THRESHOLD=0.65
TOP_K_RESULTS=5
```

### Quality Settings

Adjust quality parameters in `backend/config.py`:

```python
# OpenAI Settings
OPENAI_MODEL = "gpt-4o"
OPENAI_MAX_TOKENS = 1000
OPENAI_TEMPERATURE = 0.3

# Increase for stricter matching
SIMILARITY_THRESHOLD = 0.7

# Adjust chunk size for better context
CHUNK_SIZE = 1000

# Modify quality filtering
MIN_CHUNK_SIZE = 100
MIN_WORDS_PER_CHUNK = 15
```

## How It Works

### 1. Document Processing

- **Text Extraction**: Extract text from various file formats using appropriate libraries
- **OCR Processing**: Convert images to text using Tesseract OCR
- **Smart Cleaning**: Remove noise while preserving meaningful content and structure
- **Quality Validation**: Ensure extracted content meets quality standards

### 2. Text Chunking

- **Semantic Splitting**: Split documents using RecursiveCharacterTextSplitter with semantic separators
- **Quality Filtering**: Remove low-quality chunks using multiple criteria (length, word count, noise patterns)
- **Quality Scoring**: Calculate quality scores (0.0-1.0) based on length, structure, and content diversity
- **Overlap Strategy**: Maintain context between chunks with configurable overlap
- **Metadata Preservation**: Keep document information and chunk positioning data

### 3. Vector Embeddings

- **Sentence Transformers**: Convert text to high-dimensional vectors using all-MiniLM-L6-v2
- **Similarity Search**: Find relevant chunks using cosine similarity with quality filtering
- **Multi-stage Filtering**: Apply content validation and quality checks to search results
- **Vector Database**: Store embeddings in ChromaDB for fast retrieval

### 4. RAG (Retrieval-Augmented Generation)

- **Query Processing**: Convert questions to embeddings and build context from conversation history
- **Context Retrieval**: Find most relevant document chunks with similarity threshold filtering
- **Answer Generation**:
  - **Primary**: Use OpenAI GPT-4o for superior understanding and response generation
  - **Fallback**: Use local RAG system with question classification and structured answers
- **Content Extraction**: Extract relevant information based on question type and relevance scoring
- **Confidence Scoring**: Calculate confidence based on similarity scores and result quality

## Performance Tips

### For Large Documents

- Increase `CHUNK_SIZE` for better context (default: 800)
- Use `all-mpnet-base-v2` model for better quality (requires more memory)
- Consider document preprocessing for better chunking

### For Better Accuracy

- Enable OpenAI GPT-4o integration for superior answers
- Increase `SIMILARITY_THRESHOLD` for stricter matching (default: 0.65)
- Increase `TOP_K_RESULTS` for more context (default: 5)
- Use conversation history for context continuity
- Adjust quality filtering parameters in config.py

### For Faster Processing

- Use `all-MiniLM-L6-v2` model (default)
- Reduce `CHUNK_SIZE` for faster processing
- Limit document uploads to essential files
- Disable progress bars in config.py

### For Cost Optimization

- Use local RAG system for simple queries
- Enable OpenAI only for complex questions
- Monitor API usage in OpenAI dashboard
- Set appropriate token limits in config.py

## Troubleshooting

### Startup Script Issues

1. **Frontend Not Starting**

   - **Error**: "Failed to open frontend: [WinError 2] The system cannot find the file specified"
   - **Solution**: This issue has been fixed in the latest update. The startup script now uses absolute paths and proper subprocess handling
   - **Alternative**: If issues persist, use manual startup method described above

2. **Port Already in Use**

   - **Error**: "Address already in use" or "Port 8000/8080 is already in use"
   - **Solution**:

     ```bash
     # Kill processes using the ports
     # On Windows:
     netstat -ano | findstr :8000
     taskkill /PID <PID> /F

     # On Linux/Mac:
     lsof -ti:8000 | xargs kill -9
     ```

3. **Python Path Issues**

   - **Error**: "python: command not found" or similar
   - **Solution**: Ensure Python is in your PATH or use the full path:

     ```bash
     # Use uv (recommended)
     uv run start.py

     # Or use full Python path
     /path/to/python start.py
     ```

### Common Issues

1. **Poor Quality Results**

   - Check similarity threshold settings in config.py
   - Verify document quality and format
   - Try rephrasing questions with different keywords
   - Enable OpenAI integration for better answers

2. **OpenAI API Issues**

   - Verify API key is correctly set in .env file
   - Check OpenAI API status and rate limits
   - System automatically falls back to local generation
   - Monitor API usage and costs

3. **Import Errors**

   - Ensure all dependencies are installed: `uv pip install -r backend/requirements.txt`
   - Check Python version (3.8+ required)
   - Verify virtual environment activation

4. **Memory Issues**

   - Reduce chunk size in config.py
   - Use smaller embedding model
   - Limit number of concurrent document uploads

5. **OCR Issues**

   - Install Tesseract OCR for image processing
   - Ensure image quality is sufficient
   - Check image format support

6. **FAISS Warnings**

   - **Warning**: "Could not load library with AVX2 support"
   - **Impact**: Minimal - system falls back to standard FAISS
   - **Solution**: This is normal and doesn't affect functionality

### Getting Help

- Check the logs in the backend console for detailed error messages
- Verify configuration settings in `backend/config.py`
- Ensure all required dependencies are properly installed
- Test with simple text files first before trying complex documents
- Monitor OpenAI API usage and costs
- Use the startup script for easier debugging and automatic service management

## Technical Details

### Quality Filtering System

The system implements a comprehensive quality filtering pipeline:

1. **Text Cleaning**: Removes noise while preserving meaningful content
2. **Chunk Quality**: Filters chunks based on length, word count, and structure
3. **Search Quality**: Applies similarity thresholds and content validation
4. **Answer Quality**: Uses OpenAI GPT-4o or local RAG with question classification

### OpenAI Integration

- **Model**: GPT-4o for superior understanding and generation
- **System Prompt**: Carefully crafted to ensure accurate, document-based answers
- **Fallback**: Automatic fallback to local RAG system
- **Cost Control**: Configurable token limits and temperature settings

### Supported File Formats

- **PDF**: Using pdfplumber and unstructured libraries
- **DOCX**: Using python-docx library
- **TXT/MD**: Direct text processing
- **Images**: OCR using Tesseract and PIL

### Vector Store

- **Database**: ChromaDB with persistent storage
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Similarity**: Cosine similarity with quality filtering
- **Indexing**: FAISS for fast similarity search

## License

This project is licensed under the MIT License - see the LICENSE file for details.
