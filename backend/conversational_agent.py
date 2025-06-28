import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
from pathlib import Path
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import our custom modules
from document_processor import DocumentProcessor
from text_chunker import TextChunker
from vector_store import VectorStore

logger = logging.getLogger(__name__)

class ConversationalAgent:
    """Conversational agent that can answer questions about documents using RAG with OpenAI"""
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 storage_path: str = "./vector_store",
                 openai_api_key: str = None,
                 use_openai: bool = True):
        """
        Initialize the conversational agent
        
        Args:
            embedding_model: Sentence transformer model name
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            storage_path: Path for vector store
            openai_api_key: OpenAI API key for GPT-4o integration
            use_openai: Whether to use OpenAI for answer generation
        """
        self.document_processor = DocumentProcessor()
        self.text_chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.vector_store = VectorStore(model_name=embedding_model, storage_path=storage_path)
        
        # OpenAI configuration
        self.use_openai = use_openai
        if openai_api_key:
            openai.api_key = openai_api_key
        elif use_openai:
            # Try to get from environment variable (loaded from .env file)
            openai.api_key = os.getenv('OPENAI_API_KEY')
            if not openai.api_key:
                logger.warning("OpenAI API key not found in .env file. Falling back to local answer generation.")
                self.use_openai = False
            else:
                logger.info("OpenAI API key loaded from .env file successfully.")
        
        # Conversation history
        self.conversation_history = []
        
        # Configuration
        self.max_context_length = 4000
        self.similarity_threshold = 0.5
        self.top_k_results = 5
        
        logger.info(f"Conversational agent initialized successfully. OpenAI enabled: {self.use_openai}")
    
    def add_document(self, file_path: str) -> Dict[str, Any]:
        """
        Add a document to the agent's knowledge base
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary with processing results
        """
        try:
            logger.info(f"Adding document: {file_path}")
            
            # Process the document
            document_data = self.document_processor.process_document(file_path)
            
            # Chunk the document
            chunks = self.text_chunker.chunk_document(document_data)
            
            # Merge small chunks if needed
            if len(chunks) > 1:
                chunks = self.text_chunker.merge_small_chunks(chunks, min_size=200)
            
            # Add to vector store
            success = self.vector_store.add_documents(chunks)
            
            if success:
                result = {
                    'success': True,
                    'filename': document_data['metadata']['filename'],
                    'chunks_created': len(chunks),
                    'word_count': document_data['metadata']['word_count'],
                    'file_size': document_data['file_size'],
                    'processing_time': datetime.now().isoformat()
                }
                logger.info(f"Document added successfully: {result}")
                return result
            else:
                return {
                    'success': False,
                    'error': 'Failed to add document to vector store',
                    'filename': Path(file_path).name
                }
                
        except Exception as e:
            logger.error(f"Error adding document {file_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'filename': Path(file_path).name
            }
    
    def ask_question(self, question: str, use_history: bool = True) -> Dict[str, Any]:
        """
        Ask a question about the documents
        
        Args:
            question: The question to ask
            use_history: Whether to include conversation history
            
        Returns:
            Dictionary with answer and context
        """
        try:
            if not question.strip():
                return {
                    'success': False,
                    'error': 'Question cannot be empty'
                }
            
            logger.info(f"Processing question: {question}")
            
            # Build context from conversation history if enabled
            context_query = question
            if use_history and self.conversation_history:
                # Add recent conversation context
                recent_context = self._build_context_from_history()
                context_query = f"{recent_context}\n\nQuestion: {question}"
            
            # Classify question type for adaptive similarity threshold
            question_type = self._classify_question(question.lower())
            
            # Use lower threshold for definition questions to be more inclusive
            adaptive_threshold = 0.4 if question_type == 'definition' else self.similarity_threshold
            
            logger.info(f"Question type: {question_type}, Using similarity threshold: {adaptive_threshold}")
            
            # Search for relevant document chunks with adaptive threshold
            search_results = self.vector_store.search(
                query=context_query,
                top_k=self.top_k_results,
                similarity_threshold=adaptive_threshold
            )
            
            if not search_results:
                return {
                    'success': True,
                    'answer': f"I couldn't find any relevant information about '{question}' in the documents. Here are some suggestions:\n\n**Try these approaches:**\n• **Rephrase your question** with different keywords (e.g., 'neural networks' instead of 'RNN')\n• **Ask a broader question** about the topic\n• **Check if your documents have been uploaded** successfully\n• **Try asking about specific aspects** mentioned in your documents\n\n**Example questions you could try:**\n• What are neural networks?\n• How do neural networks work?\n• What types of neural networks are there?\n• What is machine learning?",
                    'context_used': [],
                    'confidence': 0.0,
                    'search_results_count': 0,
                    'answer_method': 'no_results',
                    'metadata': {
                        'top_similarity_score': 0,
                        'average_similarity_score': 0
                    }
                }
            
            # Generate answer using OpenAI GPT-4o if available, otherwise fallback to local method
            if self.use_openai and openai.api_key:
                answer = self._generate_openai_answer(question, search_results)
                answer_method = 'openai_gpt4o'
            else:
                answer = self._generate_improved_answer(question, self._build_context_from_search_results(search_results), search_results, question_type)
                answer_method = 'local_rag'
            
            # Calculate confidence based on similarity scores
            confidence = self._calculate_confidence(search_results)
            
            # Store in conversation history
            conversation_entry = {
                'timestamp': datetime.now().isoformat(),
                'question': question,
                'answer': answer,
                'context_used': [result['content'][:200] + "..." for result in search_results] if search_results else [],
                'confidence': confidence,
                'search_results_count': len(search_results),
                'answer_method': answer_method
            }
            self.conversation_history.append(conversation_entry)
            
            # Limit conversation history
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            
            return {
                'success': True,
                'answer': answer,
                'context_used': [result['content'][:200] + "..." for result in search_results] if search_results else [],
                'confidence': confidence,
                'search_results_count': len(search_results),
                'answer_method': answer_method,
                'metadata': {
                    'top_similarity_score': search_results[0]['similarity_score'] if search_results else 0,
                    'average_similarity_score': sum(r['similarity_score'] for r in search_results) / len(search_results) if search_results else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return {
                'success': False,
                'error': f"An error occurred while processing your question: {str(e)}"
            }
    
    def _generate_openai_answer(self, question: str, search_results: List[Dict[str, Any]]) -> str:
        """
        Generate answer using OpenAI
        
        Args:
            question: The user's question
            search_results: Relevant document chunks
            
        Returns:
            Generated answer from OpenAI
        """
        try:
            # Check if we have search results
            if not search_results:
                return "I couldn't find any relevant information in the documents to answer your question. This could mean:\n\n1. The documents don't contain information about this topic\n2. The question might need to be rephrased\n3. The documents haven't been uploaded yet\n\nPlease try:\n- Rephrasing your question with different keywords\n- Asking about a different topic that might be covered in your documents\n- Checking if your documents have been successfully uploaded"
            
            # Prepare context from search results
            context_parts = []
            for i, result in enumerate(search_results, 1):
                content = result['content']
                metadata = result['metadata']
                filename = metadata.get('filename', 'Unknown')
                similarity = result['similarity_score']
                
                context_parts.append(f"Document {i} ({filename}, similarity: {similarity:.2f}):\n{content}\n")
            
            context = "\n".join(context_parts)
            
            # Create system prompt
            system_prompt = """You are a helpful AI assistant that answers questions based on provided document content. 

Your task is to:
1. Understand the user's question
2. Analyze the provided document chunks
3. Generate a comprehensive, accurate answer based ONLY on the document content
4. If the documents don't contain enough information to answer the question, clearly state this
5. Provide specific references to the document content when possible
6. Be concise but thorough

IMPORTANT FORMATTING GUIDELINES:
- Use clear, concise language
- Break down complex concepts into digestible parts
- Use proper markdown formatting for structure and readability
- Use headers (##) to organize sections
- Use bullet points (•) for lists
- Use **bold text** for key terms and concepts
- Use proper paragraph breaks for readability
- Start with a brief definition or overview
- Organize information logically (general to specific, or chronological)
- End with a brief summary or conclusion when appropriate

MARKDOWN STRUCTURE EXAMPLES:
- Use ## for main sections
- Use ### for subsections
- Use bullet points for lists
- Use **bold** for emphasis
- Use proper spacing between sections

Content guidelines:
- Only use information from the provided document chunks
- If you're unsure about something, say so
- Don't make up information not present in the documents
- Structure your answer logically and clearly
- If the question is unclear, ask for clarification
- Make the answer engaging and easy to follow"""

            # Create user message
            user_message = f"""Question: {question}

Document Content:
{context}

Please provide a comprehensive answer based on the document content above."""

            # Call OpenAI API
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=1000,
                temperature=0.3,  # Lower temperature for more focused answers
                timeout=30
            )
            
            answer = response.choices[0].message.content.strip()
            logger.info("Generated answer using OpenAI GPT-4o")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating OpenAI answer: {e}")
            # Fallback to local method
            logger.info("Falling back to local answer generation")
            return self._generate_improved_answer(question, self._build_context_from_search_results(search_results), search_results, question_type)
    
    def _build_context_from_search_results(self, search_results: List[Dict[str, Any]]) -> str:
        """Build context string from search results"""
        if not search_results:
            return ""
            
        context_parts = []
        
        for i, result in enumerate(search_results, 1):
            content = result['content']
            metadata = result['metadata']
            filename = metadata.get('filename', 'Unknown')
            
            context_parts.append(f"Document {i} ({filename}):\n{content}\n")
        
        return "\n".join(context_parts)
    
    def _build_context_from_history(self) -> str:
        """Build context from recent conversation history"""
        if not self.conversation_history:
            return ""
        
        # Get last 3 exchanges
        recent_history = self.conversation_history[-6:]  # Last 3 Q&A pairs
        
        context_parts = ["Recent conversation context:"]
        for entry in recent_history:
            context_parts.append(f"Q: {entry['question']}")
            context_parts.append(f"A: {entry['answer'][:300]}...")
        
        return "\n".join(context_parts)
    
    def _generate_improved_answer(self, question: str, context: str, search_results: List[Dict[str, Any]], question_type: str) -> str:
        """
        Generate improved answer using RAG approach with better content analysis
        """
        if not search_results:
            return "I don't have enough information to answer this question."
        
        # Extract relevant information based on question type
        relevant_info = self._extract_relevant_info(question, search_results, question_type)
        
        if not relevant_info:
            return "I found some content in the documents, but it doesn't seem to directly answer your question. The content might be related but not specific enough to provide a clear answer."
        
        # Generate structured answer
        answer = self._create_structured_answer(question, relevant_info, question_type)
        
        return answer
    
    def _classify_question(self, question: str) -> str:
        """Classify the type of question being asked"""
        question_lower = question.lower()
        
        # Check for definition questions (what is X, define X, etc.)
        if (any(word in question_lower for word in ['what is', 'define', 'definition', 'meaning', 'what are']) or
            (question_lower.startswith('what is ') and len(question_lower.split()) <= 4)):
            return 'definition'
        elif any(word in question_lower for word in ['how', 'process', 'steps', 'method']):
            return 'process'
        elif any(word in question_lower for word in ['why', 'reason', 'cause', 'because']):
            return 'explanation'
        elif any(word in question_lower for word in ['when', 'time', 'date', 'period']):
            return 'temporal'
        elif any(word in question_lower for word in ['where', 'location', 'place']):
            return 'location'
        elif any(word in question_lower for word in ['who', 'person', 'author', 'creator']):
            return 'person'
        else:
            return 'general'
    
    def _extract_relevant_info(self, question: str, search_results: List[Dict[str, Any]], question_type: str) -> List[str]:
        """Extract relevant information based on question type and content"""
        relevant_info = []
        question_terms = question.lower().split()
        
        for result in search_results[:3]:  # Use top 3 results
            content = result['content']
            similarity_score = result['similarity_score']
            
            # Only use high-quality matches
            if similarity_score < 0.7:
                continue
            
            # Split content into sentences
            sentences = content.split('. ')
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 10:
                    continue
                
                # Check if sentence contains question terms
                sentence_lower = sentence.lower()
                matching_terms = sum(1 for term in question_terms if term in sentence_lower)
                
                # Score the sentence relevance
                relevance_score = matching_terms / len(question_terms) if question_terms else 0
                
                # Add sentence if it's relevant enough
                if relevance_score > 0.3 or len(sentence) > 50:  # Either relevant or substantial
                    relevant_info.append({
                        'text': sentence,
                        'relevance': relevance_score,
                        'length': len(sentence)
                    })
        
        # Sort by relevance and length
        relevant_info.sort(key=lambda x: (x['relevance'], x['length']), reverse=True)
        
        # Return top relevant sentences
        return [info['text'] for info in relevant_info[:5]]
    
    def _create_structured_answer(self, question: str, relevant_info: List[str], question_type: str) -> str:
        """Create a structured answer based on question type and relevant information"""
        if not relevant_info:
            return "I couldn't find specific information to answer your question."
        
        # Extract the main term/concept being asked about
        question_lower = question.lower()
        main_term = "RNNs"  # Default
        
        # Try to extract the actual term from the question
        if question_lower.startswith('what is '):
            words = question_lower.split()
            if len(words) >= 3:
                # Extract the term after "what is"
                potential_term = ' '.join(words[2:]).strip('?').strip()
                if potential_term:
                    # Convert to proper case (first letter uppercase)
                    main_term = potential_term.upper() if len(potential_term) <= 5 else potential_term.title()
        
        # Start with a clear introduction based on question type
        if question_type == 'definition':
            intro = f"## {main_term}\n\n**{main_term}** are a type of neural network designed to handle sequential data. Here's what the documents explain:\n\n"
        elif question_type == 'process':
            intro = f"## How {main_term} Work\n\nHere's how **{main_term} work** according to the documents:\n\n"
        elif question_type == 'explanation':
            intro = f"## Why {main_term} Are Effective\n\nThe documents provide the following explanation for **why {main_term} are effective**:\n\n"
        else:
            intro = f"## About {main_term}\n\nBased on the documents, here's what I found about **{main_term}**:\n\n"
        
        # Create bullet points for better readability
        bullet_points = []
        for i, info in enumerate(relevant_info[:4], 1):  # Limit to 4 points for readability
            # Clean up the sentence and make it more readable
            cleaned_info = info.strip()
            if cleaned_info.endswith('.'):
                cleaned_info = cleaned_info[:-1]  # Remove trailing period for bullet points
            
            bullet_points.append(f"• **{cleaned_info}**")
        
        # Combine all parts
        answer = intro + "\n".join(bullet_points)
        
        # Add a brief conclusion
        answer += "\n\n---\n\nThis information comes from the document content I analyzed. If you need more specific details or have follow-up questions, please let me know!"
        
        return answer
    
    def _calculate_confidence(self, search_results: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on search results"""
        if not search_results:
            return 0.0
        
        # Average similarity score
        avg_similarity = sum(result['similarity_score'] for result in search_results) / len(search_results)
        
        # Boost confidence if we have multiple high-quality results
        if len(search_results) >= 3 and avg_similarity > 0.7:
            confidence = min(0.95, avg_similarity + 0.1)
        else:
            confidence = avg_similarity
        
        return round(confidence, 2)
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history"""
        return self.conversation_history.copy()
    
    def clear_conversation_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded documents"""
        return self.vector_store.get_document_stats()
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported document formats"""
        return self.document_processor.get_supported_formats()
    
    def clear_all_documents(self):
        """Clear all documents from the knowledge base"""
        self.vector_store.clear_all()
        self.clear_conversation_history()
        logger.info("All documents and conversation history cleared") 