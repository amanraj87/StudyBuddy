// API Configuration
const API_BASE_URL = 'http://localhost:8000';

// Configure marked.js for better markdown rendering
if (typeof marked !== 'undefined') {
    marked.setOptions({
        breaks: true,  // Convert line breaks to <br>
        gfm: true,     // Enable GitHub Flavored Markdown
        headerIds: false, // Disable header IDs for cleaner output
        mangle: false, // Don't mangle email addresses
        sanitize: false, // Allow HTML tags
        smartLists: true, // Use smarter list behavior
        smartypants: true, // Use smart typographic punctuation
        xhtml: false // Don't use XHTML output
    });
}

// Global variables
let conversationCount = 0;

// Voice Mode Variables
let voiceModeActive = false;
let voiceOutputActive = false;
let recognition = null;
let synthesis = window.speechSynthesis;
let isRecording = false;

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadProgress = document.getElementById('uploadProgress');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const uploadResults = document.getElementById('uploadResults');
const chatMessages = document.getElementById('chatMessages');
const questionInput = document.getElementById('questionInput');
const loadingOverlay = document.getElementById('loadingOverlay');
const loadingText = document.getElementById('loadingText');

// Voice Mode DOM Elements
const voiceModeToggle = document.getElementById('voiceModeToggle');
const voiceInputBtn = document.getElementById('voiceInputBtn');
const voiceOutputBtn = document.getElementById('voiceOutputBtn');
const voiceStatus = document.getElementById('voiceStatus');
const voiceStatusText = document.getElementById('voiceStatusText');
const voiceIcon = document.getElementById('voiceIcon');
const speakerIcon = document.getElementById('speakerIcon');

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    initializeVoiceRecognition();
    loadStats();
    loadConversationHistory();
    
    // Test markdown processing
    testMarkdownProcessing();
});

// Initialize voice recognition
function initializeVoiceRecognition() {
    // Check if speech recognition is supported
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
        console.warn('Speech recognition not supported');
        voiceModeToggle.disabled = true;
        voiceModeToggle.title = 'Speech recognition not supported in this browser';
        return;
    }

    // Initialize speech recognition
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';
    
    recognition.onstart = function() {
        isRecording = true;
        voiceInputBtn.classList.add('recording');
        voiceIcon.className = 'fas fa-stop';
        showVoiceStatus('Listening...');
    };
    
    recognition.onresult = function(event) {
        const transcript = event.results[0][0].transcript;
        questionInput.value = transcript;
        isRecording = false;
        voiceInputBtn.classList.remove('recording');
        voiceIcon.className = 'fas fa-microphone';
        hideVoiceStatus();
        
        // Automatically ask the question after voice input
        setTimeout(() => {
            askQuestion();
        }, 500);
    };
    
    recognition.onerror = function(event) {
        console.error('Speech recognition error:', event.error);
        isRecording = false;
        voiceInputBtn.classList.remove('recording');
        voiceIcon.className = 'fas fa-microphone';
        hideVoiceStatus();
        showToast('Voice recognition error: ' + event.error, 'error');
    };
    
    recognition.onend = function() {
        isRecording = false;
        voiceInputBtn.classList.remove('recording');
        voiceIcon.className = 'fas fa-microphone';
        hideVoiceStatus();
    };
}

// Voice Mode Functions
function toggleVoiceMode() {
    voiceModeActive = !voiceModeActive;
    
    if (voiceModeActive) {
        voiceModeToggle.classList.add('active');
        voiceModeToggle.innerHTML = '<i class="fas fa-microphone-slash"></i> Voice Mode';
        document.querySelector('.chat-input-container').classList.add('voice-mode-active');
        voiceInputBtn.style.display = 'flex';
        voiceOutputBtn.style.display = 'flex';
        showToast('Voice mode activated! Click the microphone to speak.', 'success');
    } else {
        voiceModeToggle.classList.remove('active');
        voiceModeToggle.innerHTML = '<i class="fas fa-microphone"></i> Voice Mode';
        document.querySelector('.chat-input-container').classList.remove('voice-mode-active');
        voiceInputBtn.style.display = 'none';
        voiceOutputBtn.style.display = 'none';
        hideVoiceStatus();
        
        // Stop any ongoing recording
        if (isRecording && recognition) {
            recognition.stop();
        }
        
        showToast('Voice mode deactivated', 'info');
    }
}

function startVoiceInput() {
    if (!recognition) {
        showToast('Speech recognition not supported in this browser', 'error');
        return;
    }
    
    if (isRecording) {
        recognition.stop();
    } else {
        recognition.start();
    }
}

function toggleVoiceOutput() {
    voiceOutputActive = !voiceOutputActive;
    
    if (voiceOutputActive) {
        voiceOutputBtn.classList.add('voice-output-active');
        speakerIcon.className = 'fas fa-volume-mute';
        showToast('Voice output activated! Responses will be spoken.', 'success');
    } else {
        voiceOutputBtn.classList.remove('voice-output-active');
        speakerIcon.className = 'fas fa-volume-up';
        showToast('Voice output deactivated', 'info');
    }
}

function showVoiceStatus(text) {
    voiceStatusText.textContent = text;
    voiceStatus.style.display = 'block';
}

function hideVoiceStatus() {
    voiceStatus.style.display = 'none';
}

function speakText(text) {
    if (!voiceOutputActive || !synthesis) return;
    
    // Stop any ongoing speech
    synthesis.cancel();
    
    // Clean text for speech (remove markdown)
    const cleanText = text.replace(/[#*`]/g, '').replace(/\n/g, ' ');
    
    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.rate = 0.9;
    utterance.pitch = 1;
    utterance.volume = 0.8;
    
    // Try to use a good voice
    const voices = synthesis.getVoices();
    const preferredVoice = voices.find(voice => 
        voice.lang.includes('en') && voice.name.includes('Google')
    ) || voices.find(voice => 
        voice.lang.includes('en')
    );
    
    if (preferredVoice) {
        utterance.voice = preferredVoice;
    }
    
    synthesis.speak(utterance);
}

// Initialize event listeners
function initializeEventListeners() {
    // File upload events
    fileInput.addEventListener('change', handleFileSelect);
    
    // Choose Files button event with debounce
    let isButtonClicked = false;
    document.getElementById('chooseFilesBtn').addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        
        if (!isButtonClicked) {
            isButtonClicked = true;
            fileInput.click();
            
            // Reset the flag after a short delay
            setTimeout(() => {
                isButtonClicked = false;
            }, 500);
        }
    });
    
    // Drag and drop events
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    
    // Chat input events
    questionInput.addEventListener('keypress', handleKeyPress);
}

// File upload handlers
function handleFileSelect(event) {
    const files = Array.from(event.target.files);
    if (files.length > 0) {
        uploadFiles(files);
    }
    // Clear the input value to allow selecting the same file again
    event.target.value = '';
}

function handleDragOver(event) {
    event.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(event) {
    event.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(event) {
    event.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = Array.from(event.dataTransfer.files);
    uploadFiles(files);
}

// Upload files to the server
async function uploadFiles(files) {
    if (files.length === 0) return;
    
    showLoading('Uploading documents...');
    
    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const progress = ((i + 1) / files.length) * 100;
        
        try {
            await uploadSingleFile(file, progress);
        } catch (error) {
            showUploadResult(file.name, false, error.message);
        }
    }
    
    hideLoading();
    loadStats();
    showToast('Documents uploaded successfully!', 'success');
}

async function uploadSingleFile(file, progress) {
    const formData = new FormData();
    formData.append('file', file);
    
    updateProgress(progress, `Uploading ${file.name}...`);
    
    const response = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData
    });
    
    const result = await response.json();
    
    if (result.success) {
        showUploadResult(file.name, true, `Processed ${result.chunks_created} chunks, ${result.word_count} words`);
    } else {
        throw new Error(result.error || 'Upload failed');
    }
}

function updateProgress(percentage, text) {
    progressFill.style.width = `${percentage}%`;
    progressText.textContent = text;
    uploadProgress.style.display = 'block';
}

function showUploadResult(filename, success, message) {
    const resultDiv = document.createElement('div');
    resultDiv.className = `upload-result ${success ? 'success' : 'error'}`;
    
    const icon = success ? 'fas fa-check-circle' : 'fas fa-exclamation-circle';
    resultDiv.innerHTML = `
        <i class="${icon}"></i>
        <div>
            <strong>${filename}</strong>
            <div>${message}</div>
        </div>
    `;
    
    uploadResults.appendChild(resultDiv);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        resultDiv.remove();
    }, 5000);
}

// Chat functionality
function handleKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        askQuestion();
    }
}

async function askQuestion() {
    const question = questionInput.value.trim();
    if (!question) return;
    
    // Add user message to chat
    addMessage('user', question);
    questionInput.value = '';
    
    // Show loading
    showLoading('Processing your question...');
    
    try {
        const response = await fetch(`${API_BASE_URL}/ask`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                question: question,
                use_history: true
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            addMessage('assistant', result.answer, {
                confidence: result.confidence,
                contextUsed: result.context_used
            });
            conversationCount++;
            updateStats();
        } else {
            addMessage('assistant', `Error: ${result.error}`, { isError: true });
        }
    } catch (error) {
        addMessage('assistant', 'Sorry, I encountered an error while processing your question.', { isError: true });
        console.error('Error asking question:', error);
    }
    
    hideLoading();
}

function addMessage(type, content, metadata = {}) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    
    const time = new Date().toLocaleTimeString();
    
    // Process markdown content
    const processedContent = marked.parse(content);
    
    // Debug: Log the processed content to see what's being generated
    console.log('Original content:', content);
    console.log('Processed markdown:', processedContent);
    
    let messageContent = '';
    
    if (type === 'user') {
        messageContent = `
            <div class="message-header">
                <span class="message-time">${time}</span>
                <span class="message-type">You</span>
            </div>
            <div class="message-content">
                <i class="fas fa-user"></i>
                ${content}
            </div>
        `;
    } else if (type === 'assistant') {
        const confidenceBadge = metadata.confidence ? 
            `<span class="confidence-badge">Confidence: ${(metadata.confidence * 100).toFixed(1)}%</span>` : '';
        
        messageContent = `
            <div class="message-header">
                <span class="message-time">${time}</span>
                <span class="message-type">Assistant</span>
                ${confidenceBadge}
            </div>
            <div class="message-content">
                <i class="fas fa-robot"></i>
                <div class="markdown-content">${processedContent}</div>
            </div>
        `;
        
        // Speak the response if voice output is active and it's not an error
        if (voiceOutputActive && !metadata.isError) {
            // Delay speaking to allow the message to render first
            setTimeout(() => {
                speakText(content);
            }, 500);
        }
    } else if (type === 'system') {
        messageContent = `
            <div class="message-content">
                <i class="fas fa-info-circle"></i>
                ${content}
            </div>
        `;
    }
    
    messageDiv.innerHTML = messageContent;
    chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    // Add context information if available
    if (metadata.contextUsed && metadata.contextUsed.length > 0) {
        const contextDiv = document.createElement('div');
        contextDiv.className = 'context-info';
        contextDiv.innerHTML = `
            <details>
                <summary>Context Used (${metadata.contextUsed.length} chunks)</summary>
                <div class="context-chunks">
                    ${metadata.contextUsed.map((chunk, index) => 
                        `<div class="context-chunk">
                            <strong>Chunk ${index + 1}:</strong> ${chunk}
                        </div>`
                    ).join('')}
                </div>
            </details>
        `;
        messageDiv.appendChild(contextDiv);
    }
}

// History management
async function clearHistory() {
    if (!confirm('Are you sure you want to clear the conversation history?')) return;
    
    try {
        const response = await fetch(`${API_BASE_URL}/conversation-history`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            chatMessages.innerHTML = `
                <div class="message system">
                    <div class="message-content">
                        <i class="fas fa-info-circle"></i>
                        Conversation history cleared. Start a new conversation!
                    </div>
                </div>
            `;
            conversationCount = 0;
            updateStats();
            showToast('Conversation history cleared', 'success');
        }
    } catch (error) {
        showToast('Error clearing history', 'error');
        console.error('Error clearing history:', error);
    }
}

async function clearDocuments() {
    if (!confirm('Are you sure you want to clear all documents? This will also clear the conversation history.')) return;
    
    showLoading('Clearing documents...');
    
    try {
        const response = await fetch(`${API_BASE_URL}/documents`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            chatMessages.innerHTML = `
                <div class="message system">
                    <div class="message-content">
                        <i class="fas fa-info-circle"></i>
                        All documents cleared. Upload new documents to start chatting!
                    </div>
                </div>
            `;
            uploadResults.innerHTML = '';
            conversationCount = 0;
            loadStats();
            showToast('All documents cleared', 'success');
        }
    } catch (error) {
        showToast('Error clearing documents', 'error');
        console.error('Error clearing documents:', error);
    }
    
    hideLoading();
}

async function loadConversationHistory() {
    try {
        const response = await fetch(`${API_BASE_URL}/conversation-history`);
        const data = await response.json();
        
        if (data.history && data.history.length > 0) {
            // Clear the welcome message
            chatMessages.innerHTML = '';
            
            // Add conversation history
            data.history.forEach(entry => {
                addMessage('user', entry.question);
                addMessage('assistant', entry.answer);
                conversationCount++;
            });
            
            updateStats();
        }
    } catch (error) {
        console.error('Error loading conversation history:', error);
    }
}

// Statistics
async function loadStats() {
    try {
        const response = await fetch(`${API_BASE_URL}/stats`);
        const data = await response.json();
        
        document.getElementById('docCount').textContent = data.documents.chroma_documents || 0;
        document.getElementById('chunkCount').textContent = data.documents.total_chunks || 0;
        document.getElementById('wordCount').textContent = data.documents.word_count || 0;
        document.getElementById('conversationCount').textContent = conversationCount;
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

function updateStats() {
    document.getElementById('conversationCount').textContent = conversationCount;
}

// Utility functions
function showLoading(message = 'Loading...') {
    loadingText.textContent = message;
    loadingOverlay.style.display = 'flex';
}

function hideLoading() {
    loadingOverlay.style.display = 'none';
    uploadProgress.style.display = 'none';
}

function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const icon = type === 'success' ? 'fas fa-check-circle' : 
                 type === 'error' ? 'fas fa-exclamation-circle' : 
                 'fas fa-info-circle';
    
    toast.innerHTML = `
        <i class="${icon}"></i>
        <span>${message}</span>
    `;
    
    const container = document.getElementById('toastContainer');
    container.appendChild(toast);
    
    // Auto-remove after 3 seconds
    setTimeout(() => {
        toast.remove();
    }, 3000);
}

// Error handling
window.addEventListener('error', function(event) {
    console.error('Global error:', event.error);
    showToast('An unexpected error occurred', 'error');
});

// Network status handling
window.addEventListener('online', function() {
    showToast('Connection restored', 'success');
});

window.addEventListener('offline', function() {
    showToast('No internet connection', 'error');
});

// Test function to verify markdown processing
function testMarkdownProcessing() {
    const testMarkdown = `## Test Header

This is a **bold** paragraph with some *italic* text.

- List item 1
- List item 2
- List item 3

### Subheader

Another paragraph here.`;
    
    const processed = marked.parse(testMarkdown);
    console.log('Markdown test - Original:', testMarkdown);
    console.log('Markdown test - Processed:', processed);
    
    // Test if marked is working
    if (processed.includes('<h2>') && processed.includes('<strong>') && processed.includes('<ul>')) {
        console.log('✅ Markdown processing is working correctly');
    } else {
        console.log('❌ Markdown processing may have issues');
    }
} 