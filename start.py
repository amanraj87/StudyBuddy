#!/usr/bin/env python3
"""
Startup script for Document Conversational Agent
Launches the backend server and opens the frontend in a web browser
"""

import os
import sys
import time
import webbrowser
import subprocess
import threading
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import fastapi
        import uvicorn
        import sentence_transformers
        import chromadb
        print("✓ All required dependencies are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please run: pip install -r backend/requirements.txt")
        return False

def start_backend():
    """Start the FastAPI backend server"""
    # Get the absolute path to the project root
    project_root = Path(__file__).parent.absolute()
    backend_dir = project_root / "backend"
    
    if not backend_dir.exists():
        print("✗ Backend directory not found")
        return False
    
    try:
        print("🚀 Starting backend server...")
        # Use absolute paths and run from the backend directory
        subprocess.run([sys.executable, str(backend_dir / "main.py")], 
                      cwd=str(backend_dir), check=True)
    except KeyboardInterrupt:
        print("\n🛑 Backend server stopped")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to start backend server: {e}")
        return False
    
    return True

def open_frontend():
    """Open the frontend in a web browser"""
    # Get the absolute path to the project root
    project_root = Path(__file__).parent.absolute()
    frontend_path = project_root / "frontend" / "index.html"
    
    if not frontend_path.exists():
        print("✗ Frontend not found")
        return False
    
    # Wait a moment for the backend to start
    time.sleep(3)
    
    try:
        # Try to open with a local server first
        print("🌐 Starting frontend server...")
        frontend_dir = project_root / "frontend"
        
        # Start a simple HTTP server with absolute paths
        server_process = subprocess.Popen([
            sys.executable, "-m", "http.server", "8080"
        ], cwd=str(frontend_dir))
        
        # Wait a moment for server to start
        time.sleep(2)
        
        # Open in browser
        webbrowser.open("http://localhost:8080")
        print("✓ Frontend opened in browser")
        
        # Keep the server running
        try:
            server_process.wait()
        except KeyboardInterrupt:
            server_process.terminate()
            print("\n🛑 Frontend server stopped")
        
    except Exception as e:
        print(f"✗ Failed to open frontend: {e}")
        return False
    
    return True

def main():
    """Main startup function"""
    print("=" * 50)
    print("🤖 Document Conversational Agent")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Get absolute paths
    project_root = Path(__file__).parent.absolute()
    
    # Check if backend and frontend exist
    if not (project_root / "backend").exists():
        print("✗ Backend directory not found")
        sys.exit(1)
    
    if not (project_root / "frontend").exists():
        print("✗ Frontend directory not found")
        sys.exit(1)
    
    print("\n📋 Starting services...")
    
    # Start backend in a separate thread
    backend_thread = threading.Thread(target=start_backend, daemon=True)
    backend_thread.start()
    
    # Start frontend
    open_frontend()
    
    print("\n✅ Application started successfully!")
    print("📖 Backend API: http://localhost:8000")
    print("🌐 Frontend: http://localhost:8080")
    print("\n💡 Press Ctrl+C to stop all services")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0) 