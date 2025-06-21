#!/bin/bash

# DiffSense - Feature Drift Detector
# Setup and Demo Script

set -e

echo "🔬 DiffSense - Feature Drift Detector"
echo "======================================"
echo "AI Berkeley Hackathon Project"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Parse command line arguments
MODE="demo"
if [ $# -gt 0 ]; then
    MODE="$1"
fi

case $MODE in
    "demo")
        echo "🚀 Running Quick Demo (No dependencies required)"
        echo "This demonstrates core git analysis functionality"
        echo ""
        
        # Check Python
        if ! command_exists python3; then
            echo "❌ Error: Python 3 is required but not installed"
            exit 1
        fi
        
        # Setup backend if needed
        cd backend
        if [ ! -d "venv" ]; then
            echo "📦 Creating Python virtual environment..."
            python3 -m venv venv
        fi
        
        echo "📦 Installing minimal dependencies..."
        source venv/bin/activate
        pip install -q gitpython numpy scikit-learn
        
        echo "🔍 Running drift analysis demo..."
        python simple_demo.py
        ;;
        
    "full")
        echo "🚀 Starting Full Application (Frontend + Backend)"
        echo "This requires Node.js and will start both servers"
        echo ""
        
        # Check dependencies
        if ! command_exists python3; then
            echo "❌ Error: Python 3 is required but not installed"
            exit 1
        fi
        
        if ! command_exists node; then
            echo "❌ Error: Node.js is required but not installed"
            echo "Please install Node.js from https://nodejs.org/"
            exit 1
        fi
        
        # Setup backend
        echo "📦 Setting up backend..."
        cd backend
        if [ ! -d "venv" ]; then
            echo "Creating Python virtual environment..."
            python3 -m venv venv
        fi
        
        source venv/bin/activate
        pip install -q gitpython numpy scikit-learn fastapi uvicorn pydantic
        
        # Setup frontend
        echo "📦 Setting up frontend..."
        cd ../frontend
        if [ ! -d "node_modules" ]; then
            echo "Installing Node.js dependencies..."
            npm install
        fi
        
        # Start services
        echo "🔄 Starting services..."
        
        # Start backend in background
        echo "Starting backend server on http://localhost:8000..."
        cd ../backend
        source venv/bin/activate
        python main.py &
        BACKEND_PID=$!
        
        # Wait for backend to start
        sleep 3
        
        # Start frontend
        echo "Starting frontend server on http://localhost:3000..."
        cd ../frontend
        npm run dev &
        FRONTEND_PID=$!
        
        # Function to cleanup on exit
        cleanup() {
            echo "🛑 Shutting down services..."
            kill $BACKEND_PID 2>/dev/null || true
            kill $FRONTEND_PID 2>/dev/null || true
            exit 0
        }
        
        # Set trap to cleanup on script exit
        trap cleanup SIGINT SIGTERM
        
        echo "✅ DiffSense is running!"
        echo "   🌐 Frontend: http://localhost:3000"
        echo "   🔧 Backend API: http://localhost:8000"
        echo "   📚 API Docs: http://localhost:8000/docs"
        echo ""
        echo "Press Ctrl+C to stop all services"
        
        # Wait for user to stop
        wait
        ;;
        
    "setup")
        echo "📦 Setting up DiffSense development environment"
        echo ""
        
        # Check dependencies
        echo "Checking system dependencies..."
        
        if ! command_exists python3; then
            echo "❌ Python 3 is required but not installed"
            echo "Please install Python 3.8 or higher"
            exit 1
        else
            echo "✅ Python 3 found: $(python3 --version)"
        fi
        
        if ! command_exists git; then
            echo "❌ Git is required but not installed"
            exit 1
        else
            echo "✅ Git found: $(git --version)"
        fi
        
        if ! command_exists node; then
            echo "⚠️  Node.js not found (required for full web interface)"
            echo "   You can still run the demo with: ./setup.sh demo"
        else
            echo "✅ Node.js found: $(node --version)"
        fi
        
        # Setup backend
        echo ""
        echo "Setting up backend environment..."
        cd backend
        
        if [ ! -d "venv" ]; then
            echo "Creating Python virtual environment..."
            python3 -m venv venv
        fi
        
        source venv/bin/activate
        echo "Installing Python dependencies..."
        pip install -q -r requirements.txt || {
            echo "Installing minimal dependencies..."
            pip install -q gitpython numpy scikit-learn fastapi uvicorn pydantic
        }
        
        echo "✅ Backend setup complete"
        
        # Setup frontend if Node.js is available
        if command_exists node; then
            echo ""
            echo "Setting up frontend environment..."
            cd ../frontend
            
            echo "Installing Node.js dependencies..."
            npm install
            
            echo "✅ Frontend setup complete"
        fi
        
        echo ""
        echo "🎉 Setup complete! You can now run:"
        echo "   ./setup.sh demo     # Quick demo (no web interface)"
        echo "   ./setup.sh full     # Full application with web interface"
        ;;
        
    "help"|"-h"|"--help")
        echo "Usage: $0 [mode]"
        echo ""
        echo "Modes:"
        echo "  demo     Run quick command-line demo (default)"
        echo "  full     Start full web application"
        echo "  setup    Setup development environment"
        echo "  help     Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0           # Run demo"
        echo "  $0 demo      # Run demo"
        echo "  $0 full      # Start web app"
        echo "  $0 setup     # Setup environment"
        ;;
        
    *)
        echo "❌ Unknown mode: $MODE"
        echo "Run '$0 help' for usage information"
        exit 1
        ;;
esac
