#!/bin/bash

# Turkish Text Classification API Startup Script

echo "🚀 Starting Turkish Text Classification API..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python 3.9 or higher."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Check if models exist
if [ ! -d "models" ] || [ ! -f "models/upper_model.pkl" ]; then
    echo "⚠️  Trained models not found. Training models..."
    python train_models.py
fi

# Start the API server
echo "🌐 Starting API server..."
echo "📖 API documentation will be available at: http://localhost:8000/docs"
echo "🔍 Health check endpoint: http://localhost:8000/health"
echo ""

python api.py
