#!/bin/bash

# Enhanced Groundwater Prediction API Startup Script
# GPU-Accelerated Machine Learning System

echo "🌊 Starting Enhanced Groundwater Prediction API..."
echo "🚀 GPU-Accelerated Machine Learning System"
echo ""

# Check if required files exist
if [ ! -f "groundwater_model.joblib" ]; then
    echo "❌ Model file not found!"
    echo "📊 Please ensure groundwater_model.joblib exists in this directory"
    echo "💡 The model should have been trained automatically during setup"
    exit 1
fi

if [ ! -f "processed_groundwater_data.csv" ]; then
    echo "❌ Data file not found!"
    echo "📋 Please ensure processed_groundwater_data.csv exists in this directory"
    exit 1
fi

echo "✅ Model files verified"
echo "✅ Data files verified"
echo ""

# Check if virtual environment exists
if [ -d "venv" ] && [ -f "venv/bin/python" ]; then
    echo "🐍 Using virtual environment Python"
    PYTHON_CMD="venv/bin/python"
    UVICORN_CMD="venv/bin/uvicorn"
else
    echo "🐍 Using system Python"
    PYTHON_CMD="python3"
    UVICORN_CMD="uvicorn"
fi

# Install requirements if needed
if [ -f "requirements.txt" ]; then
    echo "📦 Checking dependencies..."
    $PYTHON_CMD -m pip install -q -r requirements.txt
fi

echo "🚀 Starting FastAPI server with GPU support..."
echo ""
echo "📱 Web Interface: http://localhost:8000"
echo "📚 API Documentation: http://localhost:8000/docs"
echo "🔍 Interactive API: http://localhost:8000/redoc"
echo "💚 Health Check: http://localhost:8000/health"
echo ""
echo "🛑 Press Ctrl+C to stop the server"
echo ""

# Start the FastAPI server
$UVICORN_CMD fastapi_app:app --host 127.0.0.1 --port 8000 --reload