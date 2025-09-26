#!/bin/bash

# Groundwater Level Prediction API Startup Script

echo "🌊 Starting Groundwater Level Prediction API..."

# Check if model exists, if not train it
if [ ! -f "groundwater_model.joblib" ]; then
    echo "📊 Training ML model (this may take a few minutes)..."
    python data_processor.py
    python ml_model.py
fi

# Check if processed data exists
if [ ! -f "processed_groundwater_data.csv" ]; then
    echo "📋 Processing groundwater data..."
    python data_processor.py
fi

echo "🚀 Starting FastAPI server..."
echo "📱 Web Interface: http://localhost:8000"
echo "📚 API Documentation: http://localhost:8000/docs"
echo "🔍 API Endpoints: http://localhost:8000/redoc"

# Start the FastAPI server
uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --reload