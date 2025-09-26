@echo off
echo 🌊 Starting Groundwater Level Prediction API...

REM Check if model exists, if not train it
if not exist "groundwater_model.joblib" (
    echo 📊 Training ML model (this may take a few minutes)...
    C:/Users/mourish/Desktop/SIH/abc/.venv/Scripts/python.exe data_processor.py
    C:/Users/mourish/Desktop/SIH/abc/.venv/Scripts/python.exe ml_model.py
)

REM Check if processed data exists
if not exist "processed_groundwater_data.csv" (
    echo 📋 Processing groundwater data...
    C:/Users/mourish/Desktop/SIH/abc/.venv/Scripts/python.exe data_processor.py
)

echo 🚀 Starting FastAPI server...
echo 📱 Web Interface: http://localhost:8000
echo 📚 API Documentation: http://localhost:8000/docs
echo 🔍 API Endpoints: http://localhost:8000/redoc

REM Start the FastAPI server
C:/Users/mourish/Desktop/SIH/abc/.venv/Scripts/uvicorn.exe fastapi_app:app --host 0.0.0.0 --port 8000 --reload