@echo off
echo 🌊 Starting Enhanced Groundwater Prediction API...
echo 🚀 GPU-Accelerated Machine Learning System
echo.

REM Check if required files exist
if not exist "groundwater_model.joblib" (
    echo ❌ Model file not found!
    echo 📊 Please ensure groundwater_model.joblib exists in this directory
    echo 💡 The model should have been trained automatically during setup
    pause
    exit /b 1
)

if not exist "processed_groundwater_data.csv" (
    echo ❌ Data file not found!
    echo 📋 Please ensure processed_groundwater_data.csv exists in this directory
    pause
    exit /b 1
)

echo ✅ Model files verified
echo ✅ Data files verified
echo.

REM Check if virtual environment exists, if not use system Python
if exist "venv\Scripts\python.exe" (
    echo 🐍 Using virtual environment Python
    set PYTHON_CMD=venv\Scripts\python.exe
    set UVICORN_CMD=venv\Scripts\uvicorn.exe
) else (
    echo 🐍 Using system Python
    set PYTHON_CMD=python
    set UVICORN_CMD=uvicorn
)

echo 🚀 Starting FastAPI server with GPU support...
echo.
echo 📱 Web Interface: http://localhost:8000
echo 📚 API Documentation: http://localhost:8000/docs
echo 🔍 Interactive API: http://localhost:8000/redoc
echo 💚 Health Check: http://localhost:8000/health
echo.
echo 🛑 Press Ctrl+C to stop the server
echo.

REM Start the FastAPI server
%UVICORN_CMD% fastapi_app:app --host 127.0.0.1 --port 8000 --reload