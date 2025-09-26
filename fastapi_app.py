from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from ml_model import GroundwaterPredictor
import uvicorn
from datetime import datetime, timedelta
import json

# Initialize FastAPI app
app = FastAPI(
    title="Groundwater Level Prediction API",
    description="API for predicting groundwater levels across Indian states and districts",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables
predictor = None
available_locations = {}
processed_data = None

# Pydantic models
class PredictionRequest(BaseModel):
    state: str
    district: str
    year: int
    month: int
    additional_features: Optional[Dict[str, float]] = None

class PredictionResponse(BaseModel):
    state: str
    district: str
    year: int
    month: int
    predicted_groundwater_level: float
    prediction_date: str

class BulkPredictionRequest(BaseModel):
    locations: List[Dict[str, str]]  # List of {state, district}
    start_year: int
    end_year: int
    months: Optional[List[int]] = None  # If None, predict for all months

class LocationInfo(BaseModel):
    states: List[str]
    districts_by_state: Dict[str, List[str]]
    total_locations: int

class HistoricalDataRequest(BaseModel):
    state: str
    district: str
    start_year: Optional[int] = 2020
    end_year: Optional[int] = 2025

class HistoricalDataResponse(BaseModel):
    state: str
    district: str
    historical_data: List[Dict]
    data_period: str
    total_records: int

class TimelineData(BaseModel):
    past_data: List[Dict]  # 2020-2023
    present_data: List[Dict]  # 2024-2025
    future_predictions: List[Dict]  # 2026+

def initialize_model():
    """Initialize the ML model and load location data"""
    global predictor, available_locations, processed_data
    
    try:
        # Initialize and load the trained model
        predictor = GroundwaterPredictor()
        
        if not predictor.load_model("groundwater_model.joblib"):
            print("Model not found, training new model...")
            # If model doesn't exist, train it
            from ml_model import main as train_model
            predictor, _ = train_model()
        
        # Load processed data to get available locations
        processed_data = pd.read_csv("processed_groundwater_data.csv")
        processed_data['date'] = pd.to_datetime(processed_data['date'])
        
        # Extract available locations
        locations_df = processed_data[['state', 'district']].drop_duplicates()
        
        available_locations = {}
        for state in locations_df['state'].unique():
            districts = locations_df[locations_df['state'] == state]['district'].unique().tolist()
            available_locations[state] = sorted(districts)
        
        print(f"Model initialized successfully with {len(available_locations)} states")
        return True
        
    except Exception as e:
        print(f"Error initializing model: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize model when the app starts"""
    success = initialize_model()
    if not success:
        print("WARNING: Model initialization failed")

@app.get("/")
async def root():
    """Serve the main web interface"""
    return FileResponse("static/index.html")

@app.get("/locations", response_model=LocationInfo)
async def get_locations():
    """Get all available states and districts"""
    if not available_locations:
        raise HTTPException(status_code=500, detail="Location data not available")
    
    states = sorted(list(available_locations.keys()))
    total_locations = sum(len(districts) for districts in available_locations.values())
    
    return LocationInfo(
        states=states,
        districts_by_state=available_locations,
        total_locations=total_locations
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_groundwater_level(request: PredictionRequest):
    """Predict groundwater level for a specific location and time"""
    
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    # Validate input
    if request.month < 1 or request.month > 12:
        raise HTTPException(status_code=400, detail="Month must be between 1 and 12")
    
    if request.year < 2025 or request.year > 2050:
        raise HTTPException(status_code=400, detail="Year must be between 2025 and 2050")
    
    try:
        # Make prediction
        prediction = predictor.predict_future(
            state=request.state.upper(),
            district=request.district.upper(),
            year=request.year,
            month=request.month,
            additional_features=request.additional_features
        )
        
        return PredictionResponse(
            state=request.state.upper(),
            district=request.district.upper(),
            year=request.year,
            month=request.month,
            predicted_groundwater_level=round(prediction, 2),
            prediction_date=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/bulk")
async def bulk_predict(request: BulkPredictionRequest):
    """Predict groundwater levels for multiple locations and time periods"""
    
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    # Validate input
    if request.end_year < request.start_year:
        raise HTTPException(status_code=400, detail="End year must be >= start year")
    
    if request.start_year < 2025 or request.end_year > 2050:
        raise HTTPException(status_code=400, detail="Years must be between 2025 and 2050")
    
    # Default to all months if not specified
    months = request.months if request.months else list(range(1, 13))
    
    predictions = []
    
    try:
        for location in request.locations:
            state = location.get("state", "").upper()
            district = location.get("district", "").upper()
            
            if not state or not district:
                continue
            
            for year in range(request.start_year, request.end_year + 1):
                for month in months:
                    prediction = predictor.predict_future(state, district, year, month)
                    
                    predictions.append({
                        "state": state,
                        "district": district,
                        "year": year,
                        "month": month,
                        "predicted_groundwater_level": round(prediction, 2)
                    })
        
        return {
            "predictions": predictions,
            "total_predictions": len(predictions),
            "prediction_date": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bulk prediction failed: {str(e)}")

@app.get("/predict/state/{state_name}")
async def predict_state_districts(state_name: str, months: Optional[int] = 12):
    """Get predictions for all districts in a state for the next N months"""
    
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    state_upper = state_name.upper()
    
    if state_upper not in available_locations:
        raise HTTPException(
            status_code=404, 
            detail=f"State '{state_name}' not found. Available states: {list(available_locations.keys())}"
        )
    
    districts = available_locations[state_upper]
    
    # Generate predictions for next N months
    current_date = datetime.now()
    predictions = []
    
    try:
        for i in range(months):
            future_date = current_date + timedelta(days=30 * i)  # Approximate months
            year = future_date.year
            month = future_date.month
            
            for district in districts:
                prediction = predictor.predict_future(state_upper, district, year, month)
                
                predictions.append({
                    "district": district,
                    "year": year,
                    "month": month,
                    "month_name": future_date.strftime("%B"),
                    "predicted_groundwater_level": round(prediction, 2)
                })
        
        return {
            "state": state_upper,
            "districts_count": len(districts),
            "months_predicted": months,
            "predictions": predictions,
            "prediction_date": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"State prediction failed: {str(e)}")

@app.get("/predict/district/{state_name}/{district_name}")
async def predict_district_timeline(
    state_name: str, 
    district_name: str,
    start_year: int = Query(2026, description="Start year for predictions"),
    end_year: int = Query(2026, description="End year for predictions")
):
    """Get monthly predictions for a specific district over a time period"""
    
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    state_upper = state_name.upper()
    district_upper = district_name.upper()
    
    # Validate location
    if state_upper not in available_locations:
        raise HTTPException(status_code=404, detail=f"State '{state_name}' not found")
    
    if district_upper not in available_locations[state_upper]:
        raise HTTPException(
            status_code=404, 
            detail=f"District '{district_name}' not found in {state_name}"
        )
    
    # Validate years
    if end_year < start_year or start_year < 2025 or end_year > 2050:
        raise HTTPException(status_code=400, detail="Invalid year range")
    
    predictions = []
    
    try:
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                prediction = predictor.predict_future(state_upper, district_upper, year, month)
                
                date_obj = datetime(year, month, 1)
                
                predictions.append({
                    "year": year,
                    "month": month,
                    "month_name": date_obj.strftime("%B"),
                    "date": date_obj.strftime("%Y-%m"),
                    "predicted_groundwater_level": round(prediction, 2)
                })
        
        return {
            "state": state_upper,
            "district": district_upper,
            "start_year": start_year,
            "end_year": end_year,
            "predictions": predictions,
            "prediction_date": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"District timeline prediction failed: {str(e)}")

@app.get("/historical/data/{state_name}/{district_name}")
async def get_historical_data(
    state_name: str,
    district_name: str,
    start_year: int = Query(2020, description="Start year for historical data"),
    end_year: int = Query(2025, description="End year for historical data")
):
    """Get historical groundwater data for a specific location"""
    
    if processed_data is None:
        raise HTTPException(status_code=500, detail="Historical data not available")
    
    state_upper = state_name.upper()
    district_upper = district_name.upper()
    
    # Validate and find matching location
    matched_state = None
    matched_district = None
    
    # Find state match
    for state in available_locations.keys():
        if state_upper == state or state_name.upper() in state or state in state_name.upper():
            matched_state = state
            break
    
    if not matched_state:
        available_states = list(available_locations.keys())
        raise HTTPException(
            status_code=404, 
            detail=f"State '{state_name}' not found. Available states: {available_states[:10]}"
        )
    
    # Find district match
    for district in available_locations[matched_state]:
        if district_upper == district or district_name.upper() in district or district in district_name.upper():
            matched_district = district
            break
    
    if not matched_district:
        available_districts = available_locations[matched_state]
        raise HTTPException(
            status_code=404, 
            detail=f"District '{district_name}' not found in {matched_state}. Available: {available_districts[:10]}"
        )
    
    try:
        # Filter data for the specific location and time range
        filtered_data = processed_data[
            (processed_data['state'] == matched_state) &
            (processed_data['district'] == matched_district) &
            (processed_data['year'] >= start_year) &
            (processed_data['year'] <= end_year)
        ].copy()
        
        if len(filtered_data) == 0:
            raise HTTPException(status_code=404, detail="No data found for specified criteria")
        
        # Sort by date
        filtered_data = filtered_data.sort_values('date')
        
        # Convert to list of dictionaries
        historical_records = []
        for _, row in filtered_data.iterrows():
            record = {
                "date": row['date'].strftime("%Y-%m-%d"),
                "year": int(row['year']),
                "month": int(row['month_num']),
                "month_name": row['month'],
                "groundwater_level": round(float(row['groundwater_level']), 2),
                "rainfall_mm": round(float(row['total_rainfall']), 2) if pd.notna(row['total_rainfall']) else 0,
                "recharge_ham": round(float(row['total_recharge']), 2) if pd.notna(row['total_recharge']) else 0,
                "data_type": "historical" if row['year'] <= 2024 else "present"
            }
            historical_records.append(record)
        
        return {
            "state": matched_state,
            "district": matched_district,
            "start_year": start_year,
            "end_year": end_year,
            "historical_data": historical_records,
            "total_records": len(historical_records),
            "data_period": f"{start_year}-{end_year}",
            "retrieval_date": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve historical data: {str(e)}")

@app.get("/historical/summary/{state_name}")
async def get_state_historical_summary(
    state_name: str,
    year: int = Query(2024, description="Year for summary data")
):
    """Get historical summary for all districts in a state for a specific year"""
    
    if processed_data is None:
        raise HTTPException(status_code=500, detail="Historical data not available")
    
    state_upper = state_name.upper()
    
    if state_upper not in available_locations:
        raise HTTPException(status_code=404, detail=f"State '{state_name}' not found")
    
    try:
        # Filter data for the state and year
        state_data = processed_data[
            (processed_data['state'] == state_upper) &
            (processed_data['year'] == year)
        ].copy()
        
        if len(state_data) == 0:
            raise HTTPException(status_code=404, detail=f"No data found for {state_name} in {year}")
        
        # Calculate district-wise averages
        district_summary = state_data.groupby('district').agg({
            'groundwater_level': ['mean', 'min', 'max'],
            'total_rainfall': 'mean',
            'total_recharge': 'mean'
        }).round(2)
        
        # Flatten column names
        district_summary.columns = ['_'.join(col).strip() for col in district_summary.columns.values]
        district_summary = district_summary.reset_index()
        
        summary_data = []
        for _, row in district_summary.iterrows():
            summary_data.append({
                "district": row['district'],
                "avg_groundwater_level": float(row['groundwater_level_mean']),
                "min_groundwater_level": float(row['groundwater_level_min']),
                "max_groundwater_level": float(row['groundwater_level_max']),
                "avg_rainfall": float(row['total_rainfall_mean']),
                "avg_recharge": float(row['total_recharge_mean'])
            })
        
        return {
            "state": state_upper,
            "year": year,
            "districts_count": len(summary_data),
            "district_summaries": sorted(summary_data, key=lambda x: x['district']),
            "state_averages": {
                "avg_groundwater_level": round(state_data['groundwater_level'].mean(), 2),
                "min_groundwater_level": round(state_data['groundwater_level'].min(), 2),
                "max_groundwater_level": round(state_data['groundwater_level'].max(), 2),
                "total_rainfall": round(state_data['total_rainfall'].mean(), 2),
                "total_recharge": round(state_data['total_recharge'].mean(), 2)
            },
            "retrieval_date": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get state summary: {str(e)}")

@app.get("/timeline/{state_name}/{district_name}")
async def get_complete_timeline(
    state_name: str,
    district_name: str,
    future_years: int = Query(2, description="Number of future years to predict")
):
    """Get complete timeline: past data, present data, and future predictions"""
    
    if processed_data is None or predictor is None:
        raise HTTPException(status_code=500, detail="Data or model not available")
    
    state_upper = state_name.upper()
    district_upper = district_name.upper()
    
    # Validate location
    if state_upper not in available_locations:
        raise HTTPException(status_code=404, detail=f"State '{state_name}' not found")
    
    if district_upper not in available_locations[state_upper]:
        raise HTTPException(
            status_code=404, 
            detail=f"District '{district_name}' not found in {state_name}"
        )
    
    try:
        # Get historical data (2020-2023)
        past_data = processed_data[
            (processed_data['state'] == state_upper) &
            (processed_data['district'] == district_upper) &
            (processed_data['year'] <= 2023)
        ].copy().sort_values('date')
        
        # Get present data (2024-2025)
        present_data = processed_data[
            (processed_data['state'] == state_upper) &
            (processed_data['district'] == district_upper) &
            (processed_data['year'].isin([2024, 2025]))
        ].copy().sort_values('date')
        
        # Generate future predictions
        current_year = 2025
        future_predictions = []
        
        for year in range(current_year + 1, current_year + future_years + 1):
            for month in range(1, 13):
                prediction = predictor.predict_future(state_upper, district_upper, year, month)
                future_predictions.append({
                    "date": f"{year}-{month:02d}-01",
                    "year": year,
                    "month": month,
                    "month_name": datetime(year, month, 1).strftime("%B"),
                    "groundwater_level": round(prediction, 2),
                    "data_type": "prediction"
                })
        
        # Format historical data
        def format_data_records(df, data_type):
            records = []
            for _, row in df.iterrows():
                records.append({
                    "date": row['date'].strftime("%Y-%m-%d"),
                    "year": int(row['year']),
                    "month": int(row['month_num']),
                    "month_name": row['month'],
                    "groundwater_level": round(float(row['groundwater_level']), 2),
                    "rainfall_mm": round(float(row['total_rainfall']), 2) if pd.notna(row['total_rainfall']) else 0,
                    "recharge_ham": round(float(row['total_recharge']), 2) if pd.notna(row['total_recharge']) else 0,
                    "data_type": data_type
                })
            return records
        
        past_records = format_data_records(past_data, "historical")
        present_records = format_data_records(present_data, "present")
        
        # Calculate trend analysis
        all_historical = past_records + present_records
        if len(all_historical) > 0:
            levels = [r['groundwater_level'] for r in all_historical]
            trend_analysis = {
                "average_level": round(sum(levels) / len(levels), 2),
                "min_level": min(levels),
                "max_level": max(levels),
                "trend_direction": "stable"  # Could be enhanced with actual trend calculation
            }
        else:
            trend_analysis = {}
        
        return {
            "state": state_upper,
            "district": district_upper,
            "timeline_data": {
                "past_data": past_records,  # 2020-2023
                "present_data": present_records,  # 2024-2025
                "future_predictions": future_predictions  # 2026+
            },
            "summary": {
                "total_past_records": len(past_records),
                "total_present_records": len(present_records),
                "total_future_predictions": len(future_predictions),
                "date_range": {
                    "past": "2020-2023",
                    "present": "2024-2025", 
                    "future": f"2026-{current_year + future_years}"
                }
            },
            "trend_analysis": trend_analysis,
            "retrieval_date": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get timeline data: {str(e)}")

@app.get("/current/status")
async def get_current_status():
    """Get current groundwater status overview across all states"""
    
    if processed_data is None:
        raise HTTPException(status_code=500, detail="Historical data not available")
    
    try:
        # Get current year data (2025)
        current_data = processed_data[processed_data['year'] == 2025].copy()
        
        if len(current_data) == 0:
            raise HTTPException(status_code=404, detail="No current data available")
        
        # Calculate state-wise current status
        state_status = current_data.groupby('state').agg({
            'groundwater_level': ['mean', 'count'],
            'total_rainfall': 'mean',
            'total_recharge': 'mean'
        }).round(2)
        
        # Flatten column names
        state_status.columns = ['_'.join(col).strip() for col in state_status.columns.values]
        state_status = state_status.reset_index()
        
        status_data = []
        for _, row in state_status.iterrows():
            status_data.append({
                "state": row['state'],
                "avg_groundwater_level": float(row['groundwater_level_mean']),
                "districts_monitored": int(row['groundwater_level_count']) // 12,  # Divide by months
                "avg_rainfall": float(row['total_rainfall_mean']),
                "avg_recharge": float(row['total_recharge_mean']),
                "status_category": "Normal" if row['groundwater_level_mean'] > 3 else "Low" if row['groundwater_level_mean'] > 1 else "Critical"
            })
        
        # Overall India status
        overall_stats = {
            "total_states": len(status_data),
            "total_districts_monitored": sum([s['districts_monitored'] for s in status_data]),
            "national_avg_groundwater_level": round(current_data['groundwater_level'].mean(), 2),
            "states_normal": len([s for s in status_data if s['status_category'] == 'Normal']),
            "states_low": len([s for s in status_data if s['status_category'] == 'Low']),
            "states_critical": len([s for s in status_data if s['status_category'] == 'Critical'])
        }
        
        return {
            "current_year": 2025,
            "overall_status": overall_stats,
            "state_wise_status": sorted(status_data, key=lambda x: x['state']),
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get current status: {str(e)}")

@app.get("/debug/data-sample")
async def get_data_sample():
    """Debug endpoint to see sample data"""
    if processed_data is None:
        return {"error": "No data loaded"}
    
    sample = processed_data.head(10).to_dict('records')
    return {
        "total_records": len(processed_data),
        "columns": list(processed_data.columns),
        "sample_data": sample,
        "unique_states": processed_data['state'].nunique(),
        "unique_districts": processed_data['district'].nunique(),
        "year_range": [int(processed_data['year'].min()), int(processed_data['year'].max())]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None and predictor.is_trained,
        "locations_loaded": len(available_locations) > 0,
        "historical_data_loaded": processed_data is not None,
        "total_historical_records": len(processed_data) if processed_data is not None else 0,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=True
    )