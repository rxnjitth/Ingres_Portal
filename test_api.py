import requests
import json
import time

# API base URL
BASE_URL = "http://127.0.0.1:8000"

def test_api_endpoints():
    """Test all API endpoints"""
    
    print("🧪 Testing Groundwater Prediction API\n")
    
    try:
        # Test 1: Health check
        print("1. Testing health check...")
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health check passed")
            print(f"   Model loaded: {health_data['model_loaded']}")
            print(f"   Locations loaded: {health_data['locations_loaded']}\n")
        else:
            print(f"❌ Health check failed: {response.status_code}\n")
            return
        
        # Test 2: Get locations
        print("2. Testing locations endpoint...")
        response = requests.get(f"{BASE_URL}/locations")
        if response.status_code == 200:
            locations_data = response.json()
            print(f"✅ Locations endpoint working")
            print(f"   States available: {len(locations_data['states'])}")
            print(f"   Total locations: {locations_data['total_locations']}")
            print(f"   Sample states: {locations_data['states'][:5]}\n")
            
            # Get a sample state and district for testing
            sample_state = locations_data['states'][0]
            sample_districts = locations_data['districts_by_state'][sample_state]
            sample_district = sample_districts[0]
        else:
            print(f"❌ Locations endpoint failed: {response.status_code}\n")
            return
        
        # Test 3: Single prediction
        print("3. Testing single prediction...")
        prediction_data = {
            "state": sample_state,
            "district": sample_district,
            "year": 2026,
            "month": 6
        }
        
        response = requests.post(
            f"{BASE_URL}/predict",
            headers={"Content-Type": "application/json"},
            json=prediction_data
        )
        
        if response.status_code == 200:
            pred_result = response.json()
            print(f"✅ Single prediction working")
            print(f"   Location: {pred_result['district']}, {pred_result['state']}")
            print(f"   Predicted level: {pred_result['predicted_groundwater_level']} meters")
            print(f"   For: {pred_result['month']}/{pred_result['year']}\n")
        else:
            print(f"❌ Single prediction failed: {response.status_code}")
            print(f"   Error: {response.text}\n")
        
        # Test 4: State overview
        print("4. Testing state overview...")
        response = requests.get(f"{BASE_URL}/predict/state/{sample_state}?months=3")
        
        if response.status_code == 200:
            state_data = response.json()
            print(f"✅ State overview working")
            print(f"   State: {state_data['state']}")
            print(f"   Districts: {state_data['districts_count']}")
            print(f"   Predictions: {len(state_data['predictions'])}")
            
            # Show sample predictions
            sample_preds = state_data['predictions'][:3]
            for pred in sample_preds:
                print(f"   - {pred['district']}: {pred['predicted_groundwater_level']}m ({pred['month_name']} {pred['year']})")
            print()
        else:
            print(f"❌ State overview failed: {response.status_code}")
            print(f"   Error: {response.text}\n")
        
        # Test 5: District timeline
        print("5. Testing district timeline...")
        response = requests.get(f"{BASE_URL}/predict/district/{sample_state}/{sample_district}?start_year=2026&end_year=2026")
        
        if response.status_code == 200:
            timeline_data = response.json()
            print(f"✅ District timeline working")
            print(f"   District: {timeline_data['district']}, {timeline_data['state']}")
            print(f"   Predictions: {len(timeline_data['predictions'])}")
            
            # Show first 3 months
            for pred in timeline_data['predictions'][:3]:
                print(f"   - {pred['month_name']} {pred['year']}: {pred['predicted_groundwater_level']}m")
            print()
        else:
            print(f"❌ District timeline failed: {response.status_code}")
            print(f"   Error: {response.text}\n")
        
        # Test 6: Bulk prediction
        print("6. Testing bulk prediction...")
        bulk_data = {
            "locations": [
                {"state": sample_state, "district": sample_district},
                {"state": sample_state, "district": sample_districts[1] if len(sample_districts) > 1 else sample_district}
            ],
            "start_year": 2026,
            "end_year": 2026,
            "months": [1, 6, 12]
        }
        
        response = requests.post(
            f"{BASE_URL}/predict/bulk",
            headers={"Content-Type": "application/json"},
            json=bulk_data
        )
        
        if response.status_code == 200:
            bulk_result = response.json()
            print(f"✅ Bulk prediction working")
            print(f"   Total predictions: {bulk_result['total_predictions']}")
            
            # Show sample predictions
            for pred in bulk_result['predictions'][:3]:
                print(f"   - {pred['district']}: {pred['predicted_groundwater_level']}m ({pred['month']}/{pred['year']})")
            print()
        else:
            print(f"❌ Bulk prediction failed: {response.status_code}")
            print(f"   Error: {response.text}\n")
        
        print("🎉 All API tests completed successfully!")
        print(f"\n📱 Web Interface: {BASE_URL}")
        print(f"📚 API Documentation: {BASE_URL}/docs")
        print(f"🔍 Interactive API: {BASE_URL}/redoc")
        
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to API at {BASE_URL}")
        print("   Make sure the FastAPI server is running:")
        print("   uvicorn fastapi_app:app --host 127.0.0.1 --port 8000")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")

if __name__ == "__main__":
    # Wait a moment for server to be ready
    print("⏳ Waiting for API server to be ready...")
    time.sleep(2)
    test_api_endpoints()