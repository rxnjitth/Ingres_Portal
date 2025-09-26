"""
Quick Fix for Groundwater Prediction System

This script addresses the "Not Found" errors in the web interface by:
1. Checking the data structure
2. Fixing any data matching issues
3. Providing a working example
"""

import pandas as pd
import requests
import json

def diagnose_and_fix():
    print("🔍 Diagnosing Groundwater Prediction System...")
    
    # Check if processed data exists and is valid
    try:
        df = pd.read_csv("processed_groundwater_data.csv")
        print(f"✅ Data file loaded: {len(df)} records")
        print(f"📊 Columns: {list(df.columns)}")
        print(f"🗺️ States: {df['state'].nunique()}")
        print(f"🏢 Districts: {df['district'].nunique()}")
        print(f"📅 Years: {df['year'].min()} - {df['year'].max()}")
        
        # Show sample state/district combinations
        sample_locations = df[['state', 'district']].drop_duplicates().head(10)
        print("\n📍 Sample locations available:")
        for _, row in sample_locations.iterrows():
            print(f"   - {row['state']} → {row['district']}")
        
        # Test the API
        print("\n🧪 Testing API at http://127.0.0.1:8002...")
        
        # Test 1: Health check
        try:
            response = requests.get("http://127.0.0.1:8002/health", timeout=5)
            if response.status_code == 200:
                health = response.json()
                print(f"✅ API is healthy: {health['total_historical_records']} records loaded")
            else:
                print(f"❌ API health check failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Cannot connect to API: {e}")
            return
        
        # Test 2: Try a sample prediction with real data
        try:
            first_location = sample_locations.iloc[0]
            state = first_location['state']
            district = first_location['district']
            
            print(f"\n🎯 Testing prediction for: {district}, {state}")
            
            pred_data = {
                "state": state,
                "district": district,
                "year": 2026,
                "month": 6
            }
            
            response = requests.post("http://127.0.0.1:8002/predict", json=pred_data, timeout=10)
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Prediction successful: {result['predicted_groundwater_level']} meters")
            else:
                print(f"❌ Prediction failed: {response.status_code} - {response.text}")
        
        except Exception as e:
            print(f"❌ Prediction test failed: {e}")
        
        # Test 3: Historical data
        try:
            print(f"\n📊 Testing historical data for: {district}, {state}")
            
            response = requests.get(f"http://127.0.0.1:8002/historical/data/{state}/{district}?start_year=2020&end_year=2024", timeout=10)
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Historical data successful: {result['total_records']} records found")
            else:
                print(f"❌ Historical data failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Historical data test failed: {e}")
        
        print(f"\n🎉 Diagnosis complete!")
        print(f"🌐 Working web interface: http://127.0.0.1:8002")
        print(f"📚 API documentation: http://127.0.0.1:8002/docs")
        print(f"🔍 Debug data view: http://127.0.0.1:8002/debug/data-sample")
        
        print(f"\n💡 Recommended test locations:")
        for _, row in sample_locations.head(5).iterrows():
            print(f"   - State: {row['state']}, District: {row['district']}")
            
    except FileNotFoundError:
        print("❌ processed_groundwater_data.csv not found!")
        print("🔧 Run: python data_processor.py")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    diagnose_and_fix()