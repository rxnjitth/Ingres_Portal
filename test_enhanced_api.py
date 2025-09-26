import subprocess
import json

def test_new_endpoints():
    """Test the new historical data endpoints"""
    
    base_url = "http://127.0.0.1:8001"
    
    print("🧪 Testing Enhanced Groundwater API with Historical Data\n")
    
    # Test 1: Current Status
    print("1. Testing Current Status endpoint...")
    try:
        result = subprocess.run([
            'powershell', '-Command', 
            f'Invoke-RestMethod -Uri "{base_url}/current/status" -Method Get'
        ], capture_output=True, text=True, timeout=15)
        
        if result.returncode == 0:
            print("✅ Current Status API working")
            # Parse some key info from the output
            output_lines = result.stdout.strip().split('\n')
            for line in output_lines[:10]:  # Show first 10 lines
                print(f"   {line}")
        else:
            print("❌ Current Status failed")
            print(f"   Error: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Current Status test error: {e}")
    
    print()
    
    # Test 2: Historical Data
    print("2. Testing Historical Data endpoint...")
    try:
        result = subprocess.run([
            'powershell', '-Command', 
            f'Invoke-RestMethod -Uri "{base_url}/historical/data/KARNATAKA/BANGALORE%20URBAN?start_year=2022&end_year=2024" -Method Get'
        ], capture_output=True, text=True, timeout=15)
        
        if result.returncode == 0:
            print("✅ Historical Data API working")
            # Show some output
            output_lines = result.stdout.strip().split('\n')
            for line in output_lines[:8]:  # Show first 8 lines
                print(f"   {line}")
        else:
            print("❌ Historical Data failed")
            print(f"   Error: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Historical Data test error: {e}")
    
    print()
    
    # Test 3: Timeline Data
    print("3. Testing Complete Timeline endpoint...")
    try:
        result = subprocess.run([
            'powershell', '-Command', 
            f'Invoke-RestMethod -Uri "{base_url}/timeline/KARNATAKA/BANGALORE%20URBAN?future_years=1" -Method Get'
        ], capture_output=True, text=True, timeout=15)
        
        if result.returncode == 0:
            print("✅ Timeline API working")
            # Show some output
            output_lines = result.stdout.strip().split('\n')
            for line in output_lines[:6]:  # Show first 6 lines
                print(f"   {line}")
        else:
            print("❌ Timeline failed")
            print(f"   Error: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Timeline test error: {e}")
    
    print()
    
    print("🎉 Enhanced API Testing Complete!")
    print(f"📱 Enhanced Web Interface: http://127.0.0.1:8001")
    print(f"📚 Updated API Documentation: http://127.0.0.1:8001/docs")
    print("\nNew Features Available:")
    print("- 📊 Historical Data Viewing (2020-2025)")
    print("- 🌊 Current Status Dashboard")  
    print("- 📈 Complete Timeline Analysis")
    print("- 📉 Interactive Charts and Graphs")

if __name__ == "__main__":
    test_new_endpoints()