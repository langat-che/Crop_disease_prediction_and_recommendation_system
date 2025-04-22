#!/usr/bin/env python
# test_api.py - Script to test the API is working correctly

import requests
import sys
import os
import argparse

def test_api(base_url):
    """Test the API by making a request to the root endpoint."""
    print(f"Testing API at {base_url}...")
    
    try:
        # Test the root endpoint
        response = requests.get(f"{base_url}")
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses
        
        data = response.json()
        print(f"API Response: {data}")
        
        # Check if models are loaded
        if "loaded_models" in data and len(data["loaded_models"]) > 0:
            print(f"✅ API is working and has loaded {len(data['loaded_models'])} models: {', '.join(data['loaded_models'])}")
            return True
        else:
            print("⚠️ API is responding but no models are loaded!")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Error connecting to API: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the Crop Disease Predictor API")
    parser.add_argument("--url", default="http://localhost:10000", help="Base URL of the API (default: http://localhost:10000)")
    
    args = parser.parse_args()
    
    if test_api(args.url):
        print("API test passed!")
        sys.exit(0)
    else:
        print("API test failed!")
        sys.exit(1)
