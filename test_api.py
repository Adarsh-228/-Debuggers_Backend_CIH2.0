import requests
import json

API_URL = "http://localhost:5000"

def test_api():
    print("🧪 Testing Maternal Risk Prediction API")
    print("=" * 50)
    
    print("\n1. Testing API Health...")
    try:
        response = requests.get(f"{API_URL}/api/health")
        if response.status_code == 200:
            health = response.json()
            print(f"✅ API is healthy!")
            print(f"   Models loaded: {health['models_loaded']}")
            print(f"   Available models: {health['available_models']}")
        else:
            print("❌ API health check failed")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the Flask app is running!")
        return
    
    print("\n2. Getting model information...")
    response = requests.get(f"{API_URL}/api/models")
    if response.status_code == 200:
        models_info = response.json()
        print(f"✅ Available models: {list(models_info['models'].keys())}")
        print(f"   Best model: {models_info.get('best_model', 'Unknown')}")
        print(f"   Required features: {models_info['feature_names']}")
    
    print("\n3. Testing predictions...")
    
    test_cases = [
        {
            "name": "Low Risk Case",
            "data": {
                "Age": 25,
                "SystolicBP": 110,
                "DiastolicBP": 70,
                "BS": 6.5,
                "BodyTemp": 98.0,
                "HeartRate": 75
            }
        },
        {
            "name": "Mid Risk Case",
            "data": {
                "Age": 35,
                "SystolicBP": 130,
                "DiastolicBP": 85,
                "BS": 8.0,
                "BodyTemp": 99.0,
                "HeartRate": 80
            }
        },
        {
            "name": "High Risk Case",
            "data": {
                "Age": 42,
                "SystolicBP": 150,
                "DiastolicBP": 95,
                "BS": 15.0,
                "BodyTemp": 100.0,
                "HeartRate": 90
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n   Test {i}: {test_case['name']}")
        print(f"   Input: {test_case['data']}")
        
        response = requests.post(
            f"{API_URL}/api/predict",
            json=test_case['data'],
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Prediction successful!")
            
            if 'predictions' in result:
                if isinstance(result['predictions'], dict) and 'model' in result['predictions']:
                    pred = result['predictions']
                    print(f"      {pred['model']}: {pred['prediction']} (confidence: {pred.get('probability', 'N/A')})")
                else:
                    for model_name, pred in result['predictions'].items():
                        confidence = ""
                        if pred.get('probability'):
                            max_prob = max(pred['probability'].values())
                            confidence = f" (confidence: {max_prob:.2%})"
                        print(f"      {model_name}: {pred['prediction']}{confidence}")
                
                if 'recommended_model' in result:
                    print(f"   ⭐ Recommended model: {result['recommended_model']}")
        else:
            print(f"   ❌ Prediction failed: {response.text}")
    
    print("\n4. Testing specific model prediction...")
    test_data = {
        "Age": 30,
        "SystolicBP": 140,
        "DiastolicBP": 90,
        "BS": 12.0,
        "BodyTemp": 99.5,
        "HeartRate": 85,
        "model": "XGBoost"
    }
    
    response = requests.post(
        f"{API_URL}/api/predict",
        json=test_data,
        headers={'Content-Type': 'application/json'}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"   ✅ XGBoost-only prediction: {result['predictions']['prediction']}")
    else:
        print(f"   ❌ XGBoost prediction failed")
    
    print("\n5. Testing array input format...")
    features_array = {
        "features": [30, 140, 90, 12.0, 99.5, 85]
    }
    
    response = requests.post(
        f"{API_URL}/api/predict",
        json=features_array,
        headers={'Content-Type': 'application/json'}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"   ✅ Array format prediction successful!")
    else:
        print(f"   ❌ Array format prediction failed: {response.text}")
    
    print("\n" + "=" * 50)
    print("🎉 API testing completed!")
    print("\nTo use the API:")
    print("1. Start the Flask app: python api.py")
    print("2. Open browser: http://localhost:5000")
    print("3. Or use POST requests to: http://localhost:5000/api/predict")
    print("\nExample curl command:")
    print('curl -X POST http://localhost:5000/api/predict \\')
    print('  -H "Content-Type: application/json" \\')
    print('  -d \'{"Age": 25, "SystolicBP": 120, "DiastolicBP": 80, "BS": 7.0, "BodyTemp": 98.0, "HeartRate": 75}\'')

if __name__ == "__main__":
    test_api() 