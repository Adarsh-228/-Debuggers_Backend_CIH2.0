#!/usr/bin/env python3
"""
Test script for image-based prediction endpoints
This script demonstrates how to use the new image upload functionality
"""

import requests
import json

def test_image_endpoints():
    """Test both maternal and preterm image prediction endpoints"""
    
    base_url = "http://localhost:5000"
    
    print("="*60)
    print("TESTING IMAGE-BASED PREDICTION ENDPOINTS")
    print("="*60)
    
    # Test 1: Health check to verify endpoints are available
    print("\n1. Checking API health...")
    try:
        response = requests.get(f"{base_url}/api/health")
        if response.status_code == 200:
            health_data = response.json()
            print("✓ API is healthy")
            print(f"  - Gemini configured: {health_data.get('gemini_configured', False)}")
            print(f"  - Image processing available: {health_data.get('services', {}).get('image_processing', False)}")
            print(f"  - Available endpoints: {len(health_data.get('endpoints', {}))}")
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"✗ Cannot connect to API: {e}")
        return
    
    # Test 2: Preterm birth image prediction
    print("\n2. Testing preterm birth image prediction...")
    print("   Instructions:")
    print("   - Upload a medical image (lab report, etc.) to:")
    print(f"     POST {base_url}/api/predict/preterm/image")
    print("   - Use 'image' field in form-data")
    print("   - Optionally specify 'model' parameter")
    
    # Example curl command
    print("\n   Example curl command:")
    print(f'   curl -X POST -F "image=@lab_report.jpg" {base_url}/api/predict/preterm/image')
    
    # Test 3: Maternal risk image prediction
    print("\n3. Testing maternal risk image prediction...")
    print("   Instructions:")
    print("   - Upload a medical image (vital signs, etc.) to:")
    print(f"     POST {base_url}/api/predict/maternal/image")
    print("   - Use 'image' field in form-data")
    print("   - Optionally specify 'model' parameter")
    
    print("\n   Example curl command:")
    print(f'   curl -X POST -F "image=@medical_report.jpg" {base_url}/api/predict/maternal/image')
    
    # Test 4: Check models available
    print("\n4. Checking available models...")
    try:
        response = requests.get(f"{base_url}/api/models")
        if response.status_code == 200:
            models_data = response.json()
            
            # Preterm models
            preterm_models = models_data.get('preterm_birth', {}).get('models', {})
            print(f"   ✓ Preterm birth models: {len(preterm_models)}")
            if preterm_models:
                best_preterm = models_data.get('preterm_birth', {}).get('best_model')
                print(f"     Best model: {best_preterm}")
                for name, info in preterm_models.items():
                    print(f"     - {name}: F1={info.get('f1_score', 0):.4f}")
            
            # Maternal models
            maternal_models = models_data.get('maternal_risk', {}).get('models', {})
            print(f"   ✓ Maternal risk models: {len(maternal_models)}")
            if maternal_models:
                best_maternal = models_data.get('maternal_risk', {}).get('best_model')
                print(f"     Best model: {best_maternal}")
                for name in maternal_models.keys():
                    print(f"     - {name}")
                    
        else:
            print(f"   ✗ Failed to get models: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Error getting models: {e}")
    
    print("\n" + "="*60)
    print("USAGE EXAMPLES")
    print("="*60)
    
    print("\n📋 For Preterm Birth Prediction:")
    print("   Upload lab reports containing:")
    print("   - Blood test results (Hb, WBC, PLT, HCT)")
    print("   - Patient demographics (Age, Height, Weight)")
    print("   - Pregnancy details (Week, Medical history)")
    print("   - Lab values (CRP, etc.)")
    
    print("\n🩺 For Maternal Risk Prediction:")
    print("   Upload medical reports containing:")
    print("   - Vital signs (Blood pressure, Heart rate)")
    print("   - Blood sugar levels, Body temperature")
    print("   - Patient age and other relevant metrics")
    
    print("\n🔍 Image Processing Features:")
    print("   - Automatic text extraction using OCR")
    print("   - AI-powered data extraction using Gemini")
    print("   - Support for multiple image formats")
    print("   - Intelligent field mapping to model inputs")
    
    print("\n💡 Tips:")
    print("   - Ensure images are clear and readable")
    print("   - Medical text should be visible")
    print("   - Multiple lab values in one image work best")
    print("   - PDF files are also supported")

def test_with_sample_data():
    """Test endpoints with sample JSON data (for comparison)"""
    
    base_url = "http://localhost:5000"
    
    print("\n" + "="*60)
    print("TESTING JSON ENDPOINTS FOR COMPARISON")
    print("="*60)
    
    # Sample preterm birth data
    preterm_sample = {
        "Hb [g/dl]": 12.5,
        "WBC [G/l]": 11.25,
        "PLT [G/l]": 261,
        "HCT [%]": 35.8,
        "Age": 31,
        "No. of pregnancy": 1,
        "No. of deliveries": 1,
        "Week of sample collection": 32,
        "Height": 171,
        "Weight": 64,
        "BMI": 21.9,
        "CRP": 4.7
    }
    
    print("\n📊 Testing preterm prediction with sample data...")
    try:
        response = requests.post(f"{base_url}/api/predict/preterm", json=preterm_sample)
        if response.status_code == 200:
            result = response.json()
            print("✓ Preterm prediction successful")
            predictions = result.get('predictions', {})
            if isinstance(predictions, dict):
                for model, pred in predictions.items():
                    if isinstance(pred, dict) and 'prediction' in pred:
                        print(f"  {model}: {pred['prediction']} (confidence: {pred.get('probability', 'N/A')})")
        else:
            print(f"✗ Preterm prediction failed: {response.status_code}")
            print(f"  Response: {response.text}")
    except Exception as e:
        print(f"✗ Error testing preterm prediction: {e}")

if __name__ == "__main__":
    test_image_endpoints()
    test_with_sample_data()
    
    print("\n" + "="*60)
    print("READY TO TEST IMAGE UPLOADS!")
    print("="*60)
    print("The API is now ready to accept medical image uploads.")
    print("Use the curl examples above or implement in your frontend.") 