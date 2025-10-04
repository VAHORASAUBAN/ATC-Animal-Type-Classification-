#!/usr/bin/env python3
"""
Test script for the Cattle vs Buffalo Classification System
"""

import requests
import base64
import json
from pathlib import Path
import cv2
import numpy as np

def test_api_health():
    """Test if the API is running"""
    try:
        response = requests.get('http://localhost:5000/api/health')
        if response.status_code == 200:
            print("✅ API is running")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure Flask server is running.")
        return False

def test_model_info():
    """Test model information endpoint"""
    try:
        response = requests.get('http://localhost:5000/api/model_info')
        if response.status_code == 200:
            model_info = response.json()
            print("✅ Model information retrieved")
            print(f"   Model type: {model_info.get('model_type', 'Unknown')}")
            print(f"   Classes: {list(model_info.get('class_mapping', {}).keys())}")
            return True
        else:
            print(f"❌ Model info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error getting model info: {e}")
        return False

def create_test_image():
    """Create a simple test image"""
    # Create a simple test image (white background with a dark shape)
    img = np.ones((224, 224, 3), dtype=np.uint8) * 255  # White background
    
    # Add a dark rectangle (simulating an animal)
    cv2.rectangle(img, (50, 50), (174, 174), (50, 50, 50), -1)
    
    # Encode as base64
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return f"data:image/jpeg;base64,{img_base64}"

def test_classification():
    """Test image classification"""
    try:
        test_image = create_test_image()
        
        payload = {
            "image": test_image,
            "bbox": None
        }
        
        response = requests.post(
            'http://localhost:5000/api/classify',
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Classification successful")
            print(f"   Prediction: {result.get('prediction', 'Unknown')}")
            print(f"   Confidence: {result.get('confidence', 0):.3f}")
            print(f"   Breed: {result.get('breed', 'Unknown')}")
            print(f"   Probabilities: {result.get('probabilities', {})}")
            return True
        else:
            print(f"❌ Classification failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error in classification test: {e}")
        return False

def test_with_real_image():
    """Test with a real image from the dataset"""
    dataset_path = Path("cow-and-buffalo.v1i.tensorflow")
    
    # Find a test image
    test_images = list(dataset_path.glob("test/*.jpg"))
    if not test_images:
        print("❌ No test images found")
        return False
    
    test_image_path = test_images[0]
    print(f"Testing with real image: {test_image_path.name}")
    
    try:
        # Load and encode image
        img = cv2.imread(str(test_image_path))
        if img is None:
            print("❌ Could not load test image")
            return False
        
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        payload = {
            "image": f"data:image/jpeg;base64,{img_base64}",
            "bbox": None
        }
        
        response = requests.post(
            'http://localhost:5000/api/classify',
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Real image classification successful")
            print(f"   Prediction: {result.get('prediction', 'Unknown')}")
            print(f"   Confidence: {result.get('confidence', 0):.3f}")
            print(f"   Breed: {result.get('breed', 'Unknown')}")
            return True
        else:
            print(f"❌ Real image classification failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing real image: {e}")
        return False

def main():
    """Run all tests"""
    print("Cattle vs Buffalo Classification System Test")
    print("=" * 50)
    
    tests = [
        ("API Health Check", test_api_health),
        ("Model Information", test_model_info),
        ("Synthetic Image Classification", test_classification),
        ("Real Image Classification", test_with_real_image)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"   Test failed!")
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the system setup.")
    
    return passed == total

if __name__ == "__main__":
    main()
