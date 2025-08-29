#!/usr/bin/env python3
"""
Simple test script for the Face Analytics API.
"""

import requests
import sys
from pathlib import Path

def test_health():
    """Test the health endpoint"""
    try:
        response = requests.get("http://localhost:8000/health")
        print(f"Health check: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_analyze_face(image_path: str):
    """Test face analysis with an image"""
    if not Path(image_path).exists():
        print(f"Image file not found: {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post("http://localhost:8000/analyze", files=files)
        
        print(f"Analysis status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Success: {result['success']}")
            print(f"Faces detected: {len(result['faces'])}")
            
            for i, face in enumerate(result['faces']):
                print(f"  Face {i+1}:")
                print(f"    Bounding box: {face['bbox']}")
                print(f"    Confidence: {face['confidence']:.2f}")
                print(f"    Age: {face.get('age', 'N/A')}")
                print(f"    Gender: {face.get('gender', 'N/A')} ({face.get('gender_confidence', 0):.2f})")
                print(f"    Identity: {face.get('identity', 'N/A')}")
                if face.get('person_id'):
                    print(f"    Person ID: {face['person_id']}")
        else:
            print(f"Error: {response.text}")
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"Face analysis failed: {e}")
        return False

def test_infer_age_genre(image_path: str):
    """Test age and gender inference with an image"""
    if not Path(image_path).exists():
        print(f"Image file not found: {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post("http://localhost:8000/infer_age_genre", files=files)
        
        print(f"Age/Gender inference status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Success: {result['success']}")
            print(f"Faces analyzed: {len(result['faces'])}")
            
            for i, face in enumerate(result['faces']):
                print(f"  Face {i+1}:")
                print(f"    Bounding box: {face['bbox']}")
                print(f"    Detection confidence: {face['confidence']:.2f}")
                print(f"    Age: {face['age']:.1f} years")
                print(f"    Gender: {face['gender']} (confidence: {face['gender_confidence']:.2f})")
        else:
            print(f"Error: {response.text}")
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"Age/Gender inference failed: {e}")
        return False

def test_stats():
    """Test the statistics endpoint"""
    try:
        response = requests.get("http://localhost:8000/stats")
        print(f"Stats status: {response.status_code}")
        
        if response.status_code == 200:
            stats = response.json()
            print("Statistics:")
            print(f"  Known faces: {stats.get('known_faces', 0)}")
            print(f"  Unknown faces: {stats.get('unknown_faces', 0)}")
            print(f"  Total detections: {stats.get('total_detections', 0)}")
            print(f"  Recent detections (24h): {stats.get('recent_detections_24h', 0)}")
        else:
            print(f"Error: {response.text}")
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"Stats test failed: {e}")
        return False

def main():
    print("=== Face Analytics API Test ===")
    
    # Test health endpoint
    print("\n1. Testing health endpoint...")
    if not test_health():
        print("❌ Health check failed. Make sure the API is running.")
        return 1
    print("✅ Health check passed")
    
    # Test stats endpoint
    print("\n2. Testing stats endpoint...")
    if not test_stats():
        print("❌ Stats test failed")
        return 1
    print("✅ Stats test passed")
    
    # Test face analysis if image provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
        print(f"\n3. Testing face analysis with: {image_path}")
        if not test_analyze_face(image_path):
            print("❌ Face analysis test failed")
            return 1
        print("✅ Face analysis test passed")
        
        print(f"\n4. Testing age/gender inference with: {image_path}")
        if not test_infer_age_genre(image_path):
            print("❌ Age/Gender inference test failed")
            return 1
        print("✅ Age/Gender inference test passed")
        
    else:
        print("\n3. Skipping image tests (no image provided)")
        print("   Usage: python test_api.py <path_to_image>")
        print("   This will test both /analyze and /infer_age_genre endpoints")
    
    print("\n🎉 All tests passed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())