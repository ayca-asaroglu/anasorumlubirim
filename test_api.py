"""
API Test Script

This script tests the Turkish Text Classification API endpoints.
"""

import requests
import json
import time
from typing import Dict, Any


class APITester:
    """Test class for the Turkish Text Classification API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the API tester."""
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_health_check(self) -> bool:
        """Test the health check endpoint."""
        print("🔍 Testing health check...")
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed: {data['message']}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {str(e)}")
            return False
    
    def test_model_info(self) -> bool:
        """Test the model info endpoint."""
        print("🔍 Testing model info...")
        try:
            response = self.session.get(f"{self.base_url}/model/info")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Model info retrieved:")
                print(f"   - Upper classes: {len(data['upper_level_classes'])}")
                print(f"   - Lower classes: {len(data['lower_level_classes'])}")
                print(f"   - Model version: {data['model_version']}")
                print(f"   - Is loaded: {data['is_loaded']}")
                return True
            else:
                print(f"❌ Model info failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Model info error: {str(e)}")
            return False
    
    def test_single_prediction(self) -> bool:
        """Test single text prediction."""
        print("🔍 Testing single prediction...")
        try:
            payload = {
                "text": "Sistemde bir hata oluştu ve kullanıcılar giriş yapamıyor. Lütfen bu sorunu çözün.",
                "summary": "Giriş sistemi hatası",
                "talep_tipi": "Teknik Destek",
                "reporter_birim": "IT Departmanı"
            }
            
            response = self.session.post(
                f"{self.base_url}/predict",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Single prediction successful:")
                print(f"   - Upper level: {data['upper_level_prediction']}")
                print(f"   - Lower level: {data['lower_level_prediction']}")
                print(f"   - Upper confidence: {data['upper_level_confidence']:.3f}")
                print(f"   - Lower confidence: {data['lower_level_confidence']:.3f}")
                print(f"   - Processing time: {data['processing_time_ms']:.1f}ms")
                return True
            else:
                print(f"❌ Single prediction failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Single prediction error: {str(e)}")
            return False
    
    def test_batch_prediction(self) -> bool:
        """Test batch text prediction."""
        print("🔍 Testing batch prediction...")
        try:
            payload = {
                "texts": [
                    {
                        "text": "Sistemde bir hata oluştu.",
                        "summary": "Sistem hatası",
                        "talep_tipi": "Teknik Destek"
                    },
                    {
                        "text": "Yeni kullanıcı kaydı gerekiyor.",
                        "summary": "Kullanıcı kaydı",
                        "talep_tipi": "İnsan Kaynakları"
                    },
                    {
                        "text": "Muhasebe raporu hazırlanması gerekiyor.",
                        "summary": "Muhasebe raporu",
                        "talep_tipi": "Finans"
                    }
                ]
            }
            
            response = self.session.post(
                f"{self.base_url}/predict/batch",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Batch prediction successful:")
                print(f"   - Number of predictions: {len(data['predictions'])}")
                print(f"   - Total processing time: {data['total_processing_time_ms']:.1f}ms")
                
                for i, pred in enumerate(data['predictions']):
                    print(f"   - Prediction {i+1}: {pred['upper_level_prediction']} -> {pred['lower_level_prediction']}")
                
                return True
            else:
                print(f"❌ Batch prediction failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Batch prediction error: {str(e)}")
            return False
    
    def test_classes_endpoints(self) -> bool:
        """Test the classes endpoints."""
        print("🔍 Testing classes endpoints...")
        try:
            # Test upper classes
            response = self.session.get(f"{self.base_url}/classes/upper")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Upper classes retrieved: {len(data['classes'])} classes")
            else:
                print(f"❌ Upper classes failed: {response.status_code}")
                return False
            
            # Test lower classes
            response = self.session.get(f"{self.base_url}/classes/lower")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Lower classes retrieved: {len(data['classes'])} classes")
                return True
            else:
                print(f"❌ Lower classes failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Classes endpoints error: {str(e)}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all API tests."""
        print("🧪 Starting API Tests...")
        print("=" * 50)
        
        tests = {
            "health_check": self.test_health_check(),
            "model_info": self.test_model_info(),
            "single_prediction": self.test_single_prediction(),
            "batch_prediction": self.test_batch_prediction(),
            "classes_endpoints": self.test_classes_endpoints()
        }
        
        print("\n" + "=" * 50)
        print("📊 Test Results:")
        passed = 0
        for test_name, result in tests.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"   {test_name}: {status}")
            if result:
                passed += 1
        
        print(f"\n🎯 Overall: {passed}/{len(tests)} tests passed")
        
        return tests


def main():
    """Main function to run API tests."""
    print("🚀 Turkish Text Classification API Tester")
    print("=" * 50)
    
    # Wait a moment for API to start
    print("⏳ Waiting for API to start...")
    time.sleep(2)
    
    tester = APITester()
    results = tester.run_all_tests()
    
    if all(results.values()):
        print("\n🎉 All tests passed! API is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the API server.")
    
    return results


if __name__ == "__main__":
    main()
