#!/usr/bin/env python3
import requests

def test_template_debug():
    url = "http://localhost:8000/api/upload-image"
    
    # Test với template rõ ràng
    files = {
        'file': ('test_invoice.pdf', open('test_invoice.pdf', 'rb'), 'application/pdf')
    }
    
    data = {
        'prompt': 'Test template',
        'template': 'json',  # Rõ ràng là json
        'max_tokens': 100,
        'stream': 'false'
    }
    
    try:
        print("Sending request with explicit template...")
        print(f"Data being sent: {data}")
        
        response = requests.post(url, files=files, data=data, timeout=60)
        
        print(f"Response status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Template used in response: {result.get('template_used', 'NOT_FOUND')}")
            print(f"Full response keys: {list(result.keys())}")
        else:
            print(f"Error response: {response.text}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        files['file'][1].close()

if __name__ == "__main__":
    test_template_debug()
