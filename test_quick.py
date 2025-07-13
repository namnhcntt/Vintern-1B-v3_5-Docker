#!/usr/bin/env python3
import requests

def quick_test():
    url = "http://localhost:8000/api/upload-image"
    
    files = {
        'file': ('test_invoice.pdf', open('test_invoice.pdf', 'rb'), 'application/pdf')
    }
    
    data = {
        'prompt': 'Trích xuất thông tin hóa đơn',
        'template': 'invoice',
        'max_tokens': 500,
        'stream': 'false'
    }
    
    try:
        print("Testing template_used fix...")
        response = requests.post(url, files=files, data=data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Template used: {result.get('template_used', 'None')}")
            print(f"✅ Pages: {result.get('pages', 'None')}")
        else:
            print(f"❌ Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    finally:
        files['file'][1].close()

if __name__ == "__main__":
    quick_test()
