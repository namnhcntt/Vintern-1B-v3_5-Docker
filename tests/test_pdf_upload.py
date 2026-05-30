#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import requests
import json
import time

FIXTURE_PDF = Path(__file__).parent / "fixtures" / "test_invoice.pdf"

def test_pdf_upload():
    """Test PDF upload functionality"""
    
    # API endpoint
    url = "http://localhost:8000/api/upload-image"
    
    # Test with different templates
    templates = [
        ("", "Không sử dụng template"),
        ("invoice", "Template hóa đơn"),
        ("json", "Template JSON"),
        ("table", "Template bảng")
    ]
    
    for template, description in templates:
        print(f"\n{'='*60}")
        print(f"Testing: {description}")
        print(f"{'='*60}")
        
        # Prepare form data
        files = {
            'file': ('test_invoice.pdf', open(FIXTURE_PDF, 'rb'), 'application/pdf')
        }
        
        data = {
            'prompt': 'Trích xuất thông tin từ hóa đơn này',
            'max_tokens': 2000,
            'temperature': 0.1,
            'num_beams': 1,
            'stream': 'false'
        }
        
        if template:
            data['template'] = template
        
        try:
            print(f"Đang gửi request với template: {template or 'None'}")
            start_time = time.time()
            
            response = requests.post(url, files=files, data=data, timeout=300)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            print(f"Thời gian xử lý: {processing_time:.2f} giây")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Thành công!")
                print(f"Template sử dụng: {result.get('template_used', 'None')}")
                if 'pages' in result:
                    print(f"Số trang xử lý: {result['pages']}")
                print(f"Kết quả OCR:")
                print("-" * 40)
                print(result['text'][:500] + "..." if len(result['text']) > 500 else result['text'])
                print("-" * 40)
            else:
                print(f"❌ Lỗi: {response.status_code}")
                print(f"Chi tiết: {response.text}")
                
        except requests.exceptions.Timeout:
            print("❌ Timeout - Request quá lâu")
        except Exception as e:
            print(f"❌ Lỗi: {str(e)}")
        finally:
            # Close file
            files['file'][1].close()
        
        # Wait between tests
        if template != templates[-1][0]:
            print("\nĐợi 5 giây trước test tiếp theo...")
            time.sleep(5)

def test_image_endpoint():
    """Test image endpoint with base64"""
    print(f"\n{'='*60}")
    print("Testing Image Endpoint với template")
    print(f"{'='*60}")
    
    # Convert PDF to base64 for testing (this won't work for PDF, but tests the endpoint)
    try:
        with open(FIXTURE_PDF, 'rb') as f:
            import base64
            pdf_data = f.read()
            # This is just for testing - PDF won't work with image endpoint
            base64_data = f"data:application/pdf;base64,{base64.b64encode(pdf_data).decode()}"
        
        url = "http://localhost:8000/api/image-to-text"
        data = {
            "prompt": "Trích xuất thông tin hóa đơn",
            "image": base64_data,
            "template": "invoice",
            "max_tokens": 1000,
            "stream": False
        }
        
        response = requests.post(url, json=data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Image endpoint hoạt động với template")
            print(f"Template sử dụng: {result.get('template_used', 'None')}")
        else:
            print(f"⚠️ Image endpoint không hỗ trợ PDF (expected): {response.status_code}")
            
    except Exception as e:
        print(f"⚠️ Test image endpoint: {str(e)}")

if __name__ == "__main__":
    print("🚀 Bắt đầu test tính năng OCR PDF")
    
    # Test PDF upload
    test_pdf_upload()
    
    # Test image endpoint
    test_image_endpoint()
    
    print(f"\n{'='*60}")
    print("✅ Hoàn thành test!")
    print(f"{'='*60}")
