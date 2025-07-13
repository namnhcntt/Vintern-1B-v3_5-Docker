#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

def create_test_pdf():
    """Create a test PDF with Vietnamese text for OCR testing"""
    
    # Create PDF
    pdf_filename = "test_invoice.pdf"
    c = canvas.Canvas(pdf_filename, pagesize=A4)
    width, height = A4
    
    # Set up fonts (using default fonts that support Unicode)
    c.setFont("Helvetica-Bold", 16)
    
    # Title
    c.drawCentredString(width/2, height - 2*cm, "HÓA ĐƠN BÁN HÀNG")
    c.drawCentredString(width/2, height - 2.5*cm, "SALES INVOICE")
    
    # Invoice details
    c.setFont("Helvetica", 12)
    c.drawString(2*cm, height - 4*cm, "Số hóa đơn: HD-2024-001")
    c.drawString(2*cm, height - 4.5*cm, "Ngày: 15/01/2024")
    
    # Seller information
    c.setFont("Helvetica-Bold", 14)
    c.drawString(2*cm, height - 6*cm, "Thông tin người bán:")
    c.setFont("Helvetica", 11)
    c.drawString(2*cm, height - 6.5*cm, "Công ty: ABC Technology Co., Ltd")
    c.drawString(2*cm, height - 7*cm, "Địa chỉ: 123 Nguyen Van Cu St, District 1, HCMC")
    c.drawString(2*cm, height - 7.5*cm, "Mã số thuế: 0123456789")
    c.drawString(2*cm, height - 8*cm, "Điện thoại: (028) 1234-5678")
    
    # Buyer information
    c.setFont("Helvetica-Bold", 14)
    c.drawString(2*cm, height - 9.5*cm, "Thông tin người mua:")
    c.setFont("Helvetica", 11)
    c.drawString(2*cm, height - 10*cm, "Khách hàng: XYZ Solutions Ltd")
    c.drawString(2*cm, height - 10.5*cm, "Địa chỉ: 456 Le Loi St, District 3, HCMC")
    c.drawString(2*cm, height - 11*cm, "Mã số thuế: 9876543210")
    
    # Table header
    c.setFont("Helvetica-Bold", 12)
    y_pos = height - 13*cm
    c.drawString(2*cm, y_pos, "STT")
    c.drawString(3*cm, y_pos, "Tên hàng hóa")
    c.drawString(8*cm, y_pos, "Đơn vị")
    c.drawString(10*cm, y_pos, "Số lượng")
    c.drawString(12*cm, y_pos, "Đơn giá")
    c.drawString(15*cm, y_pos, "Thành tiền")
    
    # Draw line under header
    c.line(2*cm, y_pos - 0.3*cm, 18*cm, y_pos - 0.3*cm)
    
    # Table data
    c.setFont("Helvetica", 10)
    items = [
        ("1", "Laptop Dell Inspiron 15", "Chiếc", "2", "15,000,000", "30,000,000"),
        ("2", "Chuột không dây Logitech", "Chiếc", "5", "500,000", "2,500,000"),
        ("3", "Bàn phím cơ Gaming", "Chiếc", "3", "1,200,000", "3,600,000")
    ]
    
    y_pos -= 0.8*cm
    for item in items:
        c.drawString(2*cm, y_pos, item[0])
        c.drawString(3*cm, y_pos, item[1])
        c.drawString(8*cm, y_pos, item[2])
        c.drawString(10*cm, y_pos, item[3])
        c.drawString(12*cm, y_pos, item[4])
        c.drawString(15*cm, y_pos, item[5])
        y_pos -= 0.5*cm
    
    # Draw line after items
    c.line(2*cm, y_pos, 18*cm, y_pos)
    
    # Summary
    y_pos -= 1*cm
    c.setFont("Helvetica-Bold", 11)
    c.drawString(12*cm, y_pos, "Tổng tiền hàng:")
    c.drawString(15*cm, y_pos, "36,100,000 VNĐ")
    
    y_pos -= 0.5*cm
    c.drawString(12*cm, y_pos, "Thuế VAT (10%):")
    c.drawString(15*cm, y_pos, "3,610,000 VNĐ")
    
    y_pos -= 0.5*cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(12*cm, y_pos, "Tổng cộng:")
    c.drawString(15*cm, y_pos, "39,710,000 VNĐ")
    
    # Notes
    y_pos -= 1.5*cm
    c.setFont("Helvetica", 10)
    c.drawString(2*cm, y_pos, "Ghi chú: Thanh toán trong vòng 30 ngày kể từ ngày xuất hóa đơn")
    y_pos -= 0.5*cm
    c.drawString(2*cm, y_pos, "Phương thức thanh toán: Chuyển khoản ngân hàng")
    
    # Signatures
    y_pos -= 2*cm
    c.setFont("Helvetica-Bold", 11)
    c.drawString(4*cm, y_pos, "Người mua hàng")
    c.drawString(14*cm, y_pos, "Người bán hàng")
    
    y_pos -= 0.5*cm
    c.setFont("Helvetica", 9)
    c.drawString(4*cm, y_pos, "(Ký, ghi rõ họ tên)")
    c.drawString(14*cm, y_pos, "(Ký, ghi rõ họ tên)")
    
    # Save PDF
    c.save()
    print(f"Created test PDF: {pdf_filename}")
    return pdf_filename

if __name__ == "__main__":
    create_test_pdf()
