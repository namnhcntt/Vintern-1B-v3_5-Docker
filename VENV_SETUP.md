# Virtual Environment Setup for Vintern-1B Project

## Tổng Quan
Project này đã được cấu hình để sử dụng virtual environment (venv) để cô lập các dependencies và đảm bảo môi trường phát triển nhất quán.

## Cài Đặt

### 1. Tự Động (Khuyến Nghị)

#### Trên Windows:
```bash
# Chạy script thiết lập tự động
./setup_venv.sh
# hoặc
setup_venv.bat
```

#### Trên Linux/macOS:
```bash
# Chạy script thiết lập tự động
./setup_venv.sh
```

### 2. Thủ Công

```bash
# Tạo virtual environment
python -m venv venv

# Kích hoạt virtual environment
# Trên Windows (Git Bash/MSYS2):
source venv/Scripts/activate
# Trên Linux/macOS:
source venv/bin/activate

# Cài đặt dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Sử Dụng

### Chạy Demo
```bash
# Script run_demo.sh đã được cập nhật để tự động sử dụng virtual environment
./run_demo.sh
```

### Kích Hoạt Virtual Environment Thủ Công
```bash
# Trên Windows (Git Bash/MSYS2):
source venv/Scripts/activate

# Trên Linux/macOS:
source venv/bin/activate

# Thoát khỏi virtual environment:
deactivate
```

## Cấu Trúc Thư Mục

```
project/
├── venv/                    # Virtual environment
│   ├── Scripts/            # Windows executables
│   ├── bin/               # Unix executables  
│   ├── lib/               # Installed packages
│   └── ...
├── app.py
├── requirements.txt
├── run_demo.sh            # Script đã được cập nhật để sử dụng venv
├── setup_venv.sh          # Script thiết lập venv cho Unix/Git Bash
├── setup_venv.bat         # Script thiết lập venv cho Windows
└── ...
```

## Ghi Chú

1. **Virtual Environment**: Tất cả dependencies sẽ được cài đặt trong thư mục `venv/` thay vì system-wide
2. **Script Update**: `run_demo.sh` đã được cập nhật để:
   - Tự động tạo virtual environment nếu chưa tồn tại
   - Kích hoạt virtual environment trước khi chạy
   - Sử dụng Python và pip từ virtual environment
   - Tự động deactivate khi script kết thúc

3. **Dependencies**: File `requirements.txt` đã được cập nhật với phiên bản tương thích:
   - torch >= 2.2.0 (thay vì 2.1.2 không còn available)
   - Các packages khác với phiên bản tương thích

## Xử Lý Sự Cố

### Lỗi Python không tìm thấy:
- Đảm bảo Python 3.8+ đã được cài đặt
- Thêm Python vào PATH

### Lỗi tạo virtual environment:
```bash
# Cài đặt module venv nếu cần
python -m pip install --upgrade pip
```

### Lỗi permissions trên Linux/macOS:
```bash
# Cấp quyền execute cho scripts
chmod +x setup_venv.sh run_demo.sh
```

## Requirements

- Python 3.8+
- pip
- venv module (thường đi kèm với Python)
- Ít nhất 3GB dung lượng trống cho dependencies (đặc biệt là PyTorch)
