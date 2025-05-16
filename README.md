# Vintern-1B Image-to-Text Demo

Đây là bản demo của mô hình Vintern-1B-v3.5, có khả năng trích xuất văn bản từ hình ảnh và chuyển đổi sang định dạng markdown. Mô hình này đặc biệt tốt trong việc hiểu văn bản tiếng Việt, OCR và hiểu tài liệu.

## Tính năng

- Giao diện web đơn giản để chuyển đổi hình ảnh thành văn bản
- Tải lên file bằng kéo thả
- Tùy chỉnh prompt
- Cấu hình thiết lập API
- Chức năng sao chép vào clipboard
- Hỗ trợ streaming kết quả theo thời gian thực

## Yêu cầu

- Python 3.8+
- PyTorch
- FastAPI
- Transformers
- Các phụ thuộc khác được liệt kê trong `requirements.txt`

## Cài đặt và chạy

### Cài đặt thông thường

1. Cài đặt các phụ thuộc cần thiết:

```bash
pip install -r requirements.txt
```

2. Khởi động máy chủ API:

```bash
python app.py
```

Máy chủ sẽ chạy tại http://localhost:8000.

### Sử dụng script tự động

Bạn có thể sử dụng script `run_demo.sh` để tự động cài đặt và chạy demo:

```bash
chmod +x run_demo.sh
./run_demo.sh
```

Script này sẽ:
1. Kiểm tra và cài đặt các phụ thuộc cần thiết
2. Tìm cổng trống nếu cổng 8000 đã được sử dụng
3. Khởi động máy chủ API
4. Mở trình duyệt web với giao diện người dùng

### Tối ưu hóa cho Apple Silicon

Nếu bạn đang sử dụng Mac với chip Apple Silicon (M1/M2/M3), bạn có thể sử dụng script `fix_transformers.sh` để tối ưu hóa hiệu suất:

```bash
chmod +x fix_transformers.sh
./fix_transformers.sh
```

Script này sẽ:
- Phát hiện nếu bạn đang sử dụng Apple Silicon
- Cài đặt PyTorch và TorchVision được tối ưu hóa cho Apple Silicon
- Cài đặt transformers phiên bản 4.38.0 trở lên
- Xác minh MPS (Metal Performance Shaders) có sẵn trên hệ thống của bạn

## Sử dụng Docker

### Yêu cầu

- Docker đã được cài đặt trên máy tính của bạn
- Khoảng 2GB dung lượng trống cho Docker image

### Đóng gói thành Docker image

#### Sử dụng script tự động

Cách đơn giản nhất để đóng gói ứng dụng là sử dụng script `build_docker.sh`:

```bash
chmod +x build_docker.sh
./build_docker.sh
```

Script này sẽ:
1. Kiểm tra các file cần thiết
2. Tự động phát hiện phần cứng của bạn và sử dụng Dockerfile phù hợp
3. Đóng gói ứng dụng thành Docker image với tên `vintern-image-to-text:latest`

#### Đóng gói thủ công

Nếu bạn muốn đóng gói thủ công, bạn có thể sử dụng lệnh sau:

```bash
docker build -t vintern-image-to-text:latest .
```

### Đóng gói cho nhiều kiến trúc

Để đóng gói ứng dụng cho nhiều kiến trúc (ARM64 và AMD64), bạn có thể sử dụng script `build_multiarch_docker.sh`:

```bash
chmod +x build_multiarch_docker.sh
./build_multiarch_docker.sh --all
```

Script này sẽ:
1. Tạo Docker image cho ARM64 (Apple Silicon)
2. Tạo Docker image cho AMD64 (Intel/AMD)
3. Tạo Docker image cho NVIDIA CUDA
4. Tạo manifest đa kiến trúc (nếu sử dụng tùy chọn `--push`)

Các tùy chọn có sẵn:
- `--arm64`: Build cho kiến trúc ARM64
- `--amd64`: Build cho kiến trúc AMD64
- `--cuda`: Build cho NVIDIA CUDA
- `--all`: Build cho tất cả các kiến trúc
- `--push`: Đẩy image lên Docker Hub (yêu cầu đăng nhập)

### Chạy Docker container

#### Sử dụng script tự động

Cách đơn giản nhất để chạy container là sử dụng script `run_docker.sh`:

```bash
chmod +x run_docker.sh
./run_docker.sh
```

Script này sẽ:
1. Kiểm tra xem Docker image đã tồn tại chưa
2. Dừng và xóa container cũ nếu có
3. Kiểm tra xem cổng 8000 đã được sử dụng chưa
4. Tự động ánh xạ thư mục cache Hugging Face
5. Chạy container mới với cấu hình phù hợp với phần cứng của bạn

#### Chạy thủ công

Nếu bạn muốn chạy container thủ công, bạn có thể sử dụng lệnh sau:

```bash
docker run --name vintern-demo -p 8000:8000 -d vintern-image-to-text:latest
```

Sau đó, bạn có thể truy cập ứng dụng tại:
- Web UI: http://localhost:8000
- API: http://localhost:8000/api

### Sử dụng Volume Mapping

Để tránh việc phải tải lại các file mô hình mỗi khi khởi động container, bạn có thể sử dụng volume mapping:

```bash
docker run -v $HOME/.cache/huggingface:/root/.cache/huggingface -p 8000:8000 vintern-image-to-text:latest
```

Script `run_docker.sh` đã được cập nhật để tự động tạo volume mapping. Khi bạn chạy script này, nó sẽ:

1. Tự động phát hiện thư mục cache Hugging Face trên máy host
2. Tạo thư mục này nếu nó chưa tồn tại
3. Ánh xạ thư mục này vào thư mục `/root/.cache/huggingface` trong container

Bạn có thể chỉ định một thư mục cache khác bằng cách đặt biến môi trường `MODEL_CACHE_DIR`:

```bash
MODEL_CACHE_DIR=/path/to/custom/cache ./run_docker.sh
```

## Tính năng Streaming

Streaming cho phép API trả về kết quả từng phần theo thời gian thực, thay vì phải đợi đến khi mô hình tạo ra toàn bộ văn bản. Điều này mang lại trải nghiệm người dùng tốt hơn, đặc biệt là với các câu trả lời dài.

### Cấu hình Streaming

#### Thông qua biến môi trường

Bạn có thể cấu hình streaming thông qua biến môi trường:

```bash
# Bật streaming mặc định
export VINTERN_STREAM=true

# Các biến môi trường khác
export VINTERN_MAX_TOKENS=1024
export VINTERN_TEMPERATURE=0.0
export VINTERN_DO_SAMPLE=false
export VINTERN_NUM_BEAMS=3
export VINTERN_REPETITION_PENALTY=2.5
```

#### Khi chạy Docker

Khi chạy Docker container, bạn có thể truyền biến môi trường:

```bash
docker run -e VINTERN_STREAM=true -p 8000:8000 vintern-image-to-text:latest
```

Hoặc sử dụng script `run_docker.sh` với biến môi trường:

```bash
VINTERN_STREAM=true ./run_docker.sh
```

#### Trong API Request

Bạn cũng có thể bật streaming cho từng request cụ thể bằng cách thêm tham số `stream` vào request:

```json
{
  "prompt": "Trích xuất thông tin chính trong ảnh và trả về dạng markdown.",
  "image": "base64_encoded_image_data",
  "stream": true
}
```

## Phần cứng và hiệu suất

### Độ ưu tiên phần cứng

Ứng dụng sẽ tự động phát hiện và sử dụng phần cứng theo độ ưu tiên sau:

#### Trên macOS:
1. Apple Silicon (MPS) - cho Mac M1/M2/M3
2. NVIDIA GPU (CUDA) - cho Mac với GPU NVIDIA (thông qua eGPU)
3. CPU - khi không có GPU

#### Trên Windows/Linux:
1. NVIDIA GPU (CUDA) - cho hệ thống có GPU NVIDIA
2. AMD GPU (ROCm) - cho hệ thống có GPU AMD (hỗ trợ giới hạn)
3. CPU - khi không có GPU

### Docker Images

Dự án cung cấp ba Dockerfile khác nhau để tối ưu hóa cho từng loại phần cứng:

1. **Dockerfile.arm64**: Tối ưu cho Apple Silicon (ARM64)
2. **Dockerfile.cuda**: Tối ưu cho NVIDIA GPU với CUDA
3. **Dockerfile.amd64**: Phiên bản tiêu chuẩn cho CPU (Intel/AMD)

Script `build_docker.sh` sẽ tự động phát hiện phần cứng của bạn và sử dụng Dockerfile phù hợp.

### Phát hiện phần cứng

Ứng dụng sử dụng script `detect_hardware.py` để phát hiện phần cứng có sẵn. Bạn có thể chạy script này để xem thông tin về phần cứng của mình:

```bash
python detect_hardware.py
```

Kết quả sẽ hiển thị thông tin về phần cứng được phát hiện và phần cứng tốt nhất sẽ được sử dụng.

### Yêu cầu hệ thống

#### Apple Silicon:
- macOS 12.0 (Monterey) trở lên
- Chip Apple M1, M2 hoặc M3
- Ít nhất 8GB RAM (khuyến nghị 16GB hoặc cao hơn)

#### NVIDIA GPU:
- NVIDIA GPU với CUDA Compute Capability 3.5 trở lên
- Driver NVIDIA phiên bản 450.80.02 trở lên
- Ít nhất 4GB VRAM (khuyến nghị 8GB hoặc cao hơn)
- Đối với Docker: NVIDIA Container Toolkit đã được cài đặt

#### AMD GPU:
- Hỗ trợ giới hạn thông qua ROCm
- Yêu cầu cài đặt thủ công ROCm và PyTorch với hỗ trợ ROCm

#### CPU:
- Bất kỳ CPU x86_64 hiện đại nào
- Ít nhất 8GB RAM (khuyến nghị 16GB hoặc cao hơn)

## API Endpoints

### `/api`

Endpoint này cung cấp thông tin về API.

### `/api/image-to-text`

Endpoint này nhận dữ liệu JSON với hình ảnh được mã hóa base64 và prompt.

Ví dụ request:

```json
{
  "prompt": "Trích xuất thông tin chính trong ảnh và trả về dạng markdown.",
  "image": "base64_encoded_image_data",
  "max_tokens": 1024,
  "temperature": 0.0,
  "do_sample": false,
  "num_beams": 3,
  "repetition_penalty": 2.5,
  "stream": false
}
```

### `/api/upload-image`

Endpoint này nhận form data với file upload và tham số prompt.

## Xử lý sự cố

### Container không khởi động

Nếu container không khởi động, bạn có thể kiểm tra logs:

```bash
docker logs vintern-demo
```

### Cổng 8000 đã được sử dụng

Nếu cổng 8000 đã được sử dụng, bạn có thể chỉ định một cổng khác:

```bash
docker run --name vintern-demo -p 8080:8000 -d vintern-image-to-text:latest
```

Sau đó, bạn có thể truy cập ứng dụng tại http://localhost:8080

### Lỗi "context deadline exceeded" khi chạy build_multiarch_docker.sh

Nếu bạn gặp lỗi "context deadline exceeded" khi truy cập docker.sock, hãy thử các giải pháp sau:

1. Khởi động lại Docker daemon:
   ```bash
   # Trên macOS
   killall Docker && open /Applications/Docker.app

   # Trên Linux
   sudo systemctl restart docker
   ```

2. Xóa builder instance hiện tại và tạo lại:
   ```bash
   docker buildx rm multiarch-builder
   docker buildx create --name multiarch-builder --driver docker-container --use
   ```

3. Kiểm tra quyền truy cập vào socket Docker:
   ```bash
   # Trên Linux
   sudo chmod 666 /var/run/docker.sock
   ```

4. Nếu vẫn gặp lỗi, hãy thử build cho từng kiến trúc riêng biệt:
   ```bash
   # Chỉ build cho ARM64
   ./build_multiarch_docker.sh --arm64

   # Hoặc chỉ build cho AMD64
   ./build_multiarch_docker.sh --amd64
   ```

### Lỗi "CUDA out of memory"

Nếu bạn gặp lỗi này, hãy thử:
1. Giảm kích thước ảnh đầu vào
2. Đóng các ứng dụng khác đang sử dụng GPU
3. Khởi động lại máy tính

### Lỗi "MPS operation not implemented"

Nếu bạn gặp lỗi này trên Apple Silicon, hãy đảm bảo biến môi trường `PYTORCH_ENABLE_MPS_FALLBACK=1` đã được thiết lập. Script `run_docker.sh` sẽ tự động thiết lập biến này.

### Lỗi "NVIDIA driver not found"

Nếu bạn gặp lỗi này, hãy đảm bảo:
1. Driver NVIDIA đã được cài đặt
2. NVIDIA Container Toolkit đã được cài đặt (cho Docker)
3. Lệnh `nvidia-smi` hoạt động bình thường

## Thông tin về mô hình

Mô hình Vintern-1B-v3.5 là một mô hình đa phương thức được fine-tune từ InternVL2.5-1B, với trọng tâm vào khả năng ngôn ngữ tiếng Việt. Nó xuất sắc trong:

- Nhận dạng văn bản
- OCR
- Hiểu tài liệu tiếng Việt
- Xử lý hóa đơn, văn bản pháp lý, chữ viết tay và bảng biểu

Để biết thêm thông tin về mô hình, hãy truy cập [trang Hugging Face](https://huggingface.co/5CD-AI/Vintern-1B-v3_5).

## Giấy phép

Mô hình Vintern-1B-v3.5 được phân phối theo giấy phép của 5CD-AI. Vui lòng tham khảo trang Hugging Face của mô hình để biết thêm chi tiết về giấy phép.
