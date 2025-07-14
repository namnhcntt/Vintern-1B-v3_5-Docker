# Changelog

Tất cả các thay đổi đáng chú ý của dự án này sẽ được ghi lại trong file này.

Định dạng dựa trên [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
và dự án này tuân theo [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2025-01-14

### Added
- Thống nhất tên Docker image thành `namnhcntt/vintern-image-to-text` trên tất cả các script
- Thêm hỗ trợ tag phiên bản cho Docker images
- Cải thiện script build để tạo nhiều tag cho cùng một image:
  - `latest`: Tag mặc định cho phiên bản mới nhất
  - `1.1.0`: Tag phiên bản cụ thể
  - `1.1.0-amd64`: Tag cho kiến trúc AMD64
  - `1.1.0-arm64`: Tag cho kiến trúc ARM64  
  - `1.1.0-cuda`: Tag cho NVIDIA CUDA

### Changed
- Cập nhật `build_docker.sh` để hỗ trợ tag phiên bản
- Cập nhật `build_multiarch_docker.sh` để tạo manifest đa kiến trúc với tag phiên bản
- Cập nhật `run_docker.sh` để sử dụng tên image thống nhất
- Cải thiện quy trình build và deployment Docker images

### Fixed
- Thống nhất tên Docker image giữa các script khác nhau
- Cải thiện tính nhất quán trong việc đặt tên và tag Docker images

## [1.0.0] - 2024-12-XX

### Added
- Phiên bản đầu tiên của Vintern-1B Image-to-Text Demo
- Giao diện web để chuyển đổi hình ảnh thành văn bản
- Hỗ trợ tải lên file bằng kéo thả
- Tùy chỉnh prompt và cấu hình API
- Chức năng sao chép vào clipboard
- Hỗ trợ streaming kết quả theo thời gian thực
- Docker support với nhiều kiến trúc (ARM64, AMD64, CUDA)
- Tự động phát hiện phần cứng và tối ưu hóa
- Script tự động cho cài đặt và chạy
- Hỗ trợ Apple Silicon (MPS)
- Volume mapping cho cache Hugging Face
- API endpoints cho xử lý hình ảnh
