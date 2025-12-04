# Hướng dẫn Triển khai HardCoReNAS lên Raspberry Pi

Tài liệu này hướng dẫn chi tiết cách đưa mô hình đã tối ưu (Quantized ONNX) lên Raspberry Pi và chạy đánh giá hiệu năng sử dụng Docker.

## 1. Chuẩn bị trên Raspberry Pi

Đảm bảo Raspberry Pi của bạn đã cài đặt:
- **Hệ điều hành**: Raspberry Pi OS (64-bit khuyến nghị) hoặc Ubuntu Server 20.04/22.04 (ARM64).
- **Docker**: Đã cài đặt và user có quyền chạy docker (không cần sudo).

### Cài đặt Docker (nếu chưa có):
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
sudo usermod -aG docker $USER
# Logout và Login lại để áp dụng quyền
```

## 2. Chuyển dữ liệu sang Raspberry Pi

Bạn cần copy thư mục project `HardCoReNAS` từ máy tính sang Pi. Có thể dùng `scp` hoặc `git clone` (nếu bạn đã push lên GitHub).

**Cách 1: Dùng Git (Khuyên dùng)**
```bash
# Trên Raspberry Pi
git clone <URL_REPO_CUA_BAN> HardCoReNAS
cd HardCoReNAS
# Nếu dùng LFS, nhớ cài git-lfs trên Pi và chạy: git lfs pull
```

**Cách 2: Dùng SCP (Copy trực tiếp)**
```bash
# Trên máy tính Windows (PowerShell)
scp -r C:\Data\HardCoReNAS pi@<IP_ADDRESS_CUA_PI>:~/HardCoReNAS
```

## 3. Build Docker Image trên Pi

Do kiến trúc CPU khác nhau (x86 vs ARM64), bạn cần build image trực tiếp trên Pi.

```bash
cd ~/HardCoReNAS
docker build -f Docker/Dockerfile.pi -t hardcorenas-pi .
```
*Lưu ý: Quá trình này có thể mất 15-30 phút tùy tốc độ mạng và thẻ nhớ SD.*

## 4. Chạy Demo và Đánh giá (Benchmark)

Sau khi build xong, bạn có thể chạy container để kiểm tra model.

### Chạy Container
```bash
docker run -it --rm -v $(pwd):/workspace hardcorenas-pi
```

### Trong Container:
1. **Kiểm tra thông tin Model:**
   ```bash
   ls -lh model_quantized.onnx
   # Bạn sẽ thấy kích thước file khoảng 4MB (so với 16MB gốc)
   ```

2. **Chạy Benchmark (Đo Latency & FPS):**
   ```bash
   python benchmark/evaluate.py --model_path model_quantized.onnx
   ```
   
   Kết quả mong đợi:
   - **Model**: ONNX
   - **Size**: ~4.3 MB
   - **Latency**: xx.xx ms (Thấp hơn nhiều so với chạy trên CPU thường do tối ưu hóa)

3. **Chạy Inference Test (Kiểm tra hoạt động):**
   ```bash
   python test_inference.py
   ```

## 5. (Tùy chọn) Chạy lại Pipeline trên Pi
Nếu bạn muốn thử sức mạnh của Pi, bạn có thể chạy lại quy trình nén ngay trên thiết bị (tuy nhiên sẽ chậm hơn PC):

```bash
python scripts/run_nas_pipeline.py
```

## Tổng kết
Bạn đã hoàn thành việc:
1. Tìm kiếm kiến trúc (NAS).
2. Nén mô hình (Pruning -> SVD -> Quantization).
3. Đóng gói và chạy thực tế trên thiết bị biên (Raspberry Pi).
