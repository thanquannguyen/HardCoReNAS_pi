# HardCoReNAS Research & Implementation

## Giai đoạn 1: Nghiên cứu và Ứng dụng NAS
- [x] Nghiên cứu bài báo và code <!-- id: 0 -->
    - [x] Đọc README và cấu trúc code <!-- id: 1 -->
    - [x] Tóm tắt Search Space và Search Strategy <!-- id: 2 -->
- [x] Chuẩn bị môi trường <!-- id: 3 -->
    - [x] Kiểm tra và cập nhật Dockerfile (hỗ trợ buildx cho ARM/Raspberry Pi nếu cần) <!-- id: 4 -->
    - [x] Chuẩn bị dataset (CIFAR-10) <!-- id: 5 -->
        - [x] Tải dữ liệu raw (binaries) <!-- id: 32 -->
        - [x] Chuyển đổi sang định dạng ImageFolder (train/val) <!-- id: 33 -->
- [x] Chạy NAS (HardCoReNAS) <!-- id: 6 -->
    - [x] Tạo Latency LUT (Lookup Table) giả lập hoặc đo trên Pi (nếu có thể remote) <!-- id: 7 -->
    - [x] Train Supernet (Link gốc lỗi 403 -> Dùng Dummy hoặc Train from scratch) <!-- id: 8 -->
    - [x] Thực hiện Search với ràng buộc latency (Demo với Dummy Checkpoint) <!-- id: 11 -->
- [x] Huấn luyện Baseline <!-- id: 12 -->
    - [x] Fine-tune (retrain) kiến trúc tìm được từ scratch (Demo với Dummy Checkpoint) <!-- id: 13 -->

## Giai đoạn 2: Quy trình Nén Mô hình Đa tầng (Advanced)
- [x] Chiến lược 1: Knowledge Distillation (Tích hợp sẵn trong quá trình train/finetune) <!-- id: 14 -->
- [x] Chiến lược 2: Pruning (Tỉa thưa) <!-- id: 15 -->
    - [x] Áp dụng Structured Pruning (Filter pruning) <!-- id: 16 -->
- [x] Chiến lược 3: Low-Rank Factorization <!-- id: 27 -->
    - [x] Phân rã các lớp Conv/Linear bằng SVD <!-- id: 28 -->
- [x] Chiến lược 4: Quantization (Lượng tử hóa) <!-- id: 17 -->
    - [x] Chuyển đổi mô hình sang ONNX <!-- id: 29 -->
    - [x] Thực hiện Quantization (INT8) dùng ONNX Runtime hoặc TFLite <!-- id: 18 -->

## Giai đoạn 3: Đo hiệu xuất và Phân tích trên Edge
- [x] Deployment <!-- id: 30 -->
    - [x] Đóng gói mô hình tối ưu vào Docker container cho Raspberry Pi <!-- id: 31 -->
- [x] Benchmarking <!-- id: 20 -->
    - [x] Đo độ chính xác (Accuracy) <!-- id: 21 -->
    - [x] Đo độ trễ (Latency) thực tế trên Pi (hoặc giả lập) <!-- id: 22 -->
    - [x] Đo kích thước mô hình (Model Size) <!-- id: 23 -->
- [x] Tổng hợp báo cáo <!-- id: 24 -->
    - [ ] So sánh kết quả trước và sau khi nén <!-- id: 25 -->
    - [ ] Tạo script demo <!-- id: 26 -->
