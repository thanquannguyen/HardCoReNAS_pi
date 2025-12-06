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
- [ ] Chạy NAS (HardCoReNAS) <!-- id: 6 -->
    - [ ] Tạo Latency LUT (Lookup Table) giả lập hoặc đo trên Pi (nếu có thể remote) <!-- id: 7 -->
    - [ ] Train Supernet (Sử dụng checkpoint có sẵn để tiết kiệm thời gian) <!-- id: 8 -->
    - [ ] Thực hiện Search với ràng buộc latency <!-- id: 11 -->
- [ ] Huấn luyện Baseline <!-- id: 12 -->
    - [ ] Fine-tune (retrain) kiến trúc tìm được từ scratch <!-- id: 13 -->

## Giai đoạn 2: Quy trình Nén Mô hình Đa tầng (Advanced)
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
- [ ] Chạy NAS (HardCoReNAS) <!-- id: 6 -->
    - [ ] Tạo Latency LUT (Lookup Table) giả lập hoặc đo trên Pi (nếu có thể remote) <!-- id: 7 -->
    - [ ] Train Supernet (Sử dụng checkpoint có sẵn để tiết kiệm thời gian) <!-- id: 8 -->
    - [ ] Thực hiện Search với ràng buộc latency <!-- id: 11 -->
- [ ] Huấn luyện Baseline <!-- id: 12 -->
    - [ ] Fine-tune (retrain) kiến trúc tìm được từ scratch <!-- id: 13 -->

## Giai đoạn 2: Quy trình Nén Mô hình Đa tầng (Advanced)
- [ ] Chiến lược 1: Knowledge Distillation (Tích hợp sẵn trong quá trình train/finetune) <!-- id: 14 -->
- [ ] Chiến lược 2: Pruning (Tỉa thưa) <!-- id: 15 -->
    - [ ] Áp dụng Structured Pruning (Filter pruning) <!-- id: 16 -->
- [ ] Chiến lược 3: Low-Rank Factorization <!-- id: 27 -->
    - [ ] Phân rã các lớp Conv/Linear bằng SVD <!-- id: 28 -->
- [ ] Chiến lược 4: Quantization (Lượng tử hóa) <!-- id: 17 -->
    - [x] Chuyển đổi mô hình sang ONNX <!-- id: 29 -->
    - [x] Thực hiện Quantization (INT8) dùng ONNX Runtime <!-- id: 18 -->
    - [ ] Debug Accuracy Drop (Paused: Base model accuracy 45% is too low. Retraining first.)

- [x] **Step 4: Advanced Compression Pipeline (Fixing the Flow)**
    - [x] Implement `fine_tune` function in `compress_pipeline.py` (Healing)
    - [x] Integrate Knowledge Distillation (KD) using original model as Teacher
    - [x] Update flow: Prune -> Fine-tune -> Low-Rank -> Fine-tune -> Quantize
    - [x] Update `run_nas_pipeline.py` with optimized training params (Batch 128, AMP, 50 Epochs)

- [ ] **Step 5: Retraining (Improve Baseline)**
    - [ ] Push code to GitHub & Pull on Kaggle
    - [ ] Train 50 epochs trên Kaggle (Mục tiêu > 80% accuracy)
    - [ ] Verify accuracy của `compressed_model.pth` mới
    - [ ] Re-run Quantization trên model chính xác cao
## Giai đoạn 3: Đo hiệu xuất và Phân tích trên Edge
- [ ] Deployment <!-- id: 30 -->
    - [ ] Đóng gói mô hình tối ưu vào Docker container cho Raspberry Pi <!-- id: 31 -->
- [ ] Benchmarking <!-- id: 20 -->
    - [ ] Đo độ chính xác (Accuracy) <!-- id: 21 -->
    - [ ] Đo độ trễ (Latency) thực tế trên Pi (hoặc giả lập) <!-- id: 22 -->
    - [ ] Đo kích thước mô hình (Model Size) <!-- id: 23 -->
- [ ] Tổng hợp báo cáo <!-- id: 24 -->
    - [ ] So sánh kết quả trước và sau khi nén <!-- id: 25 -->
    - [ ] Tạo script demo <!-- id: 26 -->
