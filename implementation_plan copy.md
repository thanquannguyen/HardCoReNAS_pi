# Kế hoạch Triển khai HardCoReNAS

## Mục tiêu
Nghiên cứu, chạy thực nghiệm HardCoReNAS, áp dụng nén đa tầng (Pruning + Quantization + Low-rank) và đo lường hiệu năng trên Raspberry Pi (Docker).

## User Review Required
> [!IMPORTANT]
> **Docker & Raspberry Pi**: Tôi sẽ thiết lập Dockerfile để có thể build image chạy được trên Raspberry Pi (ARM64). Bạn sẽ cần cài Docker trên Pi để chạy image này.
> **Dataset**: Vẫn sử dụng CIFAR-10 làm mẫu chuẩn.
> **Pretrained**: Sẽ sử dụng checkpoint Supernet có sẵn để tiết kiệm thời gian train (vài ngày), tập trung vào phần Search và Compression.

## Proposed Changes

### Giai đoạn 1: NAS & Docker Setup
#### [NEW] `Docker/Dockerfile.pi`
Dockerfile tối ưu cho Raspberry Pi (base image python:3.9-slim).

#### [NEW] `scripts/setup_env.bat`
Script cài đặt môi trường python trên Windows để chạy thử nghiệm.

#### [NEW] `scripts/run_nas_search.py`
Script thực hiện search kiến trúc dựa trên Latency constraint.

### Giai đoạn 2: Advanced Compression Pipeline
#### [NEW] `compression/compress_pipeline.py`
Script chính điều phối quy trình nén:
1. **Knowledge Distillation**: Đã có trong HardCoReNAS train loop.
2. **Pruning**: Sử dụng `torch.nn.utils.prune` để tỉa thưa model sau khi search.
3. **Low-Rank Factorization**: Phân rã các lớp Conv2d/Linear lớn thành các ma trận nhỏ hơn (SVD).
4. **Quantization**: Chuyển đổi sang ONNX và Quantize INT8 (dùng ONNX Runtime).

#### [NEW] `compression/low_rank.py`
Module thực hiện Low-rank decomposition.

#### [NEW] `compression/quantize_onnx.py`
Module thực hiện convert PyTorch -> ONNX -> Quantized ONNX.

### Giai đoạn 3: Benchmarking
#### [NEW] `benchmark/evaluate.py`
Script đo lường:
- Accuracy (trên validation set).
- Latency (trung bình qua N lần chạy).
- Model Size (kích thước file on disk).

## Verification Plan
### Automated Tests
- Chạy thử pipeline với `max_epochs=1` hoặc dataset nhỏ (CIFAR-10 subset) để đảm bảo code chạy không lỗi.
- Kiểm tra kích thước model sau khi nén có giảm không.
- Kiểm tra model INT8 có chạy được inference không.

### Manual Verification
- Người dùng chạy các script và xác nhận kết quả in ra màn hình.
