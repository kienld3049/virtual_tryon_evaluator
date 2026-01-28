# Virtual Try-On Evaluation Tool - Tổng Kết

## 🎯 Mục Tiêu
Đánh giá chất lượng ảnh Virtual Try-On bằng các độ đo học thuật tiêu chuẩn được sử dụng trong các bài báo nghiên cứu.

## 📊 Các Độ Đo Đã Implement

### 1. **Image Quality Metrics**
- **SSIM** (Structural Similarity Index): Đo độ tương tự về cấu trúc (0-1, cao hơn = tốt hơn)
- **PSNR** (Peak Signal-to-Noise Ratio): Đo tỷ lệ tín hiệu/nhiễu (dB, cao hơn = tốt hơn)
- **MSE** (Mean Squared Error): Đo sai số bình phương trung bình (thấp hơn = tốt hơn)
- **LPIPS** (Learned Perceptual Image Patch Similarity): Đo độ tương tự perceptual (0-1, thấp hơn = tốt hơn)

### 2. **Color & Structure Metrics**
- **Color Histogram Distance**: Đo khoảng cách histogram màu sắc (thấp hơn = tốt hơn)
- **Edge Consistency**: Đo độ nhất quán về cạnh/biên (0-1, cao hơn = tốt hơn)

### 3. **Distribution-Level Metrics**
- **FID** (Fréchet Inception Distance): Đo khoảng cách phân phối features (thấp hơn = tốt hơn)

## 🧪 Kết Quả Test
```
Testing with: TNTMEDIA-2552.jpg vs TNTMEDIA-2552_catvton_00001_.png
✅ LPIPS Score: 0.3785 (Moderate perceptual similarity)
✅ SSIM Score: 0.8546 (Excellent structural similarity)
✅ PSNR Score: 15.73 dB (Acceptable quality)
```

## 📈 Benchmark Thông Dụng

### LPIPS Thresholds:
- **Excellent**: < 0.2
- **Good**: 0.2 - 0.3
- **Moderate**: 0.3 - 0.5
- **Poor**: > 0.5

### SSIM Thresholds:
- **Excellent**: > 0.8
- **Good**: 0.6 - 0.8
- **Poor**: < 0.6

### FID Thresholds:
- **Excellent**: < 30
- **Good**: 30 - 100
- **Poor**: > 100

## 🛠️ Cách Sử Dụng

### Quick Test:
```bash
cd virtual_tryon_evaluator
python test_lpips.py
```

### Full Evaluation:
```bash
cd virtual_tryon_evaluator
python evaluate.py
```

### Custom Paths:
```bash
python evaluate.py --original_dir /path/to/original --generated_dir /path/to/generated
```

## 📁 Cấu Trúc Output

Sau khi chạy evaluation, bạn sẽ có:

```
evaluation_results/
├── detailed_metrics.csv          # Chi tiết metrics cho từng ảnh
├── summary_statistics.json       # Thống kê tổng hợp
├── evaluation_report.html        # Báo cáo HTML
├── metric_distributions.png      # Phân phối các metrics
├── correlation_matrix.png        # Ma trận tương quan
├── summary_plots.png            # Plots tổng hợp
└── visual_samples/              # Best/worst examples
    ├── ssim_best/
    ├── ssim_worst/
    ├── lpips_best/
    └── lpips_worst/
```

## 🔧 Configuration

File `config.yaml` cho phép tùy chỉnh:
- Đường dẫn input/output
- Enable/disable metrics
- Device (CPU/CUDA)
- Batch size
- Visualization settings

## 📚 Tham Khảo Học Thuật

Các metrics được implement dựa trên:

1. **SSIM**: Wang et al. "Image quality assessment: from error visibility to structural similarity" (2004)
2. **LPIPS**: Zhang et al. "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric" (2018)
3. **FID**: Heusel et al. "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium" (2017)

## ✅ Tính Năng Đã Hoàn Thành

- [x] Implement tất cả metrics chính
- [x] Support cả CPU và CUDA
- [x] Tự động resize/align images
- [x] Robust error handling
- [x] Comprehensive reporting
- [x] Visualization plots
- [x] HTML report generation
- [x] Best/worst samples extraction
- [x] Configurable via YAML
- [x] Command line interface

## 🎯 Đánh Giá Kết Quả Test

Dựa trên kết quả test ban đầu:
- **SSIM = 0.8546**: Chất lượng cấu trúc rất tốt
- **LPIPS = 0.3785**: Độ tương tự perceptual ở mức trung bình
- **PSNR = 15.73**: Chất lượng tín hiệu chấp nhận được

➡️ **Kết luận**: Model CatVTon có hiệu suất khá tốt về việc giữ nguyên cấu trúc ảnh gốc, nhưng vẫn có thể cải thiện về độ tự nhiên perceptual.

## 🚀 Hướng Dẫn Tiếp Theo

1. Chạy full evaluation trên toàn bộ dataset
2. Phân tích báo cáo HTML được tạo ra
3. Xem visual samples để hiểu rõ strengths/weaknesses
4. Sử dụng kết quả để fine-tune model parameters
