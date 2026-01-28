# Virtual Try-On Evaluation Tool

Tool đánh giá chất lượng ảnh Virtual Try-On bằng các metrics học thuật tiêu chuẩn.

## Tính năng

### Metrics được hỗ trợ:

**Image Quality Metrics:**
- **SSIM** (Structural Similarity Index): Đo độ tương đồng cấu trúc (0-1, cao hơn tốt hơn)
- **PSNR** (Peak Signal-to-Noise Ratio): Đo tỷ lệ tín hiệu/nhiễu (dB, cao hơn tốt hơn)
- **LPIPS** (Learned Perceptual Image Patch Similarity): Đo độ tương đồng perceptual (0-1, thấp hơn tốt hơn)
- **MSE** (Mean Squared Error): Sai số bình phương trung bình (thấp hơn tốt hơn)

**Perceptual Metrics:**
- **FID** (Fréchet Inception Distance): Đo khoảng cách phân phối (thấp hơn tốt hơn)

**Virtual Try-On Specific:**
- **Face Similarity**: Độ tương đồng khuôn mặt để đánh giá bảo toàn identity
- **Color Distance**: Khoảng cách histogram màu sắc
- **Edge Consistency**: Độ nhất quán của edges

## Cài đặt

1. Clone hoặc tải project
2. Cài đặt dependencies:
```bash
cd virtual_tryon_evaluator
pip install -r requirements.txt
```

3. (Optional) Cài đặt pytorch-fid cho FID metric:
```bash
pip install pytorch-fid
```

## Cấu trúc thư mục

```
virtual_tryon_evaluator/
├── config.yaml              # File cấu hình
├── requirements.txt          # Dependencies
├── evaluate.py              # Script chính
├── README.md               # Hướng dẫn này
├── metrics/                # Module tính metrics
│   ├── __init__.py
│   ├── image_quality.py
│   └── face_similarity.py
└── utils/                  # Utilities
    ├── __init__.py
    └── image_utils.py
```

## Sử dụng

### 1. Cấu hình

Chỉnh sửa file `config.yaml` với đường dẫn của bạn:

```yaml
paths:
  original_dir: "/đường/dẫn/đến/ảnh/gốc"
  generated_dir: "/đường/dẫn/đến/ảnh/sinh"
  output_dir: "./evaluation_results"
```

### 2. Chạy evaluation

**Cách 1: Sử dụng config file**
```bash
python evaluate.py
```

**Cách 2: Override paths qua command line**
```bash
python evaluate.py \
  --original_dir "/home/ldkien/Downloads/anh/12A_THPT_HATRUNG" \
  --generated_dir "/home/ldkien/Downloads/anh/CatVtion_12A_THPT_HATRUNG" \
  --output_dir "./results"
```

### 3. Xem kết quả

Sau khi chạy xong, trong thư mục output sẽ có:

```
evaluation_results/
├── detailed_metrics.csv           # Metrics chi tiết cho từng ảnh
├── summary_statistics.json        # Thống kê tổng quan  
├── evaluation_report.html         # Báo cáo HTML
├── metric_distributions.png       # Biểu đồ phân phối metrics
├── correlation_matrix.png         # Ma trận correlation
├── summary_plots.png             # Biểu đồ tổng quan
└── visual_samples/               # Ví dụ ảnh tốt nhất/tệ nhất
    ├── best_ssim/
    ├── worst_ssim/
    ├── best_lpips/
    └── worst_lpips/
```

## Mapping ảnh

Tool tự động map ảnh gốc và ảnh sinh theo pattern:
- Ảnh gốc: `TNTMEDIA-{ID}.jpg`
- Ảnh sinh: `TNTMEDIA-{ID}_catvton_00001_.png`

Ví dụ:
- `TNTMEDIA-1.jpg` → `TNTMEDIA-1_catvton_00001_.png`
- `TNTMEDIA-30.jpg` → `TNTMEDIA-30_catvton_00001_.png`

## Giải thích Metrics

### SSIM (Structural Similarity Index)
- **Range**: 0-1
- **Tốt**: > 0.8 (excellent), 0.6-0.8 (good), < 0.6 (poor)
- **Ý nghĩa**: Đo độ tương đồng cấu trúc về luminance, contrast, structure

### PSNR (Peak Signal-to-Noise Ratio)
- **Range**: 0-∞ dB
- **Tốt**: > 30 dB (good), > 40 dB (excellent)
- **Ý nghĩa**: Đo chất lượng reconstruction, cao hơn = ít noise hơn

### LPIPS (Learned Perceptual Image Patch Similarity)
- **Range**: 0-1
- **Tốt**: < 0.2 (excellent), 0.2-0.5 (moderate), > 0.5 (poor)
- **Ý nghĩa**: Đo độ tương đồng theo perception của con người

### FID (Fréchet Inception Distance)
- **Range**: 0-∞
- **Tốt**: < 30 (excellent), 30-100 (moderate), > 100 (poor)
- **Ý nghĩa**: Đo khoảng cách giữa phân phối ảnh real và generated

### Face Similarity
- **Range**: 0-1
- **Tốt**: > 0.8 (excellent), 0.6-0.8 (good), < 0.6 (poor)
- **Ý nghĩa**: Đo độ bảo toàn identity của khuôn mặt

## Cấu hình nâng cao

### Tùy chỉnh metrics

Trong `config.yaml`, bạn có thể bật/tắt các metrics:

```yaml
evaluation:
  metrics:
    enable_ssim: true
    enable_psnr: true
    enable_lpips: true
    enable_fid: true
    enable_face_similarity: true
    enable_mse: true
    enable_color_distance: true
    enable_edge_consistency: true
```

### Tùy chỉnh output

```yaml
evaluation:
  output:
    save_detailed_csv: true
    save_summary_json: true
    save_visual_samples: true
    num_visual_samples: 10
    generate_plots: true
    create_html_report: true
```

### GPU/CPU

```yaml
evaluation:
  processing:
    device: "cuda"  # "cuda" hoặc "cpu"
    batch_size: 8
```

## Troubleshooting

### Lỗi thường gặp:

1. **"No module named 'face_recognition'"**
   ```bash
   pip install face-recognition
   ```

2. **"CUDA out of memory"**
   - Đổi device sang "cpu" trong config
   - Giảm batch_size

3. **"FID metric not available"**
   ```bash
   pip install pytorch-fid
   ```

4. **"No faces detected"**
   - Normal nếu một số ảnh không có mặt rõ
   - Face similarity sẽ trả về 0.0 cho những ảnh này

### Performance

- **Thời gian ước tính**: ~5-10 giây/cặp ảnh
- **RAM**: 8GB+ recommended cho large images
- **GPU**: Optional nhưng tăng tốc LPIPS và FID

## Ví dụ kết quả

```
Key Metrics:
  SSIM (avg): 0.782 ± 0.134
  LPIPS (avg): 0.245 ± 0.089
  PSNR (avg): 23.4 ± 4.2 dB
  FID Score: 45.67
  Face Detection Rate: 87.5%
```

## Liên hệ

Nếu có vấn đề hoặc cần hỗ trợ, hãy tạo issue hoặc liên hệ.
