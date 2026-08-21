"""
Image utilities for Virtual Try-On Evaluation
"""

import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns


def load_image(image_path, target_size=None, color_format='RGB'):
    """
    Load image từ file path
    
    Args:
        image_path: đường dẫn đến file ảnh
        target_size: tuple (width, height) để resize, None để giữ nguyên
        color_format: 'RGB' hoặc 'BGR'
    
    Returns:
        numpy array của ảnh
    """
    try:
        # Load using PIL first to handle various formats
        pil_image = Image.open(image_path)
        
        # Convert to RGB if needed
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array
        image = np.array(pil_image)
        
        # Resize if specified
        if target_size is not None:
            image = cv2.resize(image, target_size)
        
        # Convert color format if needed
        if color_format == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        return image
        
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def ensure_same_size(img1, img2, method='resize_smaller'):
    """
    Đảm bảo 2 ảnh có cùng kích thước
    
    Args:
        img1, img2: numpy arrays của ảnh
        method: 'resize_smaller', 'resize_larger', 'resize_first'
    
    Returns:
        tuple (img1_resized, img2_resized)
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    if h1 == h2 and w1 == w2:
        return img1, img2
    
    if method == 'resize_smaller':
        # Resize to the smaller dimensions
        target_h = min(h1, h2)
        target_w = min(w1, w2)
    elif method == 'resize_larger':
        # Resize to the larger dimensions
        target_h = max(h1, h2)
        target_w = max(w1, w2)
    elif method == 'resize_first':
        # Resize second image to match first
        target_h, target_w = h1, w1
    else:
        raise ValueError(f"Unknown resize method: {method}")
    
    img1_resized = cv2.resize(img1, (target_w, target_h))
    img2_resized = cv2.resize(img2, (target_w, target_h))
    
    return img1_resized, img2_resized


def get_image_pairs(original_dir, generated_dir):
    """
    Tìm và map các cặp ảnh gốc-sinh support cấu trúc nested
    
    Args:
        original_dir: thư mục chứa ảnh gốc
        generated_dir: thư mục chứa ảnh sinh
    
    Returns:
        list of tuples (original_path, generated_path)
    """
    pairs = []
    
    for root, _, files in os.walk(original_dir):
        # Skip 'masks' and 'garment' directories
        if os.path.basename(root) in ['masks', 'garment']:
            continue
            
        for original_file in files:
            if not original_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            original_path = os.path.join(root, original_file)
            base_name = os.path.splitext(original_file)[0]
            
            # Check for our typical generator prefixes/suffixes
            gen_names_to_try = [
                f"result_{base_name}.jpg",
                f"result_{base_name}.png",
                f"{base_name}_catvton_00001_.png",
                original_file
            ]
            
            # If the original file is inside a specific group (e.g., group_name/images)
            rel_path = os.path.relpath(original_path, original_dir)
            path_parts = rel_path.split(os.sep)
            
            found_match = False
            
            # 1. Try finding in matching group subdirectory if nested
            if len(path_parts) >= 3 and path_parts[-2] == 'images':
                group_name = path_parts[-3]
                
                # Setup potential mask path
                mask_dir = os.path.join(original_dir, group_name, 'masks')
                mask_base = os.path.splitext(original_file)[0] + ".png"; mask_path = os.path.join(mask_dir, f"mask_{mask_base}")
                if not os.path.exists(mask_path):
                    mask_path = None
                
                for gen_name in gen_names_to_try:
                    generated_path = os.path.join(generated_dir, group_name, gen_name)
                    if os.path.exists(generated_path):
                        pairs.append((original_path, generated_path, mask_path))
                        found_match = True
                        break
                        
            # 2. If not found in group, try directly in generated dir
            if not found_match:
                # Also try looking for mask in global masks folder
                mask_path = os.path.join(original_dir, 'masks', original_file.replace('.jpg', '.png'))
                if not os.path.exists(mask_path):
                    mask_path = None
                    
                for gen_name in gen_names_to_try:
                    generated_path = os.path.join(generated_dir, gen_name)
                    if os.path.exists(generated_path):
                        pairs.append((original_path, generated_path, mask_path))
                        found_match = True
                        break
                        
            if not found_match:
                print(f"Warning: Generated file not found for {original_file}")
                
    return pairs


def create_side_by_side_comparison(img1, img2, title1="Original", title2="Generated", 
                                 figsize=(12, 6), save_path=None):
    """
    Tạo ảnh so sánh side-by-side
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    axes[0].imshow(img1)
    axes[0].set_title(title1)
    axes[0].axis('off')
    
    axes[1].imshow(img2)
    axes[1].set_title(title2)
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_metrics_visualization(metrics_df, save_dir=None):
    """
    Tạo các biểu đồ visualization cho metrics
    """
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Distribution plots
    numeric_columns = metrics_df.select_dtypes(include=[np.number]).columns
    n_cols = min(3, len(numeric_columns))
    n_rows = (len(numeric_columns) + n_cols - 1) // n_cols
    
    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes1 = [axes1] if n_cols == 1 else axes1
    else:
        axes1 = axes1.flatten()
    
    for i, col in enumerate(numeric_columns):
        if i < len(axes1):
            sns.histplot(data=metrics_df, x=col, kde=True, ax=axes1[i])
            axes1[i].set_title(f'Distribution of {col}')
    
    # Hide unused subplots
    for i in range(len(numeric_columns), len(axes1)):
        axes1[i].set_visible(False)
    
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'metric_distributions.png'), 
                   dpi=300, bbox_inches='tight')
    
    # 2. Correlation heatmap
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    correlation_matrix = metrics_df[numeric_columns].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax2)
    ax2.set_title('Correlation Matrix of Metrics')
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'correlation_matrix.png'), 
                   dpi=300, bbox_inches='tight')
    
    return fig1, fig2


def create_summary_plots(metrics_df, save_dir=None):
    """
    Tạo summary plots cho overall performance
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: SSIM vs LPIPS scatter
    if 'ssim' in metrics_df.columns and 'lpips' in metrics_df.columns:
        axes[0,0].scatter(metrics_df['ssim'], metrics_df['lpips'], alpha=0.6)
        axes[0,0].set_xlabel('SSIM (higher better)')
        axes[0,0].set_ylabel('LPIPS (lower better)')
        axes[0,0].set_title('SSIM vs LPIPS')
        axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: PSNR distribution
    if 'psnr' in metrics_df.columns:
        axes[0,1].hist(metrics_df['psnr'], bins=20, alpha=0.7, edgecolor='black')
        axes[0,1].axvline(metrics_df['psnr'].mean(), color='red', linestyle='--', 
                         label=f'Mean: {metrics_df["psnr"].mean():.2f}')
        axes[0,1].set_xlabel('PSNR (dB)')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('PSNR Distribution')
        axes[0,1].legend()
    
    # Plot 3: Face similarity if available
    if 'face_similarity' in metrics_df.columns:
        valid_face_sim = metrics_df[metrics_df['face_similarity'] > 0]['face_similarity']
        if len(valid_face_sim) > 0:
            axes[1,0].hist(valid_face_sim, bins=15, alpha=0.7, edgecolor='black')
            axes[1,0].set_xlabel('Face Similarity')
            axes[1,0].set_ylabel('Frequency')
            axes[1,0].set_title('Face Similarity Distribution')
    
    # Plot 4: Overall quality ranking
    if 'ssim' in metrics_df.columns:
        # Create a composite score for ranking
        metrics_df_copy = metrics_df.copy()
        metrics_df_copy['composite_score'] = metrics_df_copy['ssim']
        if 'lpips' in metrics_df_copy.columns:
            # Invert LPIPS since lower is better
            metrics_df_copy['composite_score'] = (metrics_df_copy['ssim'] + (1 - metrics_df_copy['lpips'])) / 2
        
        sorted_df = metrics_df_copy.sort_values('composite_score', ascending=False)
        axes[1,1].plot(range(len(sorted_df)), sorted_df['composite_score'], 'o-')
        axes[1,1].set_xlabel('Image Rank')
        axes[1,1].set_ylabel('Composite Quality Score')
        axes[1,1].set_title('Quality Ranking')
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'summary_plots.png'), 
                   dpi=300, bbox_inches='tight')
    
    return fig


_DNN_FACE_NET = None

def _get_dnn_face_net():
    """Download and cache OpenCV SSD ResNet-10 DNN face detector model."""
    global _DNN_FACE_NET
    if _DNN_FACE_NET is not None:
        return _DNN_FACE_NET
        
    try:
        model_dir = os.path.join(os.path.expanduser("~"), ".cache", "opencv_dnn")
        os.makedirs(model_dir, exist_ok=True)
        
        prototxt_path = os.path.join(model_dir, "deploy.prototxt")
        caffemodel_path = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
        
        prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
        caffemodel_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
        
        import urllib.request
        if not os.path.exists(prototxt_path):
            print("Downloading OpenCV DNN face detector prototxt...")
            urllib.request.urlretrieve(prototxt_url, prototxt_path)
        if not os.path.exists(caffemodel_path):
            print("Downloading OpenCV DNN face detector weights (~5MB)...")
            urllib.request.urlretrieve(caffemodel_url, caffemodel_path)
            
        _DNN_FACE_NET = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
        return _DNN_FACE_NET
    except Exception as e:
        print(f"Warning: Could not load OpenCV DNN face detector ({e}). Falling back to Haar Cascade.")
        return None


def blur_faces(img, blur_factor=3.0, confidence_threshold=0.5):
    """
    Detect faces using OpenCV DNN (SSD ResNet-10) and blur them to anonymize.
    Falls back to Haar Cascade if DNN model unavailable.
    
    Args:
        img: numpy array (RGB format)
        blur_factor: higher value means more blur
        confidence_threshold: confidence threshold for DNN detection (0.0 - 1.0)
    """
    if img is None:
        return None
        
    result = img.copy()
    (h, w) = result.shape[:2]
    faces = []
    
    # Try OpenCV DNN Face Detector first
    net = _get_dnn_face_net()
    if net is not None:
        try:
            # OpenCV DNN expects BGR for mean subtraction
            bgr_img = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            blob = cv2.dnn.blobFromImage(cv2.resize(bgr_img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()
            
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > confidence_threshold:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    startX, startY = max(0, startX), max(0, startY)
                    endX, endY = min(w, endX), min(h, endY)
                    
                    if endX > startX and endY > startY:
                        faces.append((startX, startY, endX - startX, endY - startY))
        except Exception as e:
            print(f"DNN face detection error: {e}")
            faces = []
            
    # Fallback to Haar Cascade if no faces found with DNN or DNN failed
    if not faces:
        gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(detected) > 0:
            faces = [(x, y, w_box, h_box) for (x, y, w_box, h_box) in detected]
            
    # Apply Gaussian Blur to detected faces
    for (x, y, w_box, h_box) in faces:
        roi = result[y:y+h_box, x:x+w_box]
        
        kw = int(w_box / blur_factor) | 1
        kh = int(h_box / blur_factor) | 1
        
        blurred_roi = cv2.GaussianBlur(roi, (kw, kh), 0)
        result[y:y+h_box, x:x+w_box] = blurred_roi
        
    return result



def save_best_worst_samples(metrics_df, image_pairs, metric_name, output_dir, n_samples=5):
    """
    Lưu các sample ảnh tốt nhất và tệ nhất theo metric
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort by metric
    is_higher_better = metric_name in ['ssim', 'psnr', 'face_similarity']
    sorted_df = metrics_df.sort_values(metric_name, ascending=not is_higher_better)
    
    # Best samples
    best_dir = os.path.join(output_dir, f'best_{metric_name}')
    worst_dir = os.path.join(output_dir, f'worst_{metric_name}')
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(worst_dir, exist_ok=True)
    
    # Save best samples
    for i in range(min(n_samples, len(sorted_df))):
        row = sorted_df.iloc[i]
        
        # Determine paths (handle both old list-based and new df-based formats)
        if 'original_path' in row and 'generated_path' in row:
            original_path = row['original_path']
            generated_path = row['generated_path']
        else:
            idx = sorted_df.index[i]
            pair = image_pairs[idx]
            original_path = pair[0]
            generated_path = pair[1]
        
        # Load images
        orig_img = load_image(original_path)
        gen_img = load_image(generated_path)
        
        if orig_img is not None and gen_img is not None:
            orig_img, gen_img = ensure_same_size(orig_img, gen_img)
            
            # Blur faces before saving visualization
            orig_img = blur_faces(orig_img)
            gen_img = blur_faces(gen_img)
            
            # Create comparison
            fig = create_side_by_side_comparison(
                orig_img, gen_img,
                title1=f"Original",
                title2=f"Generated ({metric_name}: {row[metric_name]:.3f})"
            )
            
            base_name = os.path.basename(original_path).split('.')[0]
            save_path = os.path.join(best_dir, f'{base_name}_comparison.png')
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
    
    # Save worst samples
    for i in range(min(n_samples, len(sorted_df))):
        row = sorted_df.iloc[-(i+1)]
        
        if 'original_path' in row and 'generated_path' in row:
            original_path = row['original_path']
            generated_path = row['generated_path']
        else:
            idx = sorted_df.index[-(i+1)]
            pair = image_pairs[idx]
            original_path = pair[0]
            generated_path = pair[1]
        
        # Load images
        orig_img = load_image(original_path)
        gen_img = load_image(generated_path)
        
        if orig_img is not None and gen_img is not None:
            orig_img, gen_img = ensure_same_size(orig_img, gen_img)
            
            # Blur faces before saving visualization
            orig_img = blur_faces(orig_img)
            gen_img = blur_faces(gen_img)
            
            # Create comparison
            fig = create_side_by_side_comparison(
                orig_img, gen_img,
                title1=f"Original",
                title2=f"Generated ({metric_name}: {row[metric_name]:.3f})"
            )
            
            base_name = os.path.basename(original_path).split('.')[0]
            save_path = os.path.join(worst_dir, f'{base_name}_comparison.png')
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
