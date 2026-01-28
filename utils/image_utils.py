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
    Tìm và map các cặp ảnh gốc-sinh
    
    Args:
        original_dir: thư mục chứa ảnh gốc
        generated_dir: thư mục chứa ảnh sinh
    
    Returns:
        list of tuples (original_path, generated_path)
    """
    pairs = []
    
    # Get all original images
    original_files = [f for f in os.listdir(original_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for original_file in original_files:
        # Extract base name (e.g., TNTMEDIA-1 from TNTMEDIA-1.jpg)
        base_name = os.path.splitext(original_file)[0]
        
        # Construct expected generated filename
        generated_file = f"{base_name}_catvton_00001_.png"
        
        original_path = os.path.join(original_dir, original_file)
        generated_path = os.path.join(generated_dir, generated_file)
        
        # Check if generated file exists
        if os.path.exists(generated_path):
            pairs.append((original_path, generated_path))
        else:
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
        idx = sorted_df.index[i]
        original_path, generated_path = image_pairs[idx]
        
        # Load images
        orig_img = load_image(original_path)
        gen_img = load_image(generated_path)
        
        if orig_img is not None and gen_img is not None:
            orig_img, gen_img = ensure_same_size(orig_img, gen_img)
            
            # Create comparison
            fig = create_side_by_side_comparison(
                orig_img, gen_img,
                title1=f"Original",
                title2=f"Generated ({metric_name}: {sorted_df.iloc[i][metric_name]:.3f})"
            )
            
            base_name = os.path.basename(original_path).split('.')[0]
            save_path = os.path.join(best_dir, f'{base_name}_comparison.png')
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
    
    # Save worst samples
    for i in range(min(n_samples, len(sorted_df))):
        idx = sorted_df.index[-(i+1)]
        original_path, generated_path = image_pairs[idx]
        
        # Load images
        orig_img = load_image(original_path)
        gen_img = load_image(generated_path)
        
        if orig_img is not None and gen_img is not None:
            orig_img, gen_img = ensure_same_size(orig_img, gen_img)
            
            # Create comparison
            fig = create_side_by_side_comparison(
                orig_img, gen_img,
                title1=f"Original",
                title2=f"Generated ({metric_name}: {sorted_df.iloc[-(i+1)][metric_name]:.3f})"
            )
            
            base_name = os.path.basename(original_path).split('.')[0]
            save_path = os.path.join(worst_dir, f'{base_name}_comparison.png')
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
