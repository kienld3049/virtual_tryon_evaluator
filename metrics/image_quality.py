"""
Image Quality Metrics for Virtual Try-On Evaluation
"""

import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
# Try to import torch and lpips, but handle gracefully if not available
try:
    import torch
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: torch and lpips not available. LPIPS metric will be disabled.")


def chi_square_distance(hist1, hist2):
    """
    Custom implementation of chi-square distance
    Returns: float (lower is better)
    """
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    return 0.5 * np.sum(((hist1 - hist2) ** 2) / (hist1 + hist2 + epsilon))


class ImageQualityMetrics:
    """Tính toán các độ đo chất lượng ảnh"""
    
    def __init__(self, device='cpu', lpips_net='alex'):
        # Force CPU if CUDA not available
        if device == 'cuda' and not (LPIPS_AVAILABLE and torch.cuda.is_available()):
            device = 'cpu'
            print(f"Warning: CUDA requested but not available. Using CPU instead.")
        
        self.device = device
        if LPIPS_AVAILABLE:
            self.lpips_model = lpips.LPIPS(net=lpips_net).to(device)
        else:
            self.lpips_model = None
        
    def calculate_ssim(self, img1, img2, multichannel=True):
        """
        Tính Structural Similarity Index (SSIM)
        Returns: float (0-1, higher is better)
        """
        try:
            if len(img1.shape) == 3 and multichannel:
                # For color images - Removed multichannel=True to fix API crash in scikit-image >= 0.19
                score = ssim(img1, img2, channel_axis=-1, data_range=255)
            else:
                # For grayscale images
                score = ssim(img1, img2, data_range=255)
            return float(score)
        except Exception as e:
            print("Error calculating SSIM: {}".format(e))
            return 0.0
    
    def calculate_psnr(self, img1, img2):
        """
        Tính Peak Signal-to-Noise Ratio (PSNR)
        Returns: float (dB, higher is better)
        """
        try:
            score = psnr(img1, img2, data_range=255)
            return float(score)
        except Exception as e:
            print("Error calculating PSNR: {}".format(e))
            return 0.0
    
    def calculate_mse(self, img1, img2):
        """
        Tính Mean Squared Error (MSE)
        Returns: float (lower is better)
        """
        try:
            score = mse(img1, img2)
            return float(score)
        except Exception as e:
            print("Error calculating MSE: {}".format(e))
            return float('inf')
    
    def calculate_lpips(self, img1, img2):
        """
        Tính Learned Perceptual Image Patch Similarity (LPIPS)
        Returns: float (0-1, lower is better)
        """
        if not LPIPS_AVAILABLE or self.lpips_model is None:
            print("LPIPS not available")
            return 1.0
            
        try:
            # Convert to torch tensors and normalize to [-1, 1]
            img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).float() / 127.5 - 1.0
            img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).float() / 127.5 - 1.0
            
            # Add batch dimension
            img1_tensor = img1_tensor.unsqueeze(0).to(self.device)
            img2_tensor = img2_tensor.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                score = self.lpips_model(img1_tensor, img2_tensor)
            
            return float(score.cpu().item())
        except Exception as e:
            print("Error calculating LPIPS: {}".format(e))
            return 1.0
    
    def calculate_color_histogram_distance(self, img1, img2, bins=256):
        """
        Tính khoảng cách histogram màu sắc
        Returns: float (lower is better)
        """
        try:
            # Tính histogram cho mỗi channel
            distance = 0
            for channel in range(img1.shape[2]):
                hist1 = cv2.calcHist([img1], [channel], None, [bins], [0, 256])
                hist2 = cv2.calcHist([img2], [channel], None, [bins], [0, 256])
                
                # Normalize histograms
                hist1 = hist1.flatten() / hist1.sum()
                hist2 = hist2.flatten() / hist2.sum()
                
                # Chi-square distance
                distance += chi_square_distance(hist1, hist2)
            
            return float(distance / img1.shape[2])
        except Exception as e:
            print("Error calculating color histogram distance: {}".format(e))
            return float('inf')
    
    def calculate_edge_consistency(self, img1, img2):
        """
        Tính độ nhất quán của edges giữa 2 ảnh
        Returns: float (0-1, higher is better)
        """
        try:
            # Convert to grayscale
            gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
            
            # Detect edges using Sobel
            sobelx1 = cv2.Sobel(gray1, cv2.CV_64F, 1, 0, ksize=3)
            sobely1 = cv2.Sobel(gray1, cv2.CV_64F, 0, 1, ksize=3)
            edge1 = np.sqrt(sobelx1**2 + sobely1**2)
            
            sobelx2 = cv2.Sobel(gray2, cv2.CV_64F, 1, 0, ksize=3)
            sobely2 = cv2.Sobel(gray2, cv2.CV_64F, 0, 1, ksize=3)
            edge2 = np.sqrt(sobelx2**2 + sobely2**2)
            
            # Calculate SSIM of edge maps
            edge_ssim = ssim(edge1, edge2, data_range=max(float(edge1.max()), 1e-6))
            return float(edge_ssim)
        except Exception as e:
            print("Error calculating edge consistency: {}".format(e))
            return 0.0


class FIDMetric:
    """Fréchet Inception Distance metric"""
    
    def __init__(self, device='cpu'):
        self.device = device
        # Import here to avoid issues if pytorch-fid is not installed
        try:
            from pytorch_fid import fid_score
            self.fid_score = fid_score
        except ImportError:
            print("Warning: pytorch-fid not installed. FID metric will not be available.")
            self.fid_score = None
    
    def calculate_fid_from_paths(self, path1, path2, batch_size=50, dims=2048, image_pairs=None):
        """
        Tính FID score giữa 2 thư mục ảnh
        Returns: float (lower is better)
        """
        if self.fid_score is None:
            print("FID metric not available")
            return float('inf')
            
        import tempfile
        import shutil
        import os
        
        try:
            if image_pairs:
                # Create temp lists of symbolic links if we have nested dirs to handle pytorch-fid
                temp_dir_real = tempfile.mkdtemp()
                temp_dir_fake = tempfile.mkdtemp()
                
                try:
                    for idx, pair in enumerate(image_pairs):
                        if len(pair) >= 2:
                            real_path = pair[0]
                            fake_path = pair[1]
                        else:
                            continue
                        ext_real = os.path.splitext(real_path)[1]
                        ext_fake = os.path.splitext(fake_path)[1]
                        # Use copy instead of symlink to avoid filesystem restrictions (e.g. exFAT/NTFS)
                        shutil.copy2(real_path, os.path.join(temp_dir_real, f"{idx}{ext_real}"))
                        shutil.copy2(fake_path, os.path.join(temp_dir_fake, f"{idx}{ext_fake}"))
                        
                    score = self.fid_score.calculate_fid_given_paths(
                        [temp_dir_real, temp_dir_fake], 
                        batch_size=min(batch_size, len(image_pairs) if image_pairs else batch_size),
                        device=self.device,
                        dims=dims
                    )
                finally:
                    shutil.rmtree(temp_dir_real)
                    shutil.rmtree(temp_dir_fake)
            else:
                score = self.fid_score.calculate_fid_given_paths(
                    [path1, path2], 
                    batch_size=batch_size,
                    device=self.device,
                    dims=dims
                )
            return float(score)
        except Exception as e:
            import traceback; traceback.print_exc()
            return float('inf')
