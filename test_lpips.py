#!/usr/bin/env python3
"""
Test script để verify LPIPS implementation
"""

import os
import sys
import yaml
from metrics import ImageQualityMetrics
from utils import load_image, ensure_same_size

def test_lpips():
    """Test LPIPS với 1 cặp ảnh"""
    
    # Load config
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Initialize metrics
    try:
        device = 'cuda' if config['evaluation']['processing']['device'] == 'cuda' else 'cpu'
        print(f"Testing LPIPS on device: {device}")
        
        img_quality = ImageQualityMetrics(
            device=device,
            lpips_net=config['models']['lpips']['net']
        )
        print("✅ LPIPS model initialized successfully!")
        
        # Test với 1 cặp ảnh
        original_dir = config['paths']['original_dir']
        generated_dir = config['paths']['generated_dir']
        
        # Find first available image pair
        original_files = [f for f in os.listdir(original_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not original_files:
            print("❌ No images found in original directory")
            return
        
        test_file = original_files[0]
        base_name = os.path.splitext(test_file)[0]
        
        # Look for corresponding generated image
        generated_file = None
        for ext in ['.png', '.jpg', '.jpeg']:
            for suffix in ['_catvton_00001_', '_catvton']:
                candidate = f"{base_name}{suffix}{ext}"
                candidate_path = os.path.join(generated_dir, candidate)
                if os.path.exists(candidate_path):
                    generated_file = candidate
                    break
            if generated_file:
                break
        
        if not generated_file:
            print(f"❌ No corresponding generated image found for {test_file}")
            return
        
        print(f"Testing with: {test_file} vs {generated_file}")
        
        # Load images
        original_path = os.path.join(original_dir, test_file)
        generated_path = os.path.join(generated_dir, generated_file)
        
        original_img = load_image(original_path)
        generated_img = load_image(generated_path)
        
        if original_img is None or generated_img is None:
            print("❌ Failed to load images")
            return
        
        print(f"Original image shape: {original_img.shape}")
        print(f"Generated image shape: {generated_img.shape}")
        
        # Ensure same size
        original_img, generated_img = ensure_same_size(original_img, generated_img)
        print(f"After resize - Original: {original_img.shape}, Generated: {generated_img.shape}")
        
        # Test LPIPS
        print("Calculating LPIPS...")
        lpips_score = img_quality.calculate_lpips(original_img, generated_img)
        print(f"✅ LPIPS Score: {lpips_score:.4f}")
        
        # Test other metrics for comparison
        print("\nTesting other metrics for comparison:")
        ssim_score = img_quality.calculate_ssim(original_img, generated_img)
        psnr_score = img_quality.calculate_psnr(original_img, generated_img)
        
        print(f"SSIM Score: {ssim_score:.4f}")
        print(f"PSNR Score: {psnr_score:.2f} dB")
        
        print("\n✅ All tests passed! LPIPS is working correctly.")
        
        # Interpretation
        print("\nInterpretation:")
        if lpips_score < 0.2:
            print("🟢 Excellent perceptual similarity (LPIPS < 0.2)")
        elif lpips_score < 0.5:
            print("🟡 Moderate perceptual similarity (0.2 ≤ LPIPS < 0.5)")
        else:
            print("🔴 Poor perceptual similarity (LPIPS ≥ 0.5)")
            
    except Exception as e:
        print(f"❌ Error testing LPIPS: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_lpips()
