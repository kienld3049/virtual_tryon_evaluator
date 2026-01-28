# -*- coding: utf-8 -*-
"""
Test script để kiểm tra Virtual Try-On Evaluation Tool
"""

import os
import sys
import yaml
from metrics import ImageQualityMetrics
from utils import load_image, ensure_same_size, get_image_pairs

def test_basic_functionality():
    """Test các chức năng cơ bản"""
    print("Testing Virtual Try-On Evaluation Tool...")
    print("=" * 50)
    
    # 1. Test config loading
    print("1. Testing config loading...")
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("[SUCCESS] Config loaded successfully")
    except Exception as e:
        print("[ERROR] Config loading failed: {}".format(e))
        return False
    
    # 2. Test image pairs detection
    print("\n2. Testing image pairs detection...")
    try:
        image_pairs = get_image_pairs(
            config['paths']['original_dir'],
            config['paths']['generated_dir']
        )
        print("[SUCCESS] Found {} image pairs".format(len(image_pairs)))
        if len(image_pairs) == 0:
            print("[WARNING] No image pairs found - check paths in config.yaml")
            return False
    except Exception as e:
        print("[ERROR] Image pairs detection failed: {}".format(e))
        return False
    
    # 3. Test image loading
    print("\n3. Testing image loading...")
    try:
        # Test với cặp ảnh đầu tiên
        original_path, generated_path = image_pairs[0]
        print("Testing with: {}".format(os.path.basename(original_path)))
        
        original_img = load_image(original_path)
        generated_img = load_image(generated_path)
        
        if original_img is None or generated_img is None:
            print("[ERROR] Failed to load images")
            return False
        
        print("[SUCCESS] Images loaded - Original: {}, Generated: {}".format(
            original_img.shape, generated_img.shape))
        
        # Ensure same size
        original_img, generated_img = ensure_same_size(original_img, generated_img)
        print("[SUCCESS] Images resized to same size: {}".format(original_img.shape))
        
    except Exception as e:
        print("[ERROR] Image loading failed: {}".format(e))
        return False
    
    # 4. Test metrics calculation
    print("\n4. Testing metrics calculation...")
    try:
        # Initialize metrics
        img_quality = ImageQualityMetrics(device='cpu')  # Force CPU for testing
        
        # Test SSIM
        ssim_score = img_quality.calculate_ssim(original_img, generated_img)
        print("[SUCCESS] SSIM: {:.3f}".format(ssim_score))
        
        # Test PSNR
        psnr_score = img_quality.calculate_psnr(original_img, generated_img)
        print("[SUCCESS] PSNR: {:.1f} dB".format(psnr_score))
        
        # Test MSE
        mse_score = img_quality.calculate_mse(original_img, generated_img)
        print("[SUCCESS] MSE: {:.6f}".format(mse_score))
        
        # Test LPIPS (might be slow) - DISABLED
        # print("Testing LPIPS (this might take a moment)...")
        # lpips_score = img_quality.calculate_lpips(original_img, generated_img)
        # print("[SUCCESS] LPIPS: {:.3f}".format(lpips_score))
        
        # Test additional metrics
        color_dist = img_quality.calculate_color_histogram_distance(original_img, generated_img)
        print("[SUCCESS] Color Distance: {:.3f}".format(color_dist))
        
        edge_consistency = img_quality.calculate_edge_consistency(original_img, generated_img)
        print("[SUCCESS] Edge Consistency: {:.3f}".format(edge_consistency))
        
    except Exception as e:
        print("[ERROR] Metrics calculation failed: {}".format(e))
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 50)
    print("[SUCCESS] All tests passed! Tool is ready to use.")
    print("\nTo run full evaluation:")
    print("python evaluate.py")
    
    return True

def test_sample_evaluation():
    """Test evaluation trên 1 vài sample"""
    print("\n" + "=" * 50)
    print("Running sample evaluation on first 3 image pairs...")
    
    try:
        from evaluate import VirtualTryOnEvaluator
        
        # Initialize evaluator
        evaluator = VirtualTryOnEvaluator('config.yaml')
        
        # Process first 3 pairs
        sample_pairs = evaluator.image_pairs[:3]
        results = []
        
        for i, (original_path, generated_path) in enumerate(sample_pairs):
            print("\nProcessing {}/3: {}".format(i+1, os.path.basename(original_path)))
            result = evaluator.calculate_single_pair_metrics(original_path, generated_path)
            if result:
                results.append(result)
                # Print key metrics
                ssim_val = result.get('ssim', 'N/A')
                lpips_val = result.get('lpips', 'N/A')
                if ssim_val != 'N/A':
                    print("  SSIM: {:.3f}".format(ssim_val))
                else:
                    print("  SSIM: N/A")
                if lpips_val != 'N/A':
                    print("  LPIPS: {:.3f}".format(lpips_val))
                else:
                    print("  LPIPS: N/A")
        
        print("\n[SUCCESS] Successfully processed {}/{} samples".format(len(results), 3))
        return True
        
    except Exception as e:
        print("[ERROR] Sample evaluation failed: {}".format(e))
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("Virtual Try-On Evaluation Tool - Test Script")
    print("=" * 50)
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Run basic tests
    if test_basic_functionality():
        # Run sample evaluation if basic tests pass
        test_sample_evaluation()
    else:
        print("\n[ERROR] Basic tests failed. Please check your setup.")
        sys.exit(1)
