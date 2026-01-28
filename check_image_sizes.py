#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script kiểm tra kích thước ảnh trong Virtual Try-On dataset
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import yaml
from datetime import datetime


def get_image_info(image_path):
    """
    Lấy thông tin về ảnh
    Returns: dict với width, height, channels, file_size_mb
    """
    try:
        # Get file size
        file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
        
        # Open image and get dimensions
        with Image.open(image_path) as img:
            width, height = img.size
            
            # Convert to RGB to check channels
            if img.mode == 'RGB':
                channels = 3
            elif img.mode == 'RGBA':
                channels = 4
            elif img.mode == 'L':
                channels = 1
            else:
                channels = len(img.getbands())
        
        return {
            'width': width,
            'height': height,
            'channels': channels,
            'file_size_mb': round(file_size_mb, 2),
            'status': 'OK'
        }
    except Exception as e:
        return {
            'width': None,
            'height': None,
            'channels': None,
            'file_size_mb': None,
            'status': 'ERROR: {}'.format(str(e))
        }


def get_image_pairs(original_dir, generated_dir):
    """
    Tìm và map các cặp ảnh gốc-sinh
    """
    pairs = []
    
    # Get all original images
    original_files = [f for f in os.listdir(original_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for original_file in original_files:
        # Extract base name
        base_name = os.path.splitext(original_file)[0]
        
        # Construct expected generated filename
        generated_file = "{}_catvton_00001_.png".format(base_name)
        
        original_path = os.path.join(original_dir, original_file)
        generated_path = os.path.join(generated_dir, generated_file)
        
        # Check if generated file exists
        if os.path.exists(generated_path):
            pairs.append((original_path, generated_path, original_file, generated_file))
        else:
            print("Warning: Generated file not found for {}".format(original_file))
    
    return pairs


def check_image_sizes():
    """
    Main function để kiểm tra kích thước ảnh
    """
    print("Image Size Checker for Virtual Try-On Dataset")
    print("=" * 50)
    
    # Load config
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        original_dir = config['paths']['original_dir']
        generated_dir = config['paths']['generated_dir']
    except Exception as e:
        print("Error loading config: {}".format(e))
        # Fallback to default paths
        original_dir = "../12A_THPT_HATRUNG"
        generated_dir = "../CatVtion_12A_THPT_HATRUNG"
    
    print("Original directory: {}".format(original_dir))
    print("Generated directory: {}".format(generated_dir))
    
    # Get image pairs
    print("\nFinding image pairs...")
    image_pairs = get_image_pairs(original_dir, generated_dir)
    print("Found {} image pairs".format(len(image_pairs)))
    
    if len(image_pairs) == 0:
        print("No image pairs found! Please check your paths.")
        return
    
    # Analyze each pair
    print("\nAnalyzing image sizes...")
    results = []
    
    for original_path, generated_path, original_file, generated_file in image_pairs:
        print("Processing: {}".format(original_file))
        
        # Get info for both images
        orig_info = get_image_info(original_path)
        gen_info = get_image_info(generated_path)
        
        # Check if sizes match
        size_match = (orig_info['width'] == gen_info['width'] and 
                     orig_info['height'] == gen_info['height'] and
                     orig_info['channels'] == gen_info['channels'])
        
        result = {
            'original_file': original_file,
            'generated_file': generated_file,
            'original_width': orig_info['width'],
            'original_height': orig_info['height'],
            'original_channels': orig_info['channels'],
            'original_size_mb': orig_info['file_size_mb'],
            'original_status': orig_info['status'],
            'generated_width': gen_info['width'],
            'generated_height': gen_info['height'],
            'generated_channels': gen_info['channels'],
            'generated_size_mb': gen_info['file_size_mb'],
            'generated_status': gen_info['status'],
            'size_match': size_match,
            'original_resolution': "{}x{}".format(orig_info['width'], orig_info['height']) if orig_info['width'] else 'ERROR',
            'generated_resolution': "{}x{}".format(gen_info['width'], gen_info['height']) if gen_info['width'] else 'ERROR'
        }
        
        results.append(result)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Print summary statistics
    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS")
    print("=" * 50)
    
    total_pairs = len(df)
    matching_pairs = df['size_match'].sum()
    non_matching_pairs = total_pairs - matching_pairs
    
    print("Total image pairs: {}".format(total_pairs))
    print("Pairs with matching sizes: {} ({:.1f}%)".format(
        matching_pairs, (matching_pairs/total_pairs)*100))
    print("Pairs with different sizes: {} ({:.1f}%)".format(
        non_matching_pairs, (non_matching_pairs/total_pairs)*100))
    
    # Original images statistics
    orig_valid = df[df['original_status'] == 'OK']
    if len(orig_valid) > 0:
        print("\nORIGINAL IMAGES:")
        print("  Resolution range: {}x{} to {}x{}".format(
            orig_valid['original_width'].min(), orig_valid['original_height'].min(),
            orig_valid['original_width'].max(), orig_valid['original_height'].max()))
        print("  Most common resolution: {}".format(
            orig_valid['original_resolution'].mode().iloc[0] if not orig_valid['original_resolution'].mode().empty else 'N/A'))
        print("  File size range: {:.2f} MB to {:.2f} MB".format(
            orig_valid['original_size_mb'].min(), orig_valid['original_size_mb'].max()))
        print("  Average file size: {:.2f} MB".format(orig_valid['original_size_mb'].mean()))
    
    # Generated images statistics  
    gen_valid = df[df['generated_status'] == 'OK']
    if len(gen_valid) > 0:
        print("\nGENERATED IMAGES:")
        print("  Resolution range: {}x{} to {}x{}".format(
            gen_valid['generated_width'].min(), gen_valid['generated_height'].min(),
            gen_valid['generated_width'].max(), gen_valid['generated_height'].max()))
        print("  Most common resolution: {}".format(
            gen_valid['generated_resolution'].mode().iloc[0] if not gen_valid['generated_resolution'].mode().empty else 'N/A'))
        print("  File size range: {:.2f} MB to {:.2f} MB".format(
            gen_valid['generated_size_mb'].min(), gen_valid['generated_size_mb'].max()))
        print("  Average file size: {:.2f} MB".format(gen_valid['generated_size_mb'].mean()))
    
    # Show non-matching pairs if any
    if non_matching_pairs > 0:
        print("\n" + "!" * 50)
        print("PAIRS WITH DIFFERENT SIZES:")
        print("!" * 50)
        non_matching = df[~df['size_match']]
        for _, row in non_matching.iterrows():
            print("File: {}".format(row['original_file']))
            print("  Original: {} ({} channels)".format(row['original_resolution'], row['original_channels']))
            print("  Generated: {} ({} channels)".format(row['generated_resolution'], row['generated_channels']))
            print()
    
    # Show any errors
    errors = df[(df['original_status'] != 'OK') | (df['generated_status'] != 'OK')]
    if len(errors) > 0:
        print("\n" + "!" * 50)
        print("FILES WITH ERRORS:")
        print("!" * 50)
        for _, row in errors.iterrows():
            print("File: {}".format(row['original_file']))
            if row['original_status'] != 'OK':
                print("  Original: {}".format(row['original_status']))
            if row['generated_status'] != 'OK':
                print("  Generated: {}".format(row['generated_status']))
            print()
    
    # Save detailed CSV report
    csv_path = 'image_size_report.csv'
    df.to_csv(csv_path, index=False)
    print("\nDetailed report saved to: {}".format(csv_path))
    
    # Create visualizations if there are valid images
    if len(orig_valid) > 0 and len(gen_valid) > 0:
        create_size_visualization(orig_valid, gen_valid)
    
    print("\n" + "=" * 50)
    print("Size check completed!")
    print("Check '{}' for detailed results".format(csv_path))
    return df


def create_size_visualization(orig_df, gen_df):
    """
    Tạo visualization cho kích thước ảnh
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Resolution distribution (Original)
    resolutions_orig = orig_df['original_resolution'].value_counts()
    axes[0,0].bar(range(len(resolutions_orig)), resolutions_orig.values)
    axes[0,0].set_title('Original Image Resolutions')
    axes[0,0].set_xlabel('Resolution')
    axes[0,0].set_ylabel('Count')
    axes[0,0].set_xticks(range(len(resolutions_orig)))
    axes[0,0].set_xticklabels(resolutions_orig.index, rotation=45, ha='right')
    
    # Plot 2: Resolution distribution (Generated)
    resolutions_gen = gen_df['generated_resolution'].value_counts()
    axes[0,1].bar(range(len(resolutions_gen)), resolutions_gen.values)
    axes[0,1].set_title('Generated Image Resolutions')
    axes[0,1].set_xlabel('Resolution')
    axes[0,1].set_ylabel('Count')
    axes[0,1].set_xticks(range(len(resolutions_gen)))
    axes[0,1].set_xticklabels(resolutions_gen.index, rotation=45, ha='right')
    
    # Plot 3: File size comparison
    axes[1,0].scatter(orig_df['original_size_mb'], gen_df['generated_size_mb'], alpha=0.6)
    axes[1,0].set_xlabel('Original File Size (MB)')
    axes[1,0].set_ylabel('Generated File Size (MB)')
    axes[1,0].set_title('File Size Comparison')
    # Add diagonal line
    max_size = max(orig_df['original_size_mb'].max(), gen_df['generated_size_mb'].max())
    axes[1,0].plot([0, max_size], [0, max_size], 'r--', alpha=0.5)
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Size distribution histogram
    axes[1,1].hist([orig_df['original_size_mb'], gen_df['generated_size_mb']], 
                   bins=20, alpha=0.7, label=['Original', 'Generated'])
    axes[1,1].set_xlabel('File Size (MB)')
    axes[1,1].set_ylabel('Count')
    axes[1,1].set_title('File Size Distribution')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig('image_size_analysis.png', dpi=300, bbox_inches='tight')
    print("Visualization saved to: image_size_analysis.png")
    plt.show()


if __name__ == '__main__':
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Run the check
    try:
        df = check_image_sizes()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print("Error during size check: {}".format(e))
        import traceback
        traceback.print_exc()
