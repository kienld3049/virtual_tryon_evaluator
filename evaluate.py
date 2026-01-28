# -*- coding: utf-8 -*-
"""
Main Virtual Try-On Evaluation Script
"""

import os
import sys
import yaml
import json
import argparse
import pandas as pd
from tqdm import tqdm
import numpy as np
from datetime import datetime

# Try to import torch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: torch not available. Using CPU only.")

# Import local modules
from metrics import ImageQualityMetrics, FIDMetric
from utils import (
    load_image, ensure_same_size, get_image_pairs,
    create_metrics_visualization, create_summary_plots, save_best_worst_samples
)


class VirtualTryOnEvaluator:
    """Main evaluator class cho Virtual Try-On"""
    
    def __init__(self, config_path='config.yaml'):
        """Initialize evaluator với config file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device
        if TORCH_AVAILABLE and self.config['evaluation']['processing']['device'] == 'cuda' and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        print(f"Using device: {self.device}")
        
        # Initialize metric calculators
        self._init_metrics()
        
        # Get image pairs
        self.image_pairs = get_image_pairs(
            self.config['paths']['original_dir'],
            self.config['paths']['generated_dir']
        )
        print(f"Found {len(self.image_pairs)} image pairs")
        
        # Create output directory
        self.output_dir = self.config['paths']['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _init_metrics(self):
        """Initialize metric calculation objects"""
        # Image quality metrics
        self.img_quality = ImageQualityMetrics(
            device=self.device,
            lpips_net=self.config['models']['lpips']['net']
        )
        
        # FID metric
        self.fid_metric = FIDMetric(device=self.device)
    
    def calculate_single_pair_metrics(self, original_path, generated_path):
        """Tính metrics cho 1 cặp ảnh"""
        # Load images
        original_img = load_image(original_path)
        generated_img = load_image(generated_path)
        
        if original_img is None or generated_img is None:
            return None
        
        # Ensure same size
        original_img, generated_img = ensure_same_size(original_img, generated_img)
        
        results = {
            'original_file': os.path.basename(original_path),
            'generated_file': os.path.basename(generated_path),
        }
        
        # Image Quality Metrics
        if self.config['evaluation']['metrics']['enable_ssim']:
            results['ssim'] = self.img_quality.calculate_ssim(original_img, generated_img)
        
        if self.config['evaluation']['metrics']['enable_psnr']:
            results['psnr'] = self.img_quality.calculate_psnr(original_img, generated_img)
        
        if self.config['evaluation']['metrics']['enable_mse']:
            results['mse'] = self.img_quality.calculate_mse(original_img, generated_img)
        
        if self.config['evaluation']['metrics']['enable_lpips']:
            results['lpips'] = self.img_quality.calculate_lpips(original_img, generated_img)
        
        if self.config['evaluation']['metrics']['enable_color_distance']:
            results['color_distance'] = self.img_quality.calculate_color_histogram_distance(original_img, generated_img)
        
        if self.config['evaluation']['metrics']['enable_edge_consistency']:
            results['edge_consistency'] = self.img_quality.calculate_edge_consistency(original_img, generated_img)
        
        return results
    
    def evaluate_all_pairs(self):
        """Evaluate tất cả các cặp ảnh"""
        all_results = []
        
        print("Evaluating image pairs...")
        for original_path, generated_path in tqdm(self.image_pairs, desc="Processing"):
            try:
                result = self.calculate_single_pair_metrics(original_path, generated_path)
                if result is not None:
                    all_results.append(result)
                else:
                    print(f"Failed to process: {os.path.basename(original_path)}")
            except Exception as e:
                print(f"Error processing {os.path.basename(original_path)}: {e}")
                continue
        
        return pd.DataFrame(all_results)
    
    def calculate_fid_score(self):
        """Tính FID score cho toàn bộ dataset"""
        if not self.config['evaluation']['metrics']['enable_fid']:
            return None
        
        print("Calculating FID score...")
        try:
            fid_score = self.fid_metric.calculate_fid_from_paths(
                self.config['paths']['original_dir'],
                self.config['paths']['generated_dir'],
                batch_size=self.config['evaluation']['processing']['batch_size']
            )
            return fid_score
        except Exception as e:
            print(f"Error calculating FID: {e}")
            return None
    
    def generate_summary_statistics(self, metrics_df):
        """Tạo summary statistics"""
        numeric_columns = metrics_df.select_dtypes(include=[np.number]).columns
        
        summary = {}
        for col in numeric_columns:
            summary[col] = {
                'mean': float(metrics_df[col].mean()),
                'std': float(metrics_df[col].std()),
                'median': float(metrics_df[col].median()),
                'min': float(metrics_df[col].min()),
                'max': float(metrics_df[col].max()),
                'q25': float(metrics_df[col].quantile(0.25)),
                'q75': float(metrics_df[col].quantile(0.75))
            }
        
        return summary
    
    def create_html_report(self, metrics_df, summary_stats, fid_score=None):
        """Tạo HTML report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Virtual Try-On Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 30px 0; }}
                .metric-table {{ border-collapse: collapse; width: 100%; }}
                .metric-table th, .metric-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .metric-table th {{ background-color: #f2f2f2; }}
                .highlight {{ background-color: #fffacd; }}
                .good {{ color: green; font-weight: bold; }}
                .bad {{ color: red; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Virtual Try-On Evaluation Report</h1>
                <p>Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Total Image Pairs: {len(metrics_df)}</p>
                <p>Device Used: {self.device}</p>
            </div>
            
            <div class="section">
                <h2>Summary Statistics</h2>
                <table class="metric-table">
                    <tr><th>Metric</th><th>Mean</th><th>Std</th><th>Median</th><th>Min</th><th>Max</th><th>Interpretation</th></tr>
        """
        
        # Add metric rows
        metric_interpretations = {
            'ssim': 'Higher is better (0-1)',
            'psnr': 'Higher is better (dB)',
            'lpips': 'Lower is better (0-1)',
            'mse': 'Lower is better',
            'face_similarity': 'Higher is better (0-1)',
            'color_distance': 'Lower is better',
            'edge_consistency': 'Higher is better (0-1)'
        }
        
        for metric, stats in summary_stats.items():
            if metric != 'face_detection_rate':
                interpretation = metric_interpretations.get(metric, 'N/A')
                mean_val = stats['mean']
                
                # Color coding based on typical good/bad values
                if metric in ['ssim', 'psnr', 'face_similarity', 'edge_consistency']:
                    color_class = 'good' if mean_val > 0.7 else 'bad' if mean_val < 0.5 else ''
                elif metric in ['lpips', 'mse', 'color_distance']:
                    color_class = 'good' if mean_val < 0.3 else 'bad' if mean_val > 0.7 else ''
                else:
                    color_class = ''
                
                html_content += f"""
                    <tr class="{color_class}">
                        <td>{metric.upper()}</td>
                        <td>{mean_val:.4f}</td>
                        <td>{stats['std']:.4f}</td>
                        <td>{stats['median']:.4f}</td>
                        <td>{stats['min']:.4f}</td>
                        <td>{stats['max']:.4f}</td>
                        <td>{interpretation}</td>
                    </tr>
                """
        
        # Add FID score if available
        if fid_score is not None:
            fid_color = 'good' if fid_score < 30 else 'bad' if fid_score > 100 else ''
            html_content += f"""
                <tr class="{fid_color}">
                    <td>FID</td>
                    <td colspan="5">{fid_score:.2f}</td>
                    <td>Lower is better (< 30 is good)</td>
                </tr>
            """
        
        # Add face detection rate
        if 'face_detection_rate' in summary_stats:
            detection_rate = summary_stats['face_detection_rate']
            detection_color = 'good' if detection_rate > 0.8 else 'bad' if detection_rate < 0.5 else ''
            html_content += f"""
                <tr class="{detection_color}">
                    <td>Face Detection Rate</td>
                    <td colspan="5">{detection_rate:.2%}</td>
                    <td>Percentage of images with detected faces</td>
                </tr>
            """
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Analysis</h2>
                <h3>Generated Visualizations:</h3>
                <ul>
                    <li><strong>metric_distributions.png</strong>: Distribution plots cho tất cả metrics</li>
                    <li><strong>correlation_matrix.png</strong>: Correlation matrix giữa các metrics</li>
                    <li><strong>summary_plots.png</strong>: Tổng quan performance</li>
                    <li><strong>visual_samples/</strong>: Best/worst examples cho từng metric</li>
                </ul>
                
                <h3>Key Findings:</h3>
        """
        
        # Add some automated insights
        if 'ssim' in summary_stats:
            ssim_mean = summary_stats['ssim']['mean']
            if ssim_mean > 0.8:
                html_content += "<li>✅ Excellent structural similarity (SSIM > 0.8)</li>"
            elif ssim_mean > 0.6:
                html_content += "<li>⚠️ Moderate structural similarity (0.6 < SSIM < 0.8)</li>"
            else:
                html_content += "<li>❌ Poor structural similarity (SSIM < 0.6)</li>"
        
        if 'lpips' in summary_stats:
            lpips_mean = summary_stats['lpips']['mean']
            if lpips_mean < 0.2:
                html_content += "<li>✅ Excellent perceptual similarity (LPIPS < 0.2)</li>"
            elif lpips_mean < 0.5:
                html_content += "<li>⚠️ Moderate perceptual similarity (0.2 < LPIPS < 0.5)</li>"
            else:
                html_content += "<li>❌ Poor perceptual similarity (LPIPS > 0.5)</li>"
        
        if fid_score is not None:
            if fid_score < 30:
                html_content += f"<li>✅ Excellent FID score ({fid_score:.1f} < 30)</li>"
            elif fid_score < 100:
                html_content += f"<li>⚠️ Moderate FID score (30 < {fid_score:.1f} < 100)</li>"
            else:
                html_content += f"<li>❌ Poor FID score ({fid_score:.1f} > 100)</li>"
        
        html_content += """
                </ul>
            </div>
            
            <div class="section">
                <h2>Files Generated</h2>
                <ul>
                    <li><strong>detailed_metrics.csv</strong>: Chi tiết metrics cho từng ảnh</li>
                    <li><strong>summary_statistics.json</strong>: Summary statistics</li>
                    <li><strong>evaluation_report.html</strong>: Báo cáo này</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    def run_evaluation(self):
        """Chạy toàn bộ evaluation pipeline"""
        print("Starting Virtual Try-On Evaluation...")
        print("=" * 50)
        
        # 1. Evaluate individual pairs
        metrics_df = self.evaluate_all_pairs()
        
        if len(metrics_df) == 0:
            print("No valid image pairs found!")
            return
        
        # 2. Calculate FID score
        fid_score = self.calculate_fid_score()
        
        # 3. Generate summary statistics
        summary_stats = self.generate_summary_statistics(metrics_df)
        
        # 4. Save results
        if self.config['evaluation']['output']['save_detailed_csv']:
            csv_path = os.path.join(self.output_dir, 'detailed_metrics.csv')
            metrics_df.to_csv(csv_path, index=False)
            print(f"Detailed metrics saved to: {csv_path}")
        
        if self.config['evaluation']['output']['save_summary_json']:
            summary_dict = {
                'summary_statistics': summary_stats,
                'fid_score': fid_score,
                'total_pairs': len(metrics_df),
                'evaluation_date': datetime.now().isoformat(),
                'config': self.config
            }
            json_path = os.path.join(self.output_dir, 'summary_statistics.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(summary_dict, f, indent=2, ensure_ascii=False)
            print(f"Summary statistics saved to: {json_path}")
        
        # 5. Generate visualizations
        if self.config['evaluation']['output']['generate_plots']:
            print("Generating visualizations...")
            
            # Distribution and correlation plots
            create_metrics_visualization(metrics_df, self.output_dir)
            
            # Summary plots
            create_summary_plots(metrics_df, self.output_dir)
            
            print("Visualization plots saved to output directory")
        
        # 6. Save visual samples
        if self.config['evaluation']['output']['save_visual_samples']:
            print("Saving visual samples...")
            samples_dir = os.path.join(self.output_dir, 'visual_samples')
            n_samples = self.config['evaluation']['output']['num_visual_samples']
            
            # Save best/worst for key metrics
            for metric in ['ssim', 'lpips', 'psnr']:
                if metric in metrics_df.columns:
                    save_best_worst_samples(metrics_df, self.image_pairs, metric, samples_dir, n_samples)
            
            print(f"Visual samples saved to: {samples_dir}")
        
        # 7. Generate HTML report
        if self.config['evaluation']['output']['create_html_report']:
            html_content = self.create_html_report(metrics_df, summary_stats, fid_score)
            html_path = os.path.join(self.output_dir, 'evaluation_report.html')
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"HTML report saved to: {html_path}")
        
        # 8. Print summary
        print("\n" + "=" * 50)
        print("EVALUATION COMPLETED!")
        print(f"Processed {len(metrics_df)} image pairs")
        print(f"Results saved to: {self.output_dir}")
        
        # Print key metrics
        print("\nKey Metrics:")
        if 'ssim' in summary_stats:
            print(f"  SSIM (avg): {summary_stats['ssim']['mean']:.3f} ± {summary_stats['ssim']['std']:.3f}")
        if 'lpips' in summary_stats:
            print(f"  LPIPS (avg): {summary_stats['lpips']['mean']:.3f} ± {summary_stats['lpips']['std']:.3f}")
        if 'psnr' in summary_stats:
            print(f"  PSNR (avg): {summary_stats['psnr']['mean']:.1f} ± {summary_stats['psnr']['std']:.1f} dB")
        if fid_score is not None:
            print(f"  FID Score: {fid_score:.2f}")
        if 'face_detection_rate' in summary_stats:
            print(f"  Face Detection Rate: {summary_stats['face_detection_rate']:.1%}")


def main():
    parser = argparse.ArgumentParser(description='Virtual Try-On Evaluation Tool')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--original_dir', type=str, help='Override original images directory')
    parser.add_argument('--generated_dir', type=str, help='Override generated images directory')
    parser.add_argument('--output_dir', type=str, help='Override output directory')
    
    args = parser.parse_args()
    
    # Load config
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        sys.exit(1)
    
    # Initialize evaluator
    evaluator = VirtualTryOnEvaluator(args.config)
    
    # Override paths if provided
    if args.original_dir:
        evaluator.config['paths']['original_dir'] = args.original_dir
    if args.generated_dir:
        evaluator.config['paths']['generated_dir'] = args.generated_dir
    if args.output_dir:
        evaluator.config['paths']['output_dir'] = args.output_dir
        evaluator.output_dir = args.output_dir
    
    # Run evaluation
    try:
        evaluator.run_evaluation()
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
