#!/usr/bin/env python3
"""
OCR Comparison Evaluation Script

This script compares different OCR engine results side by side.

Usage:
    python scripts/comparison_evaluation.py --predictions_list results1.csv results2.csv --ground_truth annotations.csv --names Engine1 Engine2
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from ocr_evaluation import OCREvaluator, EvaluationMetrics


def compare_ocr_engines(prediction_files: List[str], ground_truth_file: str, 
                       engine_names: List[str], output_dir: str) -> Dict[str, EvaluationMetrics]:
    """Compare multiple OCR engines."""
    
    evaluator = OCREvaluator({
        "case_sensitive": False,
        "normalize_text": True,
        "ignore_chars": ["-", ".", "_", " "]
    })
    
    results = {}
    
    # Load ground truth once
    print(f"Loading ground truth: {ground_truth_file}")
    ground_truth = evaluator.load_csv_data(Path(ground_truth_file))
    
    # Evaluate each prediction file
    for pred_file, engine_name in zip(prediction_files, engine_names):
        print(f"\n🔍 Evaluating {engine_name}...")
        print(f"   Predictions: {pred_file}")
        
        try:
            predictions = evaluator.load_csv_data(Path(pred_file))
            metrics = evaluator.evaluate_predictions(predictions, ground_truth)
            results[engine_name] = metrics
            
            print(f"   ✅ {engine_name}: Accuracy={metrics.plate_accuracy:.3f}, CER={metrics.cer:.3f}")
            
        except Exception as e:
            print(f"   ❌ Error evaluating {engine_name}: {e}")
    
    return results


def generate_comparison_plots(results: Dict[str, EvaluationMetrics], output_dir: Path):
    """Generate comparison plots."""
    if len(results) < 2:
        print("Need at least 2 engines for comparison")
        return
    
    engines = list(results.keys())
    
    # Prepare data
    metrics_data = {
        'Engine': [],
        'Plate Accuracy': [],
        'Character Error Rate': [],
        'Precision': [],
        'Recall': [],
        'F1 Score': [],
        'Avg Latency (ms)': [],
        'Throughput (plates/s)': []
    }
    
    for engine, metrics in results.items():
        metrics_data['Engine'].append(engine)
        metrics_data['Plate Accuracy'].append(metrics.plate_accuracy)
        metrics_data['Character Error Rate'].append(metrics.cer)
        metrics_data['Precision'].append(metrics.plate_precision)
        metrics_data['Recall'].append(metrics.plate_recall)
        metrics_data['F1 Score'].append(metrics.plate_f1)
        metrics_data['Avg Latency (ms)'].append(metrics.avg_latency * 1000)
        metrics_data['Throughput (plates/s)'].append(metrics.throughput)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy comparison
    accuracy_metrics = ['Plate Accuracy', 'Precision', 'Recall', 'F1 Score']
    accuracy_data = []
    for metric in accuracy_metrics:
        for i, engine in enumerate(engines):
            accuracy_data.append({
                'Engine': engine,
                'Metric': metric,
                'Value': metrics_data[metric][i]
            })
    
    df_accuracy = pd.DataFrame(accuracy_data)
    sns.barplot(data=df_accuracy, x='Metric', y='Value', hue='Engine', ax=axes[0, 0])
    axes[0, 0].set_title('Accuracy Metrics Comparison')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Error rate comparison
    axes[0, 1].bar(engines, metrics_data['Character Error Rate'], color=['#E74C3C', '#E67E22', '#F39C12'][:len(engines)])
    axes[0, 1].set_title('Character Error Rate Comparison')
    axes[0, 1].set_ylabel('CER')
    for i, v in enumerate(metrics_data['Character Error Rate']):
        axes[0, 1].text(i, v + max(metrics_data['Character Error Rate']) * 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # Latency comparison
    axes[1, 0].bar(engines, metrics_data['Avg Latency (ms)'], color=['#3498DB', '#9B59B6', '#1ABC9C'][:len(engines)])
    axes[1, 0].set_title('Average Latency Comparison')
    axes[1, 0].set_ylabel('Latency (ms)')
    for i, v in enumerate(metrics_data['Avg Latency (ms)']):
        axes[1, 0].text(i, v + max(metrics_data['Avg Latency (ms)']) * 0.01, f'{v:.1f}ms', ha='center', va='bottom')
    
    # Throughput comparison
    axes[1, 1].bar(engines, metrics_data['Throughput (plates/s)'], color=['#27AE60', '#8E44AD', '#D35400'][:len(engines)])
    axes[1, 1].set_title('Throughput Comparison')
    axes[1, 1].set_ylabel('Plates per Second')
    for i, v in enumerate(metrics_data['Throughput (plates/s)']):
        axes[1, 1].text(i, v + max(metrics_data['Throughput (plates/s)']) * 0.01, f'{v:.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    output_path = output_dir / "engine_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Comparison plots saved to {output_path}")


def save_comparison_report(results: Dict[str, EvaluationMetrics], output_dir: Path):
    """Save comparison report."""
    report_path = output_dir / "comparison_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("OCR ENGINES COMPARISON REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        # Summary table
        f.write("SUMMARY COMPARISON\n")
        f.write("-" * 30 + "\n")
        f.write(f"{'Engine':<15} {'Accuracy':<10} {'CER':<8} {'F1':<8} {'Latency(ms)':<12} {'Throughput':<10}\n")
        f.write("-" * 65 + "\n")
        
        for engine, metrics in results.items():
            f.write(f"{engine:<15} {metrics.plate_accuracy:<10.3f} {metrics.cer:<8.3f} "
                   f"{metrics.plate_f1:<8.3f} {metrics.avg_latency*1000:<12.1f} {metrics.throughput:<10.0f}\n")
        
        f.write("\nDETAILED METRICS\n")
        f.write("-" * 30 + "\n")
        
        for engine, metrics in results.items():
            f.write(f"\n{engine.upper()}\n")
            f.write(f"  Plate Accuracy: {metrics.plate_accuracy:.4f}\n")
            f.write(f"  Character Error Rate: {metrics.cer:.4f}\n")
            f.write(f"  Precision: {metrics.plate_precision:.4f}\n")
            f.write(f"  Recall: {metrics.plate_recall:.4f}\n")
            f.write(f"  F1 Score: {metrics.plate_f1:.4f}\n")
            f.write(f"  Average Latency: {metrics.avg_latency*1000:.2f}ms\n")
            f.write(f"  Throughput: {metrics.throughput:.1f} plates/sec\n")
            f.write(f"  Total Samples: {metrics.total_samples}\n")
            f.write(f"  Correct Detections: {metrics.correct_detections}\n")
            f.write(f"  Character Errors: {metrics.insertion_errors + metrics.deletion_errors + metrics.substitution_errors}\n")
        
        # Best performer analysis
        f.write("\nBEST PERFORMERS\n")
        f.write("-" * 20 + "\n")
        
        best_accuracy = max(results.items(), key=lambda x: x[1].plate_accuracy)
        best_speed = max(results.items(), key=lambda x: x[1].throughput)
        best_cer = min(results.items(), key=lambda x: x[1].cer)
        
        f.write(f"Best Accuracy: {best_accuracy[0]} ({best_accuracy[1].plate_accuracy:.3f})\n")
        f.write(f"Best Speed: {best_speed[0]} ({best_speed[1].throughput:.0f} plates/sec)\n")
        f.write(f"Lowest CER: {best_cer[0]} ({best_cer[1].cer:.3f})\n")
    
    print(f"📄 Comparison report saved to {report_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Compare OCR engine performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare FastPlateOCR with PaddleOCR
  python scripts/comparison_evaluation.py \\
    --predictions_list output/fast_results.csv output/paddle_results.csv \\
    --ground_truth lp_data/preprocessed_dataset/valid_anotaciones.csv \\
    --names FastPlateOCR PaddleOCR \\
    --output_dir comparison_results
        """
    )
    
    parser.add_argument(
        '--predictions_list',
        nargs='+',
        required=True,
        help='List of CSV files with predictions from different engines'
    )
    
    parser.add_argument(
        '--ground_truth',
        type=str,
        required=True,
        help='CSV file with ground truth annotations'
    )
    
    parser.add_argument(
        '--names',
        nargs='+',
        required=True,
        help='Names of the OCR engines corresponding to prediction files'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='comparison_results',
        help='Output directory for comparison results'
    )
    
    args = parser.parse_args()
    
    # Validation
    if len(args.predictions_list) != len(args.names):
        print("Error: Number of prediction files must match number of engine names")
        return 1
    
    if len(args.predictions_list) < 2:
        print("Error: Need at least 2 prediction files for comparison")
        return 1
    
    # Check files exist
    for pred_file in args.predictions_list:
        if not Path(pred_file).exists():
            print(f"Error: Prediction file does not exist: {pred_file}")
            return 1
    
    if not Path(args.ground_truth).exists():
        print(f"Error: Ground truth file does not exist: {args.ground_truth}")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print("🚀 Starting OCR Engine Comparison")
        print("=" * 50)
        
        # Compare engines
        results = compare_ocr_engines(
            args.predictions_list,
            args.ground_truth,
            args.names,
            args.output_dir
        )
        
        if not results:
            print("❌ No valid results obtained")
            return 1
        
        # Generate outputs
        save_comparison_report(results, output_dir)
        generate_comparison_plots(results, output_dir)
        
        # Print summary
        print("\n🎯 COMPARISON SUMMARY")
        print("=" * 40)
        
        for engine, metrics in results.items():
            print(f"{engine:.<20} Acc:{metrics.plate_accuracy:.3f} CER:{metrics.cer:.3f} "
                  f"Speed:{metrics.throughput:.0f}pps")
        
        print(f"\n📁 Results saved to: {output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 