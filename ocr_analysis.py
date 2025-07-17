#!/usr/bin/env python3
"""
OCR Analysis Tool
Compares FastPlate and PaddleOCR results against ground truth annotations
Generates confusion matrices and detailed analytics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import re
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def clean_plate_text(text):
    """Clean plate text by removing trailing underscores and normalizing"""
    if pd.isna(text):
        return ""
    return str(text).strip().rstrip('_').upper()

def normalize_image_path(path):
    """Normalize image path to match between datasets"""
    if pd.isna(path):
        return ""
    # Remove 'all/' prefix if present
    path = str(path).replace('all/', '')
    return path

def calculate_character_accuracy(predicted, actual):
    """Calculate character-level accuracy"""
    if pd.isna(predicted) or pd.isna(actual):
        return 0.0
    
    predicted = str(predicted)
    actual = str(actual)
    
    if len(predicted) == 0 and len(actual) == 0:
        return 1.0
    if len(predicted) == 0 or len(actual) == 0:
        return 0.0
    
    # Calculate edit distance (Levenshtein distance)
    def edit_distance(s1, s2):
        if len(s1) < len(s2):
            return edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    distance = edit_distance(predicted, actual)
    max_len = max(len(predicted), len(actual))
    return 1.0 - (distance / max_len)

def load_and_process_data():
    """Load and process all CSV files"""
    print("Loading data files...")
    
    # Load ground truth
    gt_df = pd.read_csv('data/merged_dataset/all_annotations.csv')
    gt_df['image_path'] = gt_df['image_path'].apply(normalize_image_path)
    gt_df['plate_text'] = gt_df['plate_text'].apply(clean_plate_text)
    print(f"Ground truth: {len(gt_df)} records")
    
    # Load FastPlate results
    fp_df = pd.read_csv('data/outputs/ocr/fast_plate_all.csv')
    fp_df['image_path'] = fp_df['image_path'].apply(normalize_image_path)
    fp_df['plate_text'] = fp_df['plate_text'].apply(clean_plate_text)
    print(f"FastPlate: {len(fp_df)} records")
    
    # Load PaddleOCR results
    po_df = pd.read_csv('data/outputs/ocr/paddleocr_v5_all.csv')
    po_df['image_path'] = po_df['image_path'].apply(normalize_image_path)
    po_df['plate_text'] = po_df['plate_text'].apply(clean_plate_text)
    print(f"PaddleOCR: {len(po_df)} records")
    
    return gt_df, fp_df, po_df

def merge_datasets(gt_df, fp_df, po_df):
    """Merge datasets on image_path"""
    print("\nMerging datasets...")
    
    # Merge ground truth with FastPlate
    merged_fp = pd.merge(gt_df, fp_df[['image_path', 'plate_text', 'confidence', 'processing_time']], 
                        on='image_path', how='left', suffixes=('_gt', '_fp'))
    
    # Merge with PaddleOCR
    merged_all = pd.merge(merged_fp, po_df[['image_path', 'plate_text', 'confidence', 'processing_time']], 
                         on='image_path', how='left', suffixes=('', '_po'))
    
    # Rename columns for clarity
    merged_all.rename(columns={
        'plate_text_gt': 'ground_truth',
        'plate_text_fp': 'fastplate',
        'plate_text': 'paddleocr',
        'confidence_x': 'confidence_fp',
        'processing_time_x': 'processing_time_fp',
        'confidence_y': 'confidence_po',
        'processing_time_y': 'processing_time_po'
    }, inplace=True)
    
    # Fill NaN values with empty strings for missing predictions
    merged_all['fastplate'] = merged_all['fastplate'].fillna('')
    merged_all['paddleocr'] = merged_all['paddleocr'].fillna('')
    
    print(f"Merged dataset: {len(merged_all)} records")
    print(f"FastPlate coverage: {(merged_all['fastplate'] != '').sum()}/{len(merged_all)} ({(merged_all['fastplate'] != '').mean()*100:.1f}%)")
    print(f"PaddleOCR coverage: {(merged_all['paddleocr'] != '').sum()}/{len(merged_all)} ({(merged_all['paddleocr'] != '').mean()*100:.1f}%)")
    
    return merged_all

def calculate_metrics(df):
    """Calculate comprehensive metrics for both OCR engines"""
    print("\nCalculating metrics...")
    
    results = {}
    
    for engine in ['fastplate', 'paddleocr']:
        # Exact match accuracy
        exact_matches = (df['ground_truth'] == df[engine]).sum()
        total_predictions = (df[engine] != '').sum()
        total_samples = len(df)
        
        exact_accuracy = exact_matches / total_samples if total_samples > 0 else 0
        prediction_coverage = total_predictions / total_samples if total_samples > 0 else 0
        
        # Character-level accuracy
        char_accuracies = []
        for _, row in df.iterrows():
            if row[engine] != '':  # Only calculate for actual predictions
                char_acc = calculate_character_accuracy(row[engine], row['ground_truth'])
                char_accuracies.append(char_acc)
        
        avg_char_accuracy = np.mean(char_accuracies) if char_accuracies else 0
        
        # Length analysis
        gt_lengths = df['ground_truth'].str.len()
        pred_lengths = df[df[engine] != ''][engine].str.len()
        
        results[engine] = {
            'exact_matches': exact_matches,
            'total_predictions': total_predictions,
            'total_samples': total_samples,
            'exact_accuracy': exact_accuracy,
            'prediction_coverage': prediction_coverage,
            'character_accuracy': avg_char_accuracy,
            'avg_gt_length': gt_lengths.mean(),
            'avg_pred_length': pred_lengths.mean() if len(pred_lengths) > 0 else 0,
            'confidence_mean': df[f'confidence_{engine[:2]}'].mean() if f'confidence_{engine[:2]}' in df.columns else 0,
            'processing_time_mean': df[f'processing_time_{engine[:2]}'].mean() if f'processing_time_{engine[:2]}' in df.columns else 0
        }
    
    return results

def create_confusion_matrices(df):
    """Create character-level confusion matrices for alphanumeric characters"""
    print("\nGenerating confusion matrices...")
    
    # Define alphanumeric characters
    alphanumeric_chars = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    char_to_idx = {char: idx for idx, char in enumerate(alphanumeric_chars)}
    
    def align_strings(s1, s2):
        """Align two strings using edit distance and return character pairs"""
        # Simple alignment: for strings of same length, pair by position
        # For different lengths, use a simple heuristic
        pairs = []
        
        if len(s1) == len(s2):
            # Same length: align by position
            for c1, c2 in zip(s1, s2):
                if c1 in char_to_idx and c2 in char_to_idx:
                    pairs.append((c1, c2))
        else:
            # Different lengths: align common characters first, then handle differences
            # This is a simplified approach - could be improved with proper edit distance alignment
            
            # First, find common characters in same positions
            min_len = min(len(s1), len(s2))
            for i in range(min_len):
                if s1[i] in char_to_idx and s2[i] in char_to_idx:
                    pairs.append((s1[i], s2[i]))
            
            # Handle extra characters (insertions/deletions)
            if len(s1) > len(s2):
                # s1 has extra characters (deletions in s2)
                for i in range(min_len, len(s1)):
                    if s1[i] in char_to_idx:
                        # This character was deleted, we'll skip it for confusion matrix
                        pass
            elif len(s2) > len(s1):
                # s2 has extra characters (insertions)
                for i in range(min_len, len(s2)):
                    if s2[i] in char_to_idx:
                        # This character was inserted, we'll skip it for confusion matrix
                        pass
        
        return pairs
    
    confusion_matrices = {}
    
    for engine in ['fastplate', 'paddleocr']:
        # Collect all character pairs (actual, predicted)
        actual_chars = []
        pred_chars = []
        
        for _, row in df.iterrows():
            if row[engine] != '':  # Only include actual predictions
                gt_text = str(row['ground_truth']).upper()
                pred_text = str(row[engine]).upper()
                
                # Get aligned character pairs
                char_pairs = align_strings(gt_text, pred_text)
                
                for gt_char, pred_char in char_pairs:
                    actual_chars.append(char_to_idx[gt_char])
                    pred_chars.append(char_to_idx[pred_char])
        
        if actual_chars and pred_chars:
            # Create confusion matrix
            cm = confusion_matrix(actual_chars, pred_chars, labels=range(len(alphanumeric_chars)))
            
            confusion_matrices[engine] = {
                'matrix': cm,
                'labels': alphanumeric_chars,
                'matrix_normalized': cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)  # Add small epsilon to avoid division by zero
            }
    
    return confusion_matrices

def plot_confusion_matrices(confusion_matrices, output_dir='data/outputs/analysis'):
    """Plot both normalized and non-normalized confusion matrices"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for engine, cm_data in confusion_matrices.items():
        cm = cm_data['matrix']
        cm_normalized = cm_data['matrix_normalized']
        labels = cm_data['labels']
        
        # Plot non-normalized confusion matrix
        plt.figure(figsize=(16, 14))
        
        # Create annotation matrix - show values only if > 0
        annotations = np.where(cm > 0, cm.astype(str), '')
        
        sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
                    xticklabels=labels, yticklabels=labels,
                    square=True, cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix (Raw Counts) - {engine.title()}', fontsize=16, pad=20)
        plt.xlabel('Predicted Characters', fontsize=14)
        plt.ylabel('Actual Characters', fontsize=14)
        plt.xticks(rotation=0, fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/confusion_matrix_{engine}_raw.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot normalized confusion matrix
        plt.figure(figsize=(16, 14))
        # Replace NaN values (from division by zero) with 0
        cm_normalized_clean = np.nan_to_num(cm_normalized, nan=0.0)
        
        # Create annotation matrix - show values only if > 0
        annotations = np.where(cm_normalized_clean > 0, 
                              np.round(cm_normalized_clean, 2).astype(str), 
                              '')
        
        sns.heatmap(cm_normalized_clean, annot=annotations, fmt='', cmap='Blues',
                    xticklabels=labels, yticklabels=labels,
                    square=True, cbar_kws={'label': 'Normalized Frequency'})
        plt.title(f'Confusion Matrix (Normalized) - {engine.title()}', fontsize=16, pad=20)
        plt.xlabel('Predicted Characters', fontsize=14)
        plt.ylabel('Actual Characters', fontsize=14)
        plt.xticks(rotation=0, fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/confusion_matrix_{engine}_normalized.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a compact view focusing on most confused characters
        # Get characters with highest confusion (off-diagonal elements)
        np.fill_diagonal(cm, 0)  # Remove diagonal for confusion analysis
        confusion_scores = np.sum(cm, axis=1) + np.sum(cm, axis=0)  # Row + column sums
        top_confused_indices = np.argsort(confusion_scores)[-20:]  # Top 20 most confused
        
        if len(top_confused_indices) > 0:
            cm_subset = cm_data['matrix'][np.ix_(top_confused_indices, top_confused_indices)]
            cm_norm_subset = cm_data['matrix_normalized'][np.ix_(top_confused_indices, top_confused_indices)]
            labels_subset = [labels[i] for i in top_confused_indices]
            
            # Plot compact raw confusion matrix
            plt.figure(figsize=(12, 10))
            
            # Create annotation matrix - show values only if > 0
            annotations_subset = np.where(cm_subset > 0, cm_subset.astype(str), '')
            
            sns.heatmap(cm_subset, annot=annotations_subset, fmt='', cmap='Blues',
                        xticklabels=labels_subset, yticklabels=labels_subset,
                        square=True, cbar_kws={'label': 'Count'})
            plt.title(f'Most Confused Characters (Raw) - {engine.title()}', fontsize=14, pad=15)
            plt.xlabel('Predicted Characters', fontsize=12)
            plt.ylabel('Actual Characters', fontsize=12)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/confusion_matrix_{engine}_compact_raw.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot compact normalized confusion matrix
            plt.figure(figsize=(12, 10))
            cm_norm_subset_clean = np.nan_to_num(cm_norm_subset, nan=0.0)
            
            # Create annotation matrix - show values only if > 0
            annotations_norm_subset = np.where(cm_norm_subset_clean > 0, 
                                              np.round(cm_norm_subset_clean, 2).astype(str), 
                                              '')
            
            sns.heatmap(cm_norm_subset_clean, annot=annotations_norm_subset, fmt='', cmap='Blues',
                        xticklabels=labels_subset, yticklabels=labels_subset,
                        square=True, cbar_kws={'label': 'Normalized Frequency'})
            plt.title(f'Most Confused Characters (Normalized) - {engine.title()}', fontsize=14, pad=15)
            plt.xlabel('Predicted Characters', fontsize=12)
            plt.ylabel('Actual Characters', fontsize=12)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/confusion_matrix_{engine}_compact_normalized.png', dpi=300, bbox_inches='tight')
            plt.close()

def analyze_character_performance(confusion_matrices, output_dir='data/outputs/analysis'):
    """Analyze per-character performance metrics"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for engine, cm_data in confusion_matrices.items():
        cm = cm_data['matrix']
        labels = cm_data['labels']
        
        # Calculate per-character metrics
        char_metrics = []
        
        for i, char in enumerate(labels):
            tp = cm[i, i]  # True positives (diagonal)
            fp = np.sum(cm[:, i]) - tp  # False positives (column sum - diagonal)
            fn = np.sum(cm[i, :]) - tp  # False negatives (row sum - diagonal)
            tn = np.sum(cm) - tp - fp - fn  # True negatives
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / np.sum(cm) if np.sum(cm) > 0 else 0
            
            char_metrics.append({
                'character': char,
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'accuracy': accuracy,
                'total_actual': tp + fn,
                'total_predicted': tp + fp
            })
        
        # Save character metrics
        char_df = pd.DataFrame(char_metrics)
        char_df.to_csv(f'{output_dir}/character_metrics_{engine}.csv', index=False)
        
        # Plot character performance
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        # Precision by character
        axes[0, 0].bar(char_df['character'], char_df['precision'], alpha=0.7, color='skyblue')
        axes[0, 0].set_title(f'Precision by Character - {engine.title()}', fontsize=14)
        axes[0, 0].set_ylabel('Precision', fontsize=12)
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Recall by character
        axes[0, 1].bar(char_df['character'], char_df['recall'], alpha=0.7, color='lightcoral')
        axes[0, 1].set_title(f'Recall by Character - {engine.title()}', fontsize=14)
        axes[0, 1].set_ylabel('Recall', fontsize=12)
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # F1 Score by character
        axes[1, 0].bar(char_df['character'], char_df['f1_score'], alpha=0.7, color='lightgreen')
        axes[1, 0].set_title(f'F1 Score by Character - {engine.title()}', fontsize=14)
        axes[1, 0].set_ylabel('F1 Score', fontsize=12)
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Character frequency
        axes[1, 1].bar(char_df['character'], char_df['total_actual'], alpha=0.7, color='gold')
        axes[1, 1].set_title(f'Character Frequency in Ground Truth - {engine.title()}', fontsize=14)
        axes[1, 1].set_ylabel('Frequency', fontsize=12)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/character_performance_{engine}.png', dpi=300, bbox_inches='tight')
        plt.close()

def create_error_analysis(df, output_dir='data/outputs/analysis'):
    """Create detailed error analysis"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("\nPerforming error analysis...")
    
    # Error types analysis
    error_analysis = {}
    
    for engine in ['fastplate', 'paddleocr']:
        errors = []
        
        for _, row in df.iterrows():
            if row[engine] != '' and row['ground_truth'] != row[engine]:
                error_type = classify_error(row['ground_truth'], row[engine])
                errors.append({
                    'image_path': row['image_path'],
                    'ground_truth': row['ground_truth'],
                    'prediction': row[engine],
                    'error_type': error_type,
                    'confidence': row.get(f'confidence_{engine[:2]}', 0)
                })
        
        error_df = pd.DataFrame(errors)
        if not error_df.empty:
            error_df.to_csv(f'{output_dir}/errors_{engine}.csv', index=False)
            
            # Error type distribution
            error_type_counts = error_df['error_type'].value_counts()
            
            plt.figure(figsize=(10, 6))
            error_type_counts.plot(kind='bar')
            plt.title(f'Error Type Distribution - {engine.title()}')
            plt.xlabel('Error Type')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/error_types_{engine}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            error_analysis[engine] = error_type_counts.to_dict()
    
    return error_analysis

def classify_error(ground_truth, prediction):
    """Classify the type of error"""
    if len(prediction) < len(ground_truth):
        return 'Under-segmentation'
    elif len(prediction) > len(ground_truth):
        return 'Over-segmentation'
    elif len(prediction) == len(ground_truth):
        # Character substitution
        diff_chars = sum(1 for a, b in zip(ground_truth, prediction) if a != b)
        if diff_chars == 1:
            return 'Single character error'
        elif diff_chars <= 3:
            return 'Multiple character error'
        else:
            return 'Complete misrecognition'
    else:
        return 'Unknown error'

def create_performance_comparison(results, output_dir='data/outputs/analysis'):
    """Create performance comparison visualizations"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("\nCreating performance comparison...")
    
    # Extract metrics for comparison
    engines = list(results.keys())
    metrics = ['exact_accuracy', 'character_accuracy', 'prediction_coverage']
    
    # Create comparison bar chart
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy comparison
    accuracies = [results[engine]['exact_accuracy'] for engine in engines]
    char_accuracies = [results[engine]['character_accuracy'] for engine in engines]
    
    axes[0, 0].bar(engines, accuracies, alpha=0.7, label='Exact Match')
    axes[0, 0].bar(engines, char_accuracies, alpha=0.7, label='Character Level')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Accuracy Comparison')
    axes[0, 0].legend()
    axes[0, 0].set_ylim(0, 1)
    
    # Coverage comparison
    coverage = [results[engine]['prediction_coverage'] for engine in engines]
    axes[0, 1].bar(engines, coverage, alpha=0.7, color='orange')
    axes[0, 1].set_ylabel('Coverage')
    axes[0, 1].set_title('Prediction Coverage')
    axes[0, 1].set_ylim(0, 1)
    
    # Processing time comparison
    proc_times = [results[engine]['processing_time_mean'] for engine in engines]
    axes[1, 0].bar(engines, proc_times, alpha=0.7, color='green')
    axes[1, 0].set_ylabel('Time (seconds)')
    axes[1, 0].set_title('Average Processing Time')
    
    # Confidence comparison
    confidences = [results[engine]['confidence_mean'] for engine in engines]
    axes[1, 1].bar(engines, confidences, alpha=0.7, color='red')
    axes[1, 1].set_ylabel('Confidence')
    axes[1, 1].set_title('Average Confidence')
    axes[1, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_report(results, error_analysis, output_dir='data/outputs/analysis'):
    """Generate comprehensive analysis report"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    report_path = f'{output_dir}/ocr_analysis_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("OCR ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("SUMMARY METRICS\n")
        f.write("-" * 20 + "\n")
        
        for engine, metrics in results.items():
            f.write(f"\n{engine.upper()}:\n")
            f.write(f"  Exact Match Accuracy: {metrics['exact_accuracy']:.3f} ({metrics['exact_accuracy']*100:.1f}%)\n")
            f.write(f"  Character Accuracy: {metrics['character_accuracy']:.3f} ({metrics['character_accuracy']*100:.1f}%)\n")
            f.write(f"  Prediction Coverage: {metrics['prediction_coverage']:.3f} ({metrics['prediction_coverage']*100:.1f}%)\n")
            f.write(f"  Exact Matches: {metrics['exact_matches']}/{metrics['total_samples']}\n")
            f.write(f"  Average Confidence: {metrics['confidence_mean']:.3f}\n")
            f.write(f"  Average Processing Time: {metrics['processing_time_mean']:.4f}s\n")
            f.write(f"  Average Ground Truth Length: {metrics['avg_gt_length']:.1f}\n")
            f.write(f"  Average Prediction Length: {metrics['avg_pred_length']:.1f}\n")
        
        f.write("\n\nERROR ANALYSIS\n")
        f.write("-" * 20 + "\n")
        
        for engine, errors in error_analysis.items():
            f.write(f"\n{engine.upper()} Error Types:\n")
            for error_type, count in errors.items():
                f.write(f"  {error_type}: {count}\n")
    
    print(f"\nReport saved to: {report_path}")

def main():
    """Main analysis function"""
    print("Starting OCR Analysis...")
    print("=" * 50)
    
    # Load and process data
    gt_df, fp_df, po_df = load_and_process_data()
    
    # Merge datasets
    merged_df = merge_datasets(gt_df, fp_df, po_df)
    
    # Calculate metrics
    results = calculate_metrics(merged_df)
    
    # Print summary
    print("\n" + "=" * 50)
    print("ANALYSIS RESULTS")
    print("=" * 50)
    
    for engine, metrics in results.items():
        print(f"\n{engine.upper()}:")
        print(f"  Exact Match Accuracy: {metrics['exact_accuracy']:.3f} ({metrics['exact_accuracy']*100:.1f}%)")
        print(f"  Character Accuracy: {metrics['character_accuracy']:.3f} ({metrics['character_accuracy']*100:.1f}%)")
        print(f"  Prediction Coverage: {metrics['prediction_coverage']:.3f} ({metrics['prediction_coverage']*100:.1f}%)")
        print(f"  Average Confidence: {metrics['confidence_mean']:.3f}")
        print(f"  Average Processing Time: {metrics['processing_time_mean']:.4f}s")
    
    # Create visualizations
    output_dir = 'data/outputs/analysis'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate confusion matrices
    confusion_matrices = create_confusion_matrices(merged_df)
    plot_confusion_matrices(confusion_matrices, output_dir)
    
    # Analyze character performance
    analyze_character_performance(confusion_matrices, output_dir)
    
    # Error analysis
    error_analysis = create_error_analysis(merged_df, output_dir)
    
    # Performance comparison
    create_performance_comparison(results, output_dir)
    
    # Generate comprehensive report
    generate_report(results, error_analysis, output_dir)
    
    # Save merged dataset
    merged_df.to_csv(f'{output_dir}/merged_results.csv', index=False)
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    print(f"Key files generated:")
    print(f"  - merged_results.csv: Complete dataset with all predictions")
    print(f"  - ocr_analysis_report.txt: Comprehensive analysis report")
    print(f"  - confusion_matrix_*_raw.png: Raw count confusion matrices (A-Z, 0-9)")
    print(f"  - confusion_matrix_*_normalized.png: Normalized confusion matrices (A-Z, 0-9)")
    print(f"  - confusion_matrix_*_compact_*.png: Compact view of most confused characters")
    print(f"  - character_metrics_*.csv: Per-character performance metrics")
    print(f"  - character_performance_*.png: Character-level precision/recall/F1 charts")
    print(f"  - error_types_*.png: Error type distributions")
    print(f"  - performance_comparison.png: Side-by-side performance comparison")
    print(f"  - errors_*.csv: Detailed error analysis files")

if __name__ == "__main__":
    main() 