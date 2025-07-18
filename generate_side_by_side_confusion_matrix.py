#!/usr/bin/env python3
"""
Generate side-by-side confusion matrices for OCR comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path

def create_side_by_side_confusion_matrices():
    """Create side-by-side confusion matrices for both OCR engines"""
    print("Generating side-by-side confusion matrices...")
    
    # Load data
    df = pd.read_csv('data/outputs/analysis/merged_results.csv')
    
    # Define alphanumeric characters
    alphanumeric_chars = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    char_to_idx = {char: idx for idx, char in enumerate(alphanumeric_chars)}
    
    def align_strings(s1, s2):
        """Simple character alignment for same-length strings"""
        pairs = []
        if len(s1) == len(s2):
            for c1, c2 in zip(s1, s2):
                if c1 in char_to_idx and c2 in char_to_idx:
                    pairs.append((c1, c2))
        return pairs
    
    # Collect character pairs for both engines
    engines = ['fastplate', 'paddleocr']
    confusion_data = {}
    
    for engine in engines:
        actual_chars = []
        pred_chars = []
        
        for _, row in df.iterrows():
            if row[engine] != '':
                gt_text = str(row['ground_truth']).upper()
                pred_text = str(row[engine]).upper()
                
                char_pairs = align_strings(gt_text, pred_text)
                for gt_char, pred_char in char_pairs:
                    actual_chars.append(char_to_idx[gt_char])
                    pred_chars.append(char_to_idx[pred_char])
        
        if actual_chars and pred_chars:
            cm = confusion_matrix(actual_chars, pred_chars, labels=range(len(alphanumeric_chars)))
            cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
            
            confusion_data[engine] = {
                'matrix': cm,
                'matrix_normalized': cm_normalized,
                'labels': alphanumeric_chars
            }
    
    # Create output directory
    output_dir = 'data/outputs/analysis'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate side-by-side plots
    create_side_by_side_raw_matrices(confusion_data, output_dir)
    create_side_by_side_normalized_matrices(confusion_data, output_dir)
    create_focused_comparison_matrices(confusion_data, output_dir)
    
    print(f"Side-by-side confusion matrices saved to: {output_dir}")

def create_side_by_side_normalized_matrices(confusion_data, output_dir):
    """Create side-by-side normalized confusion matrices with shared labels and colorbar"""
    
    # =============================================================================
    # SPACING TUNING PARAMETERS - OPTIMIZED FOR 5760x3240:
    # =============================================================================
    FIGURE_WIDTH = 19.2        # 5760 pixels at 300 DPI
    FIGURE_HEIGHT = 10.8       # 3240 pixels at 300 DPI
    COLORBAR_LEFT = 0.82       # Colorbar position from left (0.85 = closer, 0.90 = further)
    COLORBAR_WIDTH = 0.02      # Colorbar width (0.02 = thinner, 0.04 = thicker)
    COLORBAR_HEIGHT = 0.6      # Colorbar height (0.4 = shorter, 0.7 = taller)
    RIGHT_MARGIN = 0.80        # Right margin (0.80 = tighter, 0.90 = more space)
    MATRIX_SPACING = 0.07      # Space between matrices (0.02 = closer, 0.10 = further apart)
    TITLE_HEIGHT = 0.93        # Main title position (0.90 = closer to matrices, 0.95 = further)
    MATRIX_TITLE_PAD = 2       # Padding between matrix title and matrix (smaller = closer)
    # =============================================================================
    
    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    
    engines = ['fastplate', 'paddleocr']
    titles = ['FastPlate OCR', 'PaddleOCR v5']
    
    def format_value(val):
        """Format values: show .XX for values < 1, omit values < 0.005 (0.5%)"""
        if val < 0.005:  # Lowered threshold to show more small values
            return ''
        elif val < 1.0:
            return f'.{int(val * 100):02d}'  # .XX format
        else:
            return '1.0'  # Should be 1.00 on diagonal
    
    # Create the heatmaps without individual colorbars
    heatmap_objects = []
    
    for i, engine in enumerate(engines):
        if engine in confusion_data:
            cm_normalized = confusion_data[engine]['matrix_normalized']
            labels = confusion_data[engine]['labels']
            
            # Clean NaN values
            cm_normalized_clean = np.nan_to_num(cm_normalized, nan=0.0)
            
            # Create custom annotation matrix with .XX format
            annotations = np.vectorize(format_value)(cm_normalized_clean)
            
            # Create heatmap without colorbar - show y-labels for both
            hm = sns.heatmap(cm_normalized_clean, annot=annotations, fmt='', cmap='Blues',
                       xticklabels=labels, 
                       yticklabels=labels,  # Show y-labels for both matrices
                       square=True, cbar=False,  # Disable individual colorbar
                       vmin=0, vmax=1, ax=axes[i], annot_kws={'fontsize': 6})  # Smaller font for 1080p
            
            heatmap_objects.append(hm)
            
            axes[i].set_title(f'{titles[i]}', fontsize=14, pad=MATRIX_TITLE_PAD)  # Closer title padding
            axes[i].set_xlabel('Predicted Characters', fontsize=11)  # Optimized for 1080p
            axes[i].set_ylabel('Actual Characters (Ground Truth)', fontsize=11)
            axes[i].tick_params(axis='x', rotation=0, labelsize=8)  # Smaller ticks for 1080p
            axes[i].tick_params(axis='y', rotation=0, labelsize=8)
    
    # Add a single shared colorbar with tighter positioning
    if heatmap_objects:
        # Create colorbar very close to the matrices
        cbar_ax = fig.add_axes([COLORBAR_LEFT, 0.15, COLORBAR_WIDTH, COLORBAR_HEIGHT])
        cbar = fig.colorbar(heatmap_objects[0].collections[0], cax=cbar_ax)
        cbar.set_label('Normalized Frequency', fontsize=11, rotation=270, labelpad=15)
        cbar.ax.tick_params(labelsize=8)  # Smaller colorbar labels
    
    # Optimized layout for 1920x1080
    plt.subplots_adjust(
        top=0.85,           # Tighter top margin for closer title
        bottom=0.10,        # Bottom margin 
        left=0.08,          # Left margin
        right=RIGHT_MARGIN, # Right margin (tunable parameter above)
        wspace=MATRIX_SPACING  # Space between subplots (tunable parameter above)
    )
    
    plt.savefig(f'{output_dir}/confusion_matrix_side_by_side_normalized.png', dpi=300, bbox_inches='tight')  # 300 DPI for exact 5760x3240
    plt.close()

def create_side_by_side_raw_matrices(confusion_data, output_dir):
    """Create side-by-side raw count confusion matrices with shared labels and colorbar"""
    print("Creating side-by-side raw count confusion matrices...")
    
    # =============================================================================
    # SPACING TUNING PARAMETERS - OPTIMIZED FOR 5760x3240:
    # =============================================================================
    FIGURE_WIDTH = 19.2        # 5760 pixels at 300 DPI
    FIGURE_HEIGHT = 10.8       # 3240 pixels at 300 DPI
    COLORBAR_LEFT = 0.82       # Colorbar position from left (0.85 = closer, 0.90 = further)
    COLORBAR_WIDTH = 0.02      # Colorbar width (0.02 = thinner, 0.04 = thicker)
    COLORBAR_HEIGHT = 0.6      # Colorbar height (0.4 = shorter, 0.7 = taller)
    RIGHT_MARGIN = 0.80        # Right margin (0.80 = tighter, 0.90 = more space)
    MATRIX_SPACING = 0.05      # Space between matrices (0.02 = closer, 0.10 = further apart)
    TITLE_HEIGHT = 0.93        # Main title position (0.90 = closer to matrices, 0.95 = further)
    MATRIX_TITLE_PAD = 2       # Padding between matrix title and matrix (smaller = closer)
    # =============================================================================
    
    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    
    engines = ['fastplate', 'paddleocr']
    titles = ['FastPlate OCR', 'PaddleOCR v5']
    
    # Find the maximum value across both matrices for consistent scaling
    max_val = 0
    for engine in engines:
        if engine in confusion_data:
            cm = confusion_data[engine]['matrix']
            max_val = max(max_val, np.max(cm))
    
    # Create the heatmaps without individual colorbars
    heatmap_objects = []
    
    for i, engine in enumerate(engines):
        if engine in confusion_data:
            cm = confusion_data[engine]['matrix']
            labels = confusion_data[engine]['labels']
            
            # Create annotation matrix - hide zeros
            annotations = np.where(cm > 0, cm.astype(str), '')
            
            # Create heatmap without colorbar - show y-labels for both
            hm = sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
                       xticklabels=labels,
                       yticklabels=labels,  # Show y-labels for both matrices
                       square=True, cbar=False,  # Disable individual colorbar
                       vmin=0, vmax=max_val, ax=axes[i], annot_kws={'fontsize': 6})  # Smaller font for 1080p
            
            heatmap_objects.append(hm)
            
            axes[i].set_title(f'{titles[i]}', fontsize=14, pad=MATRIX_TITLE_PAD)  # Closer title padding
            axes[i].set_xlabel('Predicted Characters', fontsize=11)  # Optimized for 1080p
            axes[i].set_ylabel('Actual Characters (Ground Truth)', fontsize=11)
            axes[i].tick_params(axis='x', rotation=0, labelsize=8)  # Smaller ticks for 1080p
            axes[i].tick_params(axis='y', rotation=0, labelsize=8)
    
    # Add a single shared colorbar with tighter positioning
    if heatmap_objects:
        # Create colorbar very close to the matrices
        cbar_ax = fig.add_axes([COLORBAR_LEFT, 0.15, COLORBAR_WIDTH, COLORBAR_HEIGHT])
        cbar = fig.colorbar(heatmap_objects[0].collections[0], cax=cbar_ax)
        cbar.set_label('Count', fontsize=11, rotation=270, labelpad=15)
        cbar.ax.tick_params(labelsize=8)  # Smaller colorbar labels
    
    # Optimized layout for 1920x1080
    plt.subplots_adjust(
        top=0.85,           # Tighter top margin for closer title
        bottom=0.10,        # Bottom margin 
        left=0.08,          # Left margin
        right=RIGHT_MARGIN, # Right margin (tunable parameter above)
        wspace=MATRIX_SPACING  # Space between subplots (tunable parameter above)
    )
    
    plt.savefig(f'{output_dir}/confusion_matrix_side_by_side_raw.png', dpi=300, bbox_inches='tight')  # 300 DPI for exact 5760x3240
    plt.close()

def create_focused_comparison_matrices(confusion_data, output_dir):
    """Create focused matrices showing only problematic characters with shared labels and colorbar"""
    
    # =============================================================================
    # SPACING TUNING PARAMETERS - OPTIMIZED FOR 5760x3240:
    # =============================================================================
    FIGURE_WIDTH = 19.2        # 5760 pixels at 300 DPI
    FIGURE_HEIGHT = 10.8       # 3240 pixels at 300 DPI (could be smaller since fewer characters)
    COLORBAR_LEFT = 0.82       # Colorbar position from left (0.85 = closer, 0.90 = further)
    COLORBAR_WIDTH = 0.02      # Colorbar width (0.02 = thinner, 0.04 = thicker)
    COLORBAR_HEIGHT = 0.4      # Colorbar height (0.4 = shorter, 0.6 = taller)
    RIGHT_MARGIN = 0.80        # Right margin (0.80 = tighter, 0.90 = more space)
    MATRIX_SPACING = 0.05      # Space between matrices (0.02 = closer, 0.10 = further apart)
    TITLE_HEIGHT = 0.93        # Main title position (0.90 = closer to matrices, 0.95 = further)
    MATRIX_TITLE_PAD = 2       # Padding between matrix title and matrix (smaller = closer)
    # =============================================================================
    
    # Find characters with significant confusions in either engine
    problematic_chars = set()
    
    for engine_data in confusion_data.values():
        cm = engine_data['matrix']
        labels = engine_data['labels']
        
        # Find characters with off-diagonal elements > 3 (lowered threshold)
        for i in range(len(labels)):
            row_sum = np.sum(cm[i, :]) - cm[i, i]  # Non-diagonal sum
            col_sum = np.sum(cm[:, i]) - cm[i, i]  # Non-diagonal sum
            if row_sum > 3 or col_sum > 3:  # Lowered from 5 to 3
                problematic_chars.add(i)
    
    if len(problematic_chars) > 25:  # Limit to most problematic
        # Sort by total confusion count and take top 20
        confusion_scores = []
        for i in problematic_chars:
            total_confusion = 0
            for engine_data in confusion_data.values():
                cm = engine_data['matrix']
                total_confusion += np.sum(cm[i, :]) - cm[i, i] + np.sum(cm[:, i]) - cm[i, i]
            confusion_scores.append((i, total_confusion))
        
        confusion_scores.sort(key=lambda x: x[1], reverse=True)
        problematic_chars = [idx for idx, _ in confusion_scores[:20]]
    else:
        problematic_chars = sorted(list(problematic_chars))
    
    if problematic_chars:
        # Create focused matrices
        fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
        
        engines = ['fastplate', 'paddleocr']
        titles = ['FastPlate OCR', 'PaddleOCR v5']
        
        def format_value(val):
            """Format values: show .XX for values < 1, omit values < 0.005 (0.5%)"""
            if val < 0.005:  # Lowered threshold to show more small values
                return ''
            elif val < 1.0:
                return f'.{int(val * 100):02d}'  # .XX format
            else:
                return '1.0'
        
        # Create the heatmaps without individual colorbars
        heatmap_objects = []
        
        for i, engine in enumerate(engines):
            if engine in confusion_data:
                cm = confusion_data[engine]['matrix']
                cm_normalized = confusion_data[engine]['matrix_normalized']
                labels = confusion_data[engine]['labels']
                
                # Extract submatrix for problematic characters
                cm_subset = cm[np.ix_(problematic_chars, problematic_chars)]
                cm_norm_subset = cm_normalized[np.ix_(problematic_chars, problematic_chars)]
                labels_subset = [labels[idx] for idx in problematic_chars]
                
                # Clean normalized values
                cm_norm_subset_clean = np.nan_to_num(cm_norm_subset, nan=0.0)
                
                # Create custom annotations with .XX format
                annotations = np.vectorize(format_value)(cm_norm_subset_clean)
                
                # Create heatmap without colorbar - show y-labels for both
                hm = sns.heatmap(cm_norm_subset_clean, annot=annotations, fmt='', cmap='Blues',
                           xticklabels=labels_subset,
                           yticklabels=labels_subset,  # Show y-labels for both matrices
                           square=True, cbar=False,  # Disable individual colorbar
                           vmin=0, vmax=1, ax=axes[i], annot_kws={'fontsize': 8})  # Larger font since fewer characters
                
                heatmap_objects.append(hm)
                
                axes[i].set_title(f'{titles[i]}', fontsize=12, pad=MATRIX_TITLE_PAD)  # Closer title
                axes[i].set_xlabel('Predicted Characters', fontsize=10)
                axes[i].set_ylabel('Actual Characters (Ground Truth)', fontsize=10)
                axes[i].tick_params(axis='x', rotation=45, labelsize=9)
                axes[i].tick_params(axis='y', rotation=0, labelsize=9)
        
        # Add a single shared colorbar with tighter positioning
        if heatmap_objects:
            # Create colorbar very close to the matrices
            cbar_ax = fig.add_axes([COLORBAR_LEFT, 0.15, COLORBAR_WIDTH, COLORBAR_HEIGHT])
            cbar = fig.colorbar(heatmap_objects[0].collections[0], cax=cbar_ax)
            cbar.set_label('Normalized Frequency', fontsize=10, rotation=270, labelpad=12)
            cbar.ax.tick_params(labelsize=8)  # Smaller colorbar labels
        
        # Optimized layout for 1920x1080
        plt.subplots_adjust(
            top=0.85,           # Tighter top margin for closer title
            bottom=0.12,        # Bottom margin (more space for rotated labels)
            left=0.10,          # Left margin
            right=RIGHT_MARGIN, # Right margin (tunable parameter above)
            wspace=MATRIX_SPACING  # Space between subplots (tunable parameter above)
        )
        
        plt.savefig(f'{output_dir}/confusion_matrix_side_by_side_focused.png', dpi=300, bbox_inches='tight')  # 300 DPI for 5760x3240
        plt.close()

def create_difference_matrix(confusion_data, output_dir):
    """Create a difference matrix showing PaddleOCR - FastPlate performance"""
    if 'paddleocr' in confusion_data and 'fastplate' in confusion_data:
        paddle_norm = confusion_data['paddleocr']['matrix_normalized']
        fast_norm = confusion_data['fastplate']['matrix_normalized']
        labels = confusion_data['paddleocr']['labels']
        
        # Calculate difference (PaddleOCR - FastPlate)
        diff_matrix = np.nan_to_num(paddle_norm, nan=0.0) - np.nan_to_num(fast_norm, nan=0.0)
        
        def format_diff_value(val):
            """Format difference values with .XX notation"""
            if abs(val) < 0.005:  # Less than 0.5%
                return ''
            elif val > 0:
                if val < 1.0:
                    return f'+.{int(val * 100):02d}'
                else:
                    return f'+{val:.2f}'
            else:
                if val > -1.0:
                    return f'-.{int(abs(val) * 100):02d}'
                else:
                    return f'{val:.2f}'
        
        # Create annotations for significant differences
        annotations = np.vectorize(format_diff_value)(diff_matrix)
        
        plt.figure(figsize=(16, 14))
        sns.heatmap(diff_matrix, annot=annotations, fmt='', cmap='RdBu_r',
                   xticklabels=labels, yticklabels=labels,
                   square=True, center=0,
                   cbar_kws={'label': 'Difference (PaddleOCR - FastPlate)'})
        
        plt.title('Performance Difference Matrix\n(PaddleOCR - FastPlate)', fontsize=16, pad=20)
        plt.xlabel('Predicted Characters', fontsize=14)
        plt.ylabel('Actual Characters (Ground Truth)', fontsize=14)
        plt.xticks(rotation=0, fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/confusion_matrix_difference.png', dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    create_side_by_side_confusion_matrices()
    
    # Also create difference matrix
    print("Creating difference matrix...")
    df = pd.read_csv('data/outputs/analysis/merged_results.csv')
    alphanumeric_chars = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    char_to_idx = {char: idx for idx, char in enumerate(alphanumeric_chars)}
    
    confusion_data = {}
    for engine in ['fastplate', 'paddleocr']:
        actual_chars = []
        pred_chars = []
        
        for _, row in df.iterrows():
            if row[engine] != '':
                gt_text = str(row['ground_truth']).upper()
                pred_text = str(row[engine]).upper()
                
                if len(gt_text) == len(pred_text):
                    for c1, c2 in zip(gt_text, pred_text):
                        if c1 in char_to_idx and c2 in char_to_idx:
                            actual_chars.append(char_to_idx[c1])
                            pred_chars.append(char_to_idx[c2])
        
        if actual_chars and pred_chars:
            cm = confusion_matrix(actual_chars, pred_chars, labels=range(len(alphanumeric_chars)))
            cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
            confusion_data[engine] = {
                'matrix_normalized': cm_normalized,
                'labels': alphanumeric_chars
            }
    
    create_difference_matrix(confusion_data, 'data/outputs/analysis')
    print("All side-by-side confusion matrices generated successfully!") 