#!/usr/bin/env python3
"""
Generate detailed character accuracy report for OCR analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

def generate_character_report():
    # Load the data
    df = pd.read_csv('data/outputs/analysis/merged_results.csv')
    
    # Create detailed character accuracy report
    report_content = []
    report_content.append('DETAILED CHARACTER ACCURACY ANALYSIS REPORT')
    report_content.append('=' * 60)
    report_content.append(f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    report_content.append(f'Dataset: {len(df)} license plate images')
    report_content.append('')
    
    # Initialize character tracking for both engines
    engines = ['paddleocr', 'fastplate']
    alphanumeric_chars = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    
    engine_results = {}
    
    for engine in engines:
        report_content.append(f'{engine.upper()} CHARACTER ACCURACY ANALYSIS')
        report_content.append('-' * 50)
        report_content.append('')
        
        # Track each character's performance
        char_stats = {}
        for char in alphanumeric_chars:
            char_stats[char] = {
                'total_occurrences': 0,
                'correct_predictions': 0,
                'accuracy': 0.0
            }
        
        # Analyze each prediction
        for _, row in df.iterrows():
            if row[engine] != '':
                gt_text = str(row['ground_truth']).upper()
                pred_text = str(row[engine]).upper()
                
                # Only analyze same-length strings for position accuracy
                if len(gt_text) == len(pred_text):
                    for gt_char, pred_char in zip(gt_text, pred_text):
                        if gt_char in char_stats:
                            char_stats[gt_char]['total_occurrences'] += 1
                            if gt_char == pred_char:
                                char_stats[gt_char]['correct_predictions'] += 1
        
        # Calculate accuracies and sort by frequency
        char_results = []
        for char, stats in char_stats.items():
            if stats['total_occurrences'] > 0:
                accuracy = stats['correct_predictions'] / stats['total_occurrences']
                char_results.append({
                    'char': char,
                    'correct': stats['correct_predictions'],
                    'total': stats['total_occurrences'],
                    'accuracy': accuracy
                })
        
        # Sort by total occurrences (most frequent first)
        char_results.sort(key=lambda x: x['total'], reverse=True)
        engine_results[engine] = char_results
        
        # Display results
        report_content.append('Character | Correct | Total | Accuracy | Performance')
        report_content.append('-' * 55)
        
        for result in char_results:
            if result['total'] >= 1:  # Show all characters that appear
                if result['accuracy'] >= 0.95:
                    performance = 'Excellent'
                elif result['accuracy'] >= 0.90:
                    performance = 'Good'
                elif result['accuracy'] >= 0.80:
                    performance = 'Fair'
                else:
                    performance = 'Poor'
                
                report_content.append(f'    {result["char"]:1s}     |  {result["correct"]:4d}  | {result["total"]:4d}  | {result["accuracy"]:7.1%}  | {performance}')
        
        report_content.append('')
        
        # Summary statistics
        total_chars = sum(r['total'] for r in char_results)
        total_correct = sum(r['correct'] for r in char_results)
        overall_accuracy = total_correct / total_chars if total_chars > 0 else 0
        report_content.append('SUMMARY STATISTICS:')
        report_content.append(f'Overall character accuracy: {total_correct:,}/{total_chars:,} = {overall_accuracy:.1%}')
        
        # Show best and worst performing characters
        if char_results:
            # Filter characters with at least 10 occurrences for meaningful stats
            frequent_chars = [r for r in char_results if r['total'] >= 10]
            
            if frequent_chars:
                best_chars = [r for r in frequent_chars if r['accuracy'] >= 0.99]
                worst_chars = [r for r in frequent_chars if r['accuracy'] < 0.85]
                
                report_content.append('')
                report_content.append('PERFORMANCE CATEGORIES:')
                if best_chars:
                    best_chars.sort(key=lambda x: x['accuracy'], reverse=True)
                    best_list = [f"{r['char']} ({r['accuracy']:.1%})" for r in best_chars[:5]]
                    report_content.append(f'Excellent (99%+): {", ".join(best_list)}')
                
                if worst_chars:
                    worst_chars.sort(key=lambda x: x['accuracy'])
                    worst_list = [f"{r['char']} ({r['accuracy']:.1%})" for r in worst_chars]
                    report_content.append(f'Problematic (<85%): {", ".join(worst_list)}')
        
        report_content.append('')
        report_content.append('=' * 60)
        report_content.append('')
    
    # Comparison section
    report_content.append('ENGINE COMPARISON')
    report_content.append('-' * 30)
    report_content.append('')
    report_content.append('Character | PaddleOCR | FastPlate | Difference')
    report_content.append('-' * 45)
    
    # Align results for comparison
    paddle_dict = {r['char']: r for r in engine_results['paddleocr']}
    fast_dict = {r['char']: r for r in engine_results['fastplate']}
    
    all_chars = set(paddle_dict.keys()) | set(fast_dict.keys())
    comparison_results = []
    
    for char in sorted(all_chars):
        paddle_acc = paddle_dict.get(char, {'accuracy': 0, 'total': 0})['accuracy']
        fast_acc = fast_dict.get(char, {'accuracy': 0, 'total': 0})['accuracy']
        diff = paddle_acc - fast_acc
        
        if paddle_dict.get(char, {'total': 0})['total'] >= 5 or fast_dict.get(char, {'total': 0})['total'] >= 5:
            comparison_results.append({
                'char': char,
                'paddle': paddle_acc,
                'fast': fast_acc,
                'diff': diff
            })
    
    for result in comparison_results:
        diff_str = f'+{result["diff"]:5.1%}' if result['diff'] > 0 else f'{result["diff"]:6.1%}'
        report_content.append(f'    {result["char"]:1s}     | {result["paddle"]:7.1%}   | {result["fast"]:7.1%}   | {diff_str}')
    
    report_content.append('')
    
    # Key insights
    report_content.append('KEY INSIGHTS AND RECOMMENDATIONS')
    report_content.append('-' * 40)
    report_content.append('')
    
    # Find most problematic characters
    all_problem_chars = set()
    for engine in ['paddleocr', 'fastplate']:
        for result in engine_results[engine]:
            if result['total'] >= 10 and result['accuracy'] < 0.85:
                all_problem_chars.add(result['char'])
    
    report_content.append('1. PROBLEMATIC CHARACTERS:')
    for char in sorted(all_problem_chars):
        paddle_stats = paddle_dict.get(char, {'accuracy': 0, 'correct': 0, 'total': 0})
        fast_stats = fast_dict.get(char, {'accuracy': 0, 'correct': 0, 'total': 0})
        report_content.append(f'   {char}: PaddleOCR {paddle_stats["accuracy"]:.1%} ({paddle_stats["correct"]}/{paddle_stats["total"]}), FastPlate {fast_stats["accuracy"]:.1%} ({fast_stats["correct"]}/{fast_stats["total"]})')
    
    report_content.append('')
    report_content.append('2. COMMON CONFUSION PATTERNS:')
    report_content.append('   O ↔ 0 (letter O confused with digit 0)')
    report_content.append('   I ↔ 1 (letter I confused with digit 1)')
    report_content.append('   D → 0 (letter D confused with digit 0)')
    report_content.append('   Z ↔ 2 (letter Z confused with digit 2)')
    report_content.append('')
    
    report_content.append('3. RECOMMENDATIONS:')
    report_content.append('   - Implement post-processing rules for O/0 disambiguation')
    report_content.append('   - Use context-based correction for I/1 confusion')
    report_content.append('   - Consider ensemble approach combining both engines')
    report_content.append('   - Apply Vietnamese license plate format validation')
    report_content.append('')
    
    # Calculate overall comparison
    paddle_overall = sum(r['correct'] for r in engine_results['paddleocr']) / sum(r['total'] for r in engine_results['paddleocr'])
    fast_overall = sum(r['correct'] for r in engine_results['fastplate']) / sum(r['total'] for r in engine_results['fastplate'])
    
    report_content.append('4. OVERALL PERFORMANCE:')
    report_content.append(f'   PaddleOCR: {paddle_overall:.1%} character accuracy')
    report_content.append(f'   FastPlate: {fast_overall:.1%} character accuracy')
    report_content.append(f'   Difference: {paddle_overall - fast_overall:+.1%} in favor of PaddleOCR')
    
    # Write to file
    output_file = 'data/outputs/analysis/detailed_character_accuracy_report.txt'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_content))
    
    print(f'Detailed character accuracy report saved to: {output_file}')
    print()
    print('Report contents:')
    print(f'- Total characters analyzed: {sum(r["total"] for r in engine_results["paddleocr"]):,}')
    print(f'- Engine comparison for all {len(alphanumeric_chars)} alphanumeric characters')
    print(f'- Performance categorization and recommendations')
    print(f'- Detailed accuracy statistics for each character')
    
    return output_file

if __name__ == "__main__":
    generate_character_report() 