"""
Generate comprehensive comparison tables for thesis with detailed model specifications,
performance metrics, export format analysis, and efficiency calculations.
"""

import pandas as pd
import os
import sys

# Add the tools directory to path to import model_specifications
sys.path.append(os.path.join(os.path.dirname(__file__)))
from model_specifications import (
    get_model_specs_df, get_export_format_specs_df, get_hardware_specs_df,
    EXPORT_FORMAT_SPECS
)

def parse_benchmark_log(log_path):
    """Parses a single benchmark log file and returns a DataFrame."""
    import re
    
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    data = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Skip header and legend lines
        if any(keyword in line for keyword in ['Benchmarks complete', 'Benchmarks legend', 'Format Status', '---']):
            continue
            
        # Look for data rows that start with a number
        match = re.match(r'^\s*(\d+)\s+(.+?)\s+(✅|❌|❎)\s+([\d.]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)', line)
        if match:
            format_name = match.group(2).strip()
            status = match.group(3)
            size = float(match.group(4)) if match.group(4) != '-' else 0.0
            mAP = float(match.group(5)) if match.group(5) != '-' else 0.0
            latency = float(match.group(6)) if match.group(6) != '-' else 0.0
            fps = float(match.group(7)) if match.group(7) != '-' else 0.0
            data.append([format_name, status, size, mAP, latency, fps])
    
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=['Format', 'Status', 'Size_MB', 'mAP50-95', 'Latency_ms_im', 'FPS'])
    return df

def generate_comprehensive_model_table():
    """Generate a comprehensive model comparison table."""
    
    # Load model specifications
    model_specs = get_model_specs_df()
    
    # Load performance data for both detection tasks
    detection_types = ["License Plate", "Vehicles"]
    models = ["YOLO11n", "YOLOv10n", "YOLOv5u", "YOLOv8n"]
    
    all_performance_data = []
    
    for det_type in detection_types:
        for model in models:
            log_path = os.path.join("traffic-monitor-resources/Results", det_type, model, "benchmarks.log")
            if os.path.exists(log_path):
                df = parse_benchmark_log(log_path)
                if not df.empty:
                    # Filter for PyTorch format
                    pytorch_data = df[df['Format'] == 'PyTorch']
                    if not pytorch_data.empty:
                        perf_row = pytorch_data.iloc[0].copy()
                        perf_row['Model'] = model
                        perf_row['Detection_Task'] = det_type
                        all_performance_data.append(perf_row)
    
    performance_df = pd.DataFrame(all_performance_data)
    
    # Merge with model specifications
    comprehensive_data = []
    
    for _, spec_row in model_specs.iterrows():
        model_name = spec_row['Model']
        
        # Get performance for both tasks
        vehicle_perf = performance_df[
            (performance_df['Model'] == model_name) & 
            (performance_df['Detection_Task'] == 'Vehicles')
        ]
        plate_perf = performance_df[
            (performance_df['Model'] == model_name) & 
            (performance_df['Detection_Task'] == 'License Plate')
        ]
        
        # Create comprehensive row
        comp_row = {
            'Model': model_name,
            'Release_Year': spec_row['release_year'],
            'Parameters_M': spec_row['parameters_millions'],
            'GFLOPs': spec_row['gflops_640'],
            'Architecture_Type': spec_row['architecture_type'],
            'Backbone': spec_row['backbone'],
            'Neck': spec_row['neck'],
            'Head': spec_row['head'],
            'Anchor_Based': spec_row['anchor_based'],
            'Key_Features': spec_row['key_features'],
        }
        
        # Add vehicle detection performance
        if not vehicle_perf.empty:
            vp = vehicle_perf.iloc[0]
            comp_row.update({
                'Vehicle_mAP50_95': vp['mAP50-95'],
                'Vehicle_Latency_ms': vp['Latency_ms_im'],
                'Vehicle_FPS': vp['FPS'],
                'Vehicle_Size_MB': vp['Size_MB'],
            })
        
        # Add plate detection performance
        if not plate_perf.empty:
            pp = plate_perf.iloc[0]
            comp_row.update({
                'Plate_mAP50_95': pp['mAP50-95'],
                'Plate_Latency_ms': pp['Latency_ms_im'],
                'Plate_FPS': pp['FPS'],
                'Plate_Size_MB': pp['Size_MB'],
            })
        
        # Calculate efficiency metrics
        if not vehicle_perf.empty:
            vp = vehicle_perf.iloc[0]
            comp_row.update({
                'Vehicle_FPS_per_GFLOP': vp['FPS'] / spec_row['gflops_640'],
                'Vehicle_mAP_per_GFLOP': vp['mAP50-95'] / spec_row['gflops_640'],
                'Vehicle_FPS_per_Param': vp['FPS'] / spec_row['parameters_millions'],
            })
        
        if not plate_perf.empty:
            pp = plate_perf.iloc[0]
            comp_row.update({
                'Plate_FPS_per_GFLOP': pp['FPS'] / spec_row['gflops_640'],
                'Plate_mAP_per_GFLOP': pp['mAP50-95'] / spec_row['gflops_640'],
                'Plate_FPS_per_Param': pp['FPS'] / spec_row['parameters_millions'],
            })
        
        comprehensive_data.append(comp_row)
    
    comprehensive_df = pd.DataFrame(comprehensive_data)
    return comprehensive_df.round(3)

def generate_export_format_analysis():
    """Generate export format performance analysis."""
    
    detection_types = ["License Plate", "Vehicles"]
    models = ["YOLO11n", "YOLOv10n", "YOLOv5u", "YOLOv8n"]
    
    format_analysis_data = []
    
    for det_type in detection_types:
        for model in models:
            log_path = os.path.join("traffic-monitor-resources/Results", det_type, model, "benchmarks.log")
            if os.path.exists(log_path):
                df = parse_benchmark_log(log_path)
                if not df.empty:
                    # Filter for successful exports and desired formats
                    desired_formats = ['PyTorch', 'ONNX', 'TensorRT']
                    successful_df = df[
                        (df['Status'] == '✅') & 
                        (df['Format'].isin(desired_formats))
                    ]
                    
                    for _, row in successful_df.iterrows():
                        format_specs = EXPORT_FORMAT_SPECS.get(row['Format'], {})
                        
                        analysis_row = {
                            'Model': model,
                            'Detection_Task': det_type,
                            'Export_Format': row['Format'],
                            'Size_MB': row['Size_MB'],
                            'mAP50_95': row['mAP50-95'],
                            'Latency_ms': row['Latency_ms_im'],
                            'FPS': row['FPS'],
                            'Precision': format_specs.get('precision', 'Unknown'),
                            'Optimization_Level': format_specs.get('optimization_level', 'Unknown'),
                            'Hardware_Requirements': format_specs.get('hardware_requirements', 'Unknown'),
                            'Deployment_Complexity': format_specs.get('deployment_complexity', 'Unknown'),
                            'Inference_Backend': format_specs.get('inference_backend', 'Unknown'),
                        }
                        
                        # Calculate speedup vs PyTorch
                        pytorch_row = df[df['Format'] == 'PyTorch']
                        if not pytorch_row.empty:
                            pytorch_fps = pytorch_row.iloc[0]['FPS']
                            pytorch_latency = pytorch_row.iloc[0]['Latency_ms_im']
                            analysis_row['FPS_Speedup_vs_PyTorch'] = row['FPS'] / pytorch_fps
                            analysis_row['Latency_Reduction_vs_PyTorch'] = pytorch_latency / row['Latency_ms_im']
                        
                        format_analysis_data.append(analysis_row)
    
    format_analysis_df = pd.DataFrame(format_analysis_data)
    return format_analysis_df.round(3)

def generate_ocr_comparison_table():
    """Generate OCR engine comparison table."""
    import json
    
    ocr_data = []
    
    # Load OCR evaluation reports
    engines = [
        ("Fast-Plate-OCR", "traffic-monitor-resources/eval_all_ocr/fpo/evaluation_report.json"),
        ("PPOCRv5", "traffic-monitor-resources/eval_all_ocr/paddle/evaluation_report.json")
    ]
    
    for engine_name, report_path in engines:
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                data = json.load(f)
            
            summary = data.get('summary', {})
            performance = data.get('performance', {})
            detailed = data.get('detailed_metrics', {})
            
            ocr_row = {
                'OCR_Engine': engine_name,
                'Plate_Accuracy': float(summary.get('Plate-level Accuracy', '0.0')),
                'Character_Error_Rate': float(summary.get('Character Error Rate (CER)', '0.0')),
                'Precision': float(summary.get('Plate-level Precision', '0.0')),
                'Recall': float(summary.get('Plate-level Recall', '0.0')),
                'F1_Score': float(summary.get('Plate-level F1 Score', '0.0')),
                'Avg_Latency_ms': float(performance.get('Average Latency (ms)', '0.0')),
                'Median_Latency_ms': float(performance.get('Median Latency (ms)', '0.0')),
                'Min_Latency_ms': float(performance.get('Min Latency (ms)', '0.0')),
                'Max_Latency_ms': float(performance.get('Max Latency (ms)', '0.0')),
                'Throughput_plates_sec': float(performance.get('Throughput (plates/sec)', '0.0')),
                'Total_Samples': detailed.get('total_samples', 0),
                'Correct_Detections': detailed.get('correct_detections', 0),
                'Total_Characters': detailed.get('total_characters', 0),
                'Correct_Characters': detailed.get('correct_characters', 0),
                'Insertion_Errors': detailed.get('insertion_errors', 0),
                'Deletion_Errors': detailed.get('deletion_errors', 0),
                'Substitution_Errors': detailed.get('substitution_errors', 0),
            }
            
            # Calculate additional metrics
            if ocr_row['Total_Characters'] > 0:
                ocr_row['Character_Accuracy'] = ocr_row['Correct_Characters'] / ocr_row['Total_Characters']
            
            if ocr_row['Avg_Latency_ms'] > 0:
                ocr_row['Efficiency_Score'] = ocr_row['Plate_Accuracy'] / ocr_row['Avg_Latency_ms']
            
            ocr_data.append(ocr_row)
    
    ocr_df = pd.DataFrame(ocr_data)
    return ocr_df.round(4)

def generate_all_comprehensive_tables():
    """Generate all comprehensive tables and save them."""
    
    output_dir = os.path.join('traffic-monitor-resources', 'Evaluation_Results', 'comprehensive_tables')
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating comprehensive comparison tables...")
    
    # 1. Model specifications table
    model_specs_df = get_model_specs_df()
    model_specs_df.to_csv(os.path.join(output_dir, "model_specifications.csv"), index=False)
    print(f"[SUCCESS] Model specifications saved to {os.path.join(output_dir, 'model_specifications.csv')}")
    
    # 2. Comprehensive model performance table
    try:
        comprehensive_df = generate_comprehensive_model_table()
        comprehensive_df.to_csv(os.path.join(output_dir, "comprehensive_model_performance.csv"), index=False)
        print(f"[SUCCESS] Comprehensive model performance saved to {os.path.join(output_dir, 'comprehensive_model_performance.csv')}")
        print("\nComprehensive Model Performance Table Preview:")
        print(comprehensive_df[['Model', 'Parameters_M', 'GFLOPs', 'Vehicle_mAP50_95', 'Vehicle_FPS', 'Plate_mAP50_95', 'Plate_FPS']].to_string(index=False))
    except Exception as e:
        print(f"[WARNING] Error generating comprehensive model table: {e}")
    
    # 3. Export format analysis
    try:
        format_analysis_df = generate_export_format_analysis()
        format_analysis_df.to_csv(os.path.join(output_dir, "export_format_analysis.csv"), index=False)
        print(f"[SUCCESS] Export format analysis saved to {os.path.join(output_dir, 'export_format_analysis.csv')}")
    except Exception as e:
        print(f"[WARNING] Error generating export format analysis: {e}")
    
    # 4. Export format specifications
    export_specs_df = get_export_format_specs_df()
    export_specs_df.to_csv(os.path.join(output_dir, "export_format_specifications.csv"), index=False)
    print(f"[SUCCESS] Export format specifications saved to {os.path.join(output_dir, 'export_format_specifications.csv')}")
    
    # 5. Hardware specifications
    hardware_specs_df = get_hardware_specs_df()
    hardware_specs_df.to_csv(os.path.join(output_dir, "hardware_specifications.csv"), index=False)
    print(f"[SUCCESS] Hardware specifications saved to {os.path.join(output_dir, 'hardware_specifications.csv')}")
    
    # 6. OCR comparison table
    try:
        ocr_df = generate_ocr_comparison_table()
        ocr_df.to_csv(os.path.join(output_dir, "ocr_engine_comparison.csv"), index=False)
        print(f"[SUCCESS] OCR engine comparison saved to {os.path.join(output_dir, 'ocr_engine_comparison.csv')}")
        print("\nOCR Engine Comparison Preview:")
        print(ocr_df[['OCR_Engine', 'Plate_Accuracy', 'Character_Error_Rate', 'Avg_Latency_ms', 'Throughput_plates_sec']].to_string(index=False))
    except Exception as e:
        print(f"[WARNING] Error generating OCR comparison table: {e}")
    
    print("\n[SUCCESS] All comprehensive tables generated successfully!")
    print(f"Output directory: {output_dir}")
    
    return output_dir

if __name__ == "__main__":
    generate_all_comprehensive_tables() 