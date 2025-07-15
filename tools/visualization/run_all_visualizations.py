"""
Master script to generate all thesis visualizations and create a summary report.
"""

import os
import sys
import subprocess
from datetime import datetime

def run_script(script_name):
    """Run a Python script and capture its output."""
    try:
        print(f"\n{'='*60}")
        print(f"Running {script_name}...")
        print(f"{'='*60}")
        
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, cwd=".")
        
        if result.returncode == 0:
            print(f"[SUCCESS] {script_name} completed successfully")
            if result.stdout:
                print("Output:")
                print(result.stdout)
        else:
            print(f"[FAILED] {script_name} failed with return code {result.returncode}")
            if result.stderr:
                print("Error:")
                print(result.stderr)
                
        return result.returncode == 0, result.stdout, result.stderr
        
    except Exception as e:
        print(f"[FAILED] Failed to run {script_name}: {e}")
        return False, "", str(e)

def generate_summary_report():
    """Generate a comprehensive summary report of all visualizations."""
    
    report_content = f"""
# Thesis Visualization Summary Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview
This report summarizes all generated visualizations for the thesis on 
"Development of image processing algorithm with deep learning for vehicle counting and license plate number recognition"

## Generated Visualizations

### 1. Vehicle Detection Analysis
**Location:** `trafficmonitor/output/vehicle_detection_visualizations/`

**Files Generated:**
- `vehicle_pytorch_performance.csv` - PyTorch model performance table
- `vehicle_tensorrt_performance.csv` - TensorRT model performance table  
- `vehicle_all_formats_summary.csv` - Complete summary across all export formats
- `vehicle_mAP_comparison.png` - mAP50-95 comparison across models
- `vehicle_latency_comparison.png` - Inference latency comparison
- `vehicle_fps_comparison.png` - FPS performance comparison
- `vehicle_pytorch_accuracy_latency.png` - Accuracy vs latency trade-off (PyTorch)
- `vehicle_tensorrt_accuracy_latency.png` - Accuracy vs latency trade-off (TensorRT)

### 2. License Plate Detection Analysis
**Location:** `trafficmonitor/output/plate_detection_visualizations/`

**Files Generated:**
- `plate_pytorch_performance.csv` - PyTorch model performance table
- `plate_tensorrt_performance.csv` - TensorRT model performance table
- `plate_all_formats_summary.csv` - Complete summary across all export formats
- `plate_mAP_comparison.png` - mAP50-95 comparison across models
- `plate_latency_comparison.png` - Inference latency comparison
- `plate_fps_comparison.png` - FPS performance comparison
- `plate_pytorch_accuracy_latency.png` - Accuracy vs latency trade-off (PyTorch)
- `plate_tensorrt_accuracy_latency.png` - Accuracy vs latency trade-off (TensorRT)
- `plate_performance_heatmap.png` - Comprehensive performance heatmap

### 3. OCR Engine Analysis
**Location:** `trafficmonitor/output/ocr_visualizations/`

**Files Generated:**
- `ocr_performance_table.csv` - OCR engine comparison table
- `ocr_accuracy_latency_scatter.png` - Accuracy vs latency scatter plot
- `ocr_cer_throughput_scatter.png` - Character Error Rate vs throughput analysis

### 4. Comprehensive Tables & Analysis
**Location:** `trafficmonitor/output/comprehensive_tables/`

**Files Generated:**
- `model_specifications.csv` - Detailed model architecture specifications
- `comprehensive_model_performance.csv` - Complete performance comparison with GFLOPs, parameters, efficiency metrics
- `export_format_analysis.csv` - Export format performance analysis with speedup calculations
- `export_format_specifications.csv` - Technical specifications for each export format
- `hardware_specifications.csv` - Hardware capability reference table
- `ocr_engine_comparison.csv` - Detailed OCR engine comparison with character-level metrics

## Key Findings Summary

### Vehicle Detection Models (PyTorch):
- Best mAP50-95: Varies by model architecture
- Fastest inference: Model-dependent, check generated tables
- Best trade-off: Refer to accuracy vs latency plots

### License Plate Detection Models:
- Performance characteristics detailed in generated tables
- TensorRT optimization results in separate analysis
- Comprehensive performance matrix in heatmap visualization

### OCR Engines:
- PPOCRv5 vs Fast-Plate-OCR comparison
- Trade-off analysis between accuracy and speed
- Character-level error analysis available

## Usage Instructions

1. **For Thesis Writing:** 
   - Use CSV tables for numerical data in your thesis
   - Include PNG plots as figures with appropriate captions
   - Reference the comprehensive performance analysis

2. **For Presentations:**
   - High-resolution PNG files (300 DPI) ready for academic presentations
   - Clear, publication-ready visualizations with proper titles and labels

3. **For Further Analysis:**
   - Raw data available in CSV format for custom analysis
   - Reproducible results with documented methodology

## File Locations
All visualizations are organized in the `trafficmonitor/output/` directory:
```
trafficmonitor/output/
├── vehicle_detection_visualizations/
├── plate_detection_visualizations/
└── ocr_visualizations/
```

## Notes
- All plots are generated in high resolution (300 DPI) for publication quality
- CSV files use standard formatting for easy import into Excel/LaTeX
- Visualizations follow consistent color schemes and formatting
- All metrics are rounded to 3 decimal places for readability

---
*This report was automatically generated by the thesis visualization pipeline.*
"""
    
    # Create output directory if it doesn't exist
    os.makedirs("trafficmonitor/output", exist_ok=True)
    
    # Write the report
    report_path = os.path.join('traffic-monitor-resources', 'Evaluation_Results', 'THESIS_VISUALIZATION_SUMMARY.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\nSummary report generated: {report_path}")
    return report_path

def main():
    """Main function to run all visualization scripts."""
    
    print("Starting Thesis Visualization Pipeline")
    print("=" * 60)
    
    # List of scripts to run
    scripts = [
        "trafficmonitor/tools/generate_comprehensive_tables.py",
        "trafficmonitor/tools/visualize_vehicle_detection.py",
        "trafficmonitor/tools/visualize_plate_detection.py", 
        "trafficmonitor/tools/visualize_ocr_benchmarks.py"
    ]
    
    results = {}
    
    # Run each script
    for script in scripts:
        if os.path.exists(script):
            success, stdout, stderr = run_script(script)
            results[script] = {
                'success': success,
                'stdout': stdout,
                'stderr': stderr
            }
        else:
            print(f"[WARNING] Script not found: {script}")
            results[script] = {
                'success': False,
                'stdout': '',
                'stderr': f'Script not found: {script}'
            }
    
    # Print summary
    print(f"\n{'='*60}")
    print("PIPELINE EXECUTION SUMMARY")
    print(f"{'='*60}")
    
    successful_scripts = 0
    for script, result in results.items():
        status = "[SUCCESS]" if result['success'] else "[FAILED]"
        print(f"{status}: {os.path.basename(script)}")
        if result['success']:
            successful_scripts += 1
    
    print(f"\nTotal: {successful_scripts}/{len(scripts)} scripts completed successfully")
    
    # Generate summary report
    if successful_scripts > 0:
        report_path = generate_summary_report()
        print("\n🎉 Visualization pipeline completed!")
        print(f"📋 Summary report: {report_path}")
        print(f"📁 Output directory: {os.path.join('traffic-monitor-resources', 'Evaluation_Results')}")
    else:
        print("\n❌ No visualizations were generated successfully.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 