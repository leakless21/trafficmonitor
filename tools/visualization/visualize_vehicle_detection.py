import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import sys

# Add the tools directory to path to import model_specifications
sys.path.append(os.path.join(os.path.dirname(__file__)))
from model_specifications import get_model_spec, get_efficiency_metrics

def parse_benchmark_log(log_path):
    """Parses a single benchmark log file and returns a DataFrame."""
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    data = []
    # Skip header lines until we find the data rows
    start_parsing = False
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

def generate_vehicle_detection_visualizations(base_path="traffic-monitor-resources/Results"):
    """
    Generates visualizations specifically for vehicle detection benchmarks.
    """
    detection_type = "Vehicles"
    models = ["YOLO11n", "YOLOv10n", "YOLOv5u", "YOLOv8n"]
    
    all_data = []

    for model in models:
        log_path = os.path.join(base_path, detection_type, model, "benchmarks.log")
        if os.path.exists(log_path):
            df = parse_benchmark_log(log_path)
            if not df.empty:
                df['Model'] = model
                all_data.append(df)
        else:
            print(f"Warning: Log file not found for {detection_type}/{model} at {log_path}")

    if not all_data:
        print("No vehicle detection benchmark data found to generate visualizations.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)

    # Filter for successful exports only
    combined_df = combined_df[combined_df['Status'] == '✅']

    output_dir = os.path.join('traffic-monitor-resources', 'Evaluation_Results', 'vehicle_detection_visualizations')
    os.makedirs(output_dir, exist_ok=True)

    # --- Generate Tables ---
    # Enhanced PyTorch performance table with model specifications
    pytorch_data = combined_df[combined_df['Format'] == 'PyTorch'].copy()
    
    # Add model specifications
    for idx, row in pytorch_data.iterrows():
        model_name = row['Model']
        specs = get_model_spec(model_name)
        if specs:
            pytorch_data.loc[idx, 'Parameters_M'] = specs.get('parameters_millions', 0)
            pytorch_data.loc[idx, 'GFLOPs'] = specs.get('gflops_640', 0)
            pytorch_data.loc[idx, 'Architecture'] = specs.get('architecture_type', 'Unknown')
            pytorch_data.loc[idx, 'Backbone'] = specs.get('backbone', 'Unknown')
            pytorch_data.loc[idx, 'Anchor_Based'] = specs.get('anchor_based', False)
            
            # Calculate efficiency metrics
            efficiency = get_efficiency_metrics(model_name, row['FPS'], row['mAP50-95'])
            pytorch_data.loc[idx, 'FPS_per_GFLOP'] = efficiency.get('fps_per_gflop', 0)
            pytorch_data.loc[idx, 'mAP_per_GFLOP'] = efficiency.get('map_per_gflop', 0)
            pytorch_data.loc[idx, 'Efficiency_Score'] = efficiency.get('efficiency_score', 0)
    
    # Select and order columns for display
    detailed_columns = ['Model', 'Parameters_M', 'GFLOPs', 'Size_MB', 'mAP50-95', 
                       'Latency_ms_im', 'FPS', 'FPS_per_GFLOP', 'mAP_per_GFLOP', 
                       'Efficiency_Score', 'Architecture', 'Backbone', 'Anchor_Based']
    
    pytorch_detailed = pytorch_data[detailed_columns].round(3)
    pytorch_detailed.to_csv(os.path.join(output_dir, "vehicle_pytorch_detailed_performance.csv"), index=False)
    
    # Basic performance table (for compatibility)
    pytorch_basic = pytorch_data[['Model', 'Size_MB', 'mAP50-95', 'Latency_ms_im', 'FPS']].round(3)
    pytorch_basic.to_csv(os.path.join(output_dir, "vehicle_pytorch_performance.csv"), index=False)
    
    print(f"Vehicle PyTorch detailed performance table saved to {os.path.join(output_dir, 'vehicle_pytorch_detailed_performance.csv')}")
    print(f"Vehicle PyTorch basic performance table saved to {os.path.join(output_dir, 'vehicle_pytorch_performance.csv')}")
    print("\nVehicle Detection - PyTorch Detailed Performance Table:")
    print(pytorch_detailed.to_string(index=False))

    # TensorRT performance table
    tensorrt_data = combined_df[combined_df['Format'] == 'TensorRT']
    if not tensorrt_data.empty:
        tensorrt_df = tensorrt_data[['Model', 'mAP50-95', 'Latency_ms_im', 'FPS']].round(3)
        tensorrt_df.to_csv(os.path.join(output_dir, "vehicle_tensorrt_performance.csv"), index=False)
        print(f"Vehicle TensorRT performance table saved to {os.path.join(output_dir, 'vehicle_tensorrt_performance.csv')}")
        print("\nVehicle Detection - TensorRT Performance Table:")
        print(tensorrt_df.to_string(index=False))

    # All formats summary
    summary_df = combined_df.groupby(['Model', 'Format']).agg({
        'mAP50-95': 'first',
        'Latency_ms_im': 'first',
        'FPS': 'first',
        'Size_MB': 'first'
    }).round(3).reset_index()
    summary_df.to_csv(os.path.join(output_dir, "vehicle_all_formats_summary.csv"), index=False)
    print(f"Vehicle all formats summary saved to {os.path.join(output_dir, 'vehicle_all_formats_summary.csv')}")

    # --- Generate Plots ---

    # Plot 1: mAP50-95 comparison across models and formats
    plt.figure(figsize=(14, 8))
    sns.barplot(data=combined_df, x='Model', y='mAP50-95', hue='Format', palette='viridis')
    plt.title('Vehicle Detection: mAP50-95 Comparison Across Models and Formats', fontsize=16, fontweight='bold')
    plt.ylabel('mAP50-95', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Export Format', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "vehicle_mAP_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Vehicle mAP comparison plot saved to {os.path.join(output_dir, 'vehicle_mAP_comparison.png')}")

    # Plot 2: Latency comparison
    plt.figure(figsize=(14, 8))
    sns.barplot(data=combined_df, x='Model', y='Latency_ms_im', hue='Format', palette='magma')
    plt.title('Vehicle Detection: Inference Latency Comparison (Lower is Better)', fontsize=16, fontweight='bold')
    plt.ylabel('Latency (ms/image)', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Export Format', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "vehicle_latency_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Vehicle latency comparison plot saved to {os.path.join(output_dir, 'vehicle_latency_comparison.png')}")

    # Plot 3: FPS comparison
    plt.figure(figsize=(14, 8))
    sns.barplot(data=combined_df, x='Model', y='FPS', hue='Format', palette='plasma')
    plt.title('Vehicle Detection: FPS Comparison (Higher is Better)', fontsize=16, fontweight='bold')
    plt.ylabel('Frames Per Second (FPS)', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Export Format', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "vehicle_fps_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Vehicle FPS comparison plot saved to {os.path.join(output_dir, 'vehicle_fps_comparison.png')}")

    # Plot 4: GFLOPs vs mAP scatter plot
    if not pytorch_data.empty and 'GFLOPs' in pytorch_data.columns:
        plt.figure(figsize=(10, 7))
        sns.scatterplot(data=pytorch_data, x='GFLOPs', y='mAP50-95', size='FPS', 
                       hue='Model', sizes=(100, 400), alpha=0.7, palette='deep')
        plt.title('Vehicle Detection: Model Complexity vs Accuracy (PyTorch)', fontsize=16, fontweight='bold')
        plt.xlabel('GFLOPs (Computational Complexity)', fontsize=12)
        plt.ylabel('mAP50-95', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add model labels
        for _, row in pytorch_data.iterrows():
            plt.annotate(f"{row['Model']}\n({row['Parameters_M']}M params)", 
                        (row['GFLOPs'], row['mAP50-95']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, alpha=0.8)
        
        plt.legend(title='FPS', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "vehicle_complexity_accuracy.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Vehicle complexity vs accuracy plot saved to {os.path.join(output_dir, 'vehicle_complexity_accuracy.png')}")

    # Plot 5: Accuracy vs Latency scatter for PyTorch
    pytorch_subset = combined_df[combined_df['Format'] == 'PyTorch']
    if not pytorch_subset.empty:
        plt.figure(figsize=(10, 7))
        sns.scatterplot(data=pytorch_subset, x='Latency_ms_im', y='mAP50-95', hue='Model', s=200, palette='deep')
        plt.title('Vehicle Detection: Accuracy vs Latency Trade-off (PyTorch)', fontsize=16, fontweight='bold')
        plt.xlabel('Latency (ms/image)', fontsize=12)
        plt.ylabel('mAP50-95', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add model labels
        for _, row in pytorch_subset.iterrows():
            plt.annotate(row['Model'], 
                        (row['Latency_ms_im'], row['mAP50-95']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "vehicle_pytorch_accuracy_latency.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Vehicle PyTorch accuracy vs latency plot saved to {os.path.join(output_dir, 'vehicle_pytorch_accuracy_latency.png')}")

    # Plot 6: Accuracy vs Latency scatter for TensorRT
    if not tensorrt_data.empty:
        plt.figure(figsize=(10, 7))
        sns.scatterplot(data=tensorrt_data, x='Latency_ms_im', y='mAP50-95', hue='Model', s=200, palette='deep')
        plt.title('Vehicle Detection: Accuracy vs Latency Trade-off (TensorRT)', fontsize=16, fontweight='bold')
        plt.xlabel('Latency (ms/image)', fontsize=12)
        plt.ylabel('mAP50-95', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add model labels
        for _, row in tensorrt_data.iterrows():
            plt.annotate(row['Model'], 
                        (row['Latency_ms_im'], row['mAP50-95']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "vehicle_tensorrt_accuracy_latency.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Vehicle TensorRT accuracy vs latency plot saved to {os.path.join(output_dir, 'vehicle_tensorrt_accuracy_latency.png')}")

    print(f"\nAll vehicle detection visualizations saved to: {output_dir}")


if __name__ == "__main__":
    generate_vehicle_detection_visualizations() 