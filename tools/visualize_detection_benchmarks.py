import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

def parse_benchmark_log(log_path):
    """Parses a single benchmark log file and returns a DataFrame."""
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    data = []
    # Skip header lines until we find the "Format Status" line
    start_parsing = False
    for line in lines:
        if "Format Status" in line:
            start_parsing = True
            continue
        if start_parsing and line.strip() and not line.strip().startswith('---') and not line.strip().startswith('Benchmarks'):
            # Parse line using regex to handle multi-word format names
            # Format: index format_name status size mAP latency fps
            match = re.match(r'^\s*(\d+)\s+(.+?)\s+(✅|❌|❎)\s+([\d.]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)', line)
            if match:
                index = int(match.group(1))
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

def generate_detection_visualizations(base_path="traffic-monitor-resources/Results"):
    """
    Generates visualizations for vehicle and license plate detection benchmarks.
    """
    detection_types = ["License Plate", "Vehicles"]
    models = ["YOLO11n", "YOLOv10n", "YOLOv5u", "YOLOv8n"]
    
    all_data = []

    for det_type in detection_types:
        for model in models:
            log_path = os.path.join(base_path, det_type, model, "benchmarks.log")
            if os.path.exists(log_path):
                df = parse_benchmark_log(log_path)
                if not df.empty:
                    df['DetectionType'] = det_type
                    df['Model'] = model
                    all_data.append(df)
            else:
                print(f"Warning: Log file not found for {det_type}/{model} at {log_path}")

    if not all_data:
        print("No benchmark data found to generate visualizations.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)

    # Filter for successful exports only
    combined_df = combined_df[combined_df['Status'] == '✅']

    output_dir = "trafficmonitor/output/detection_visualizations"
    os.makedirs(output_dir, exist_ok=True)

    # --- Generate Tables ---
    # PyTorch performance table
    pytorch_df = combined_df[combined_df['Format'] == 'PyTorch'].pivot_table(
        index=['DetectionType', 'Model'], 
        values=['Size_MB', 'mAP50-95', 'Latency_ms_im', 'FPS']
    ).round(3)
    pytorch_df.to_csv(os.path.join(output_dir, "pytorch_detection_performance.csv"))
    print(f"PyTorch performance table saved to {os.path.join(output_dir, 'pytorch_detection_performance.csv')}")
    print("\nPyTorch Performance Table:")
    print(pytorch_df)

    # TensorRT performance table
    tensorrt_df = combined_df[combined_df['Format'] == 'TensorRT'].pivot_table(
        index=['DetectionType', 'Model'], 
        values=['mAP50-95', 'Latency_ms_im', 'FPS']
    ).round(3)
    tensorrt_df.to_csv(os.path.join(output_dir, "tensorrt_detection_performance.csv"))
    print(f"TensorRT performance table saved to {os.path.join(output_dir, 'tensorrt_detection_performance.csv')}")
    print("\nTensorRT Performance Table:")
    print(tensorrt_df)

    # --- Generate Plots ---

    # Plot 1: mAP50-95 by Model and Format
    plt.figure(figsize=(14, 8))
    sns.barplot(x='Model', y='mAP50-95', hue='Format', data=combined_df, palette='viridis')
    plt.title('mAP50-95 Comparison Across Models and Formats')
    plt.ylabel('mAP50-95')
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mAP50-95_comparison.png"))
    plt.close()
    print(f"mAP50-95 comparison plot saved to {os.path.join(output_dir, 'mAP50-95_comparison.png')}")

    # Plot 2: Latency by Model and Format (lower is better)
    plt.figure(figsize=(14, 8))
    sns.barplot(x='Model', y='Latency_ms_im', hue='Format', data=combined_df, palette='magma')
    plt.title('Inference Latency Comparison Across Models and Formats (ms/image)')
    plt.ylabel('Latency (ms/image)')
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latency_comparison.png"))
    plt.close()
    print(f"Latency comparison plot saved to {os.path.join(output_dir, 'latency_comparison.png')}")

    # Plot 3: FPS by Model and Format (higher is better)
    plt.figure(figsize=(14, 8))
    sns.barplot(x='Model', y='FPS', hue='Format', data=combined_df, palette='plasma')
    plt.title('FPS Comparison Across Models and Formats')
    plt.ylabel('Frames Per Second (FPS)')
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fps_comparison.png"))
    plt.close()
    print(f"FPS comparison plot saved to {os.path.join(output_dir, 'fps_comparison.png')}")

    # Plot 4: mAP50-95 vs Latency for PyTorch models
    plt.figure(figsize=(12, 7))
    sns.scatterplot(x='Latency_ms_im', y='mAP50-95', hue='Model', style='DetectionType', 
                    data=combined_df[combined_df['Format'] == 'PyTorch'], s=200, palette='deep')
    plt.title('Accuracy vs. Latency for PyTorch Models')
    plt.xlabel('Latency (ms/image)')
    plt.ylabel('mAP50-95')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pytorch_accuracy_latency_scatter.png"))
    plt.close()
    print(f"PyTorch accuracy vs. latency scatter plot saved to {os.path.join(output_dir, 'pytorch_accuracy_latency_scatter.png')}")
    
    # Plot 5: mAP50-95 vs Latency for TensorRT models
    plt.figure(figsize=(12, 7))
    sns.scatterplot(x='Latency_ms_im', y='mAP50-95', hue='Model', style='DetectionType', 
                    data=combined_df[combined_df['Format'] == 'TensorRT'], s=200, palette='deep')
    plt.title('Accuracy vs. Latency for TensorRT Models')
    plt.xlabel('Latency (ms/image)')
    plt.ylabel('mAP50-95')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tensorrt_accuracy_latency_scatter.png"))
    plt.close()
    print(f"TensorRT accuracy vs. latency scatter plot saved to {os.path.join(output_dir, 'tensorrt_accuracy_latency_scatter.png')}")


if __name__ == "__main__":
    generate_detection_visualizations() 