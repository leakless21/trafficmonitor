import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

def parse_ocr_evaluation_report(report_path):
    """Parses an OCR evaluation report JSON file and returns key metrics."""
    with open(report_path, 'r') as f:
        data = json.load(f)
    
    summary = data.get('summary', {})
    performance = data.get('performance', {})

    plate_accuracy = float(summary.get('Plate-level Accuracy', '0.0'))
    cer = float(summary.get('Character Error Rate (CER)', '0.0'))
    avg_latency = float(performance.get('Average Latency (ms)', '0.0'))
    throughput = float(performance.get('Throughput (plates/sec)', '0.0'))

    return {
        'Plate_Accuracy': plate_accuracy,
        'CER': cer,
        'Avg_Latency_ms': avg_latency,
        'Throughput_plates_sec': throughput
    }

def generate_ocr_visualizations(base_path="traffic-monitor-resources/eval_all_ocr"):
    """
    Generates visualizations for OCR engine benchmarks.
    """
    engines = ["fpo", "paddle"]
    engine_names = {"fpo": "Fast-Plate-OCR", "paddle": "PPOCRv5"}
    
    all_data = []

    for engine_key in engines:
        report_path = os.path.join(base_path, engine_key, "evaluation_report.json")
        if os.path.exists(report_path):
            metrics = parse_ocr_evaluation_report(report_path)
            metrics['Engine'] = engine_names[engine_key]
            all_data.append(metrics)
        else:
            print(f"Warning: Report file not found for {engine_names[engine_key]} at {report_path}")

    if not all_data:
        print("No OCR evaluation data found to generate visualizations.")
        return

    combined_df = pd.DataFrame(all_data)

    output_dir = os.path.join('traffic-monitor-resources', 'Evaluation_Results', 'ocr_visualizations')
    os.makedirs(output_dir, exist_ok=True)

    # --- Generate Table ---
    combined_df.set_index('Engine', inplace=True)
    combined_df.round(4).to_csv(os.path.join(output_dir, "ocr_performance_table.csv"))
    print(f"OCR performance table saved to {os.path.join(output_dir, 'ocr_performance_table.csv')}")
    print("\nOCR Performance Table:")
    print(combined_df.round(4))

    # --- Generate Plots ---

    # Plot 1: Plate Accuracy vs. Average Latency
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='Avg_Latency_ms', y='Plate_Accuracy', hue='Engine', 
                    data=combined_df, s=200, palette='viridis')
    plt.title('OCR Engine Accuracy vs. Latency')
    plt.xlabel('Average Latency (ms/plate)')
    plt.ylabel('Plate-level Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    for i, row in combined_df.reset_index().iterrows():
        plt.text(row['Avg_Latency_ms'] + 0.5, row['Plate_Accuracy'] - 0.01, row['Engine'], 
                 horizontalalignment='left', size='small', color='black', weight='semibold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ocr_accuracy_latency_scatter.png"))
    plt.close()
    print(f"OCR accuracy vs. latency scatter plot saved to {os.path.join(output_dir, 'ocr_accuracy_latency_scatter.png')}")

    # Plot 2: CER vs. Throughput
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='Throughput_plates_sec', y='CER', hue='Engine', 
                    data=combined_df, s=200, palette='plasma')
    plt.title('OCR Engine CER vs. Throughput')
    plt.xlabel('Throughput (plates/sec)')
    plt.ylabel('Character Error Rate (CER)')
    plt.grid(True, linestyle='--', alpha=0.7)
    for i, row in combined_df.reset_index().iterrows():
        plt.text(row['Throughput_plates_sec'] + 2, row['CER'] - 0.001, row['Engine'], 
                 horizontalalignment='left', size='small', color='black', weight='semibold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ocr_cer_throughput_scatter.png"))
    plt.close()
    print(f"OCR CER vs. throughput scatter plot saved to {os.path.join(output_dir, 'ocr_cer_throughput_scatter.png')}")


if __name__ == "__main__":
    generate_ocr_visualizations() 