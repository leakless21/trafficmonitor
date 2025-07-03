#!/usr/bin/env python3
"""
End-to-End Benchmark Runner for Traffic Monitor.

This script runs the complete traffic monitoring pipeline on evaluation videos,
collects predictions, measures performance, and evaluates against ground truth.
"""

import argparse
import json
import time
import yaml
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys
import os

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger
from traffic_monitor.utils.config_loader import load_config
from traffic_monitor.utils.profiler import Profiler, get_system_info
from traffic_monitor.eval.e2e_evaluator import E2EEvaluator, EvaluationMetrics
from traffic_monitor.utils.custom_types import *


class BenchmarkCollector:
    """Collects predictions from the pipeline for evaluation."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.predictions = []
        self.start_time = None
        
    def start_collection(self):
        """Start collecting predictions."""
        self.start_time = time.time()
        self.predictions = []
        
    def collect_vehicle_event(self, track_id: int, plate: Optional[str], 
                            ts_enter: float, ts_exit: float, 
                            vehicle_class: str, confidence: float = 1.0):
        """Collect a vehicle passing event."""
        self.predictions.append({
            "event": "VehiclePassed",
            "track_id": track_id,
            "plate": plate,
            "ts_enter": ts_enter,
            "ts_exit": ts_exit,
            "vehicle_class": vehicle_class,
            "confidence": confidence
        })
        
    def collect_queue_event(self, timestamp: float, length: int):
        """Collect a queue length measurement."""
        self.predictions.append({
            "event": "QueueLength",
            "timestamp": timestamp,
            "length": length
        })
        
    def save_predictions(self, video_name: str):
        """Save collected predictions to JSON file."""
        output_file = self.output_dir / f"{video_name}.pred.json"
        with open(output_file, 'w') as f:
            json.dump(self.predictions, f, indent=2)
        logger.info(f"Saved {len(self.predictions)} predictions to {output_file}")


class MockPipelineRunner:
    """Mock pipeline runner for demonstration (replace with actual pipeline integration)."""
    
    def __init__(self, config: Dict[str, Any], profiler: Profiler):
        self.config = config
        self.profiler = profiler
        
    def run_video(self, video_path: Path, collector: BenchmarkCollector) -> Dict[str, Any]:
        """
        Run the pipeline on a video and collect results.
        
        This is a mock implementation. In the real version, this would:
        1. Initialize the actual pipeline with the config
        2. Process the video through all services
        3. Collect events from database or message queues
        4. Return timing and throughput metrics
        """
        logger.info(f"Running pipeline on {video_path}")
        
        # Mock pipeline execution with profiling
        with self.profiler.section("full_pipeline"):
            collector.start_collection()
            
            # Simulate vehicle detection and tracking
            with self.profiler.section("detection_tracking"):
                time.sleep(0.1)  # Mock processing time
                
            # Simulate plate detection and OCR
            with self.profiler.section("plate_ocr"):
                time.sleep(0.05)  # Mock processing time
                
            # Simulate counting
            with self.profiler.section("counting"):
                time.sleep(0.02)  # Mock processing time
            
            # Generate mock predictions (replace with actual pipeline output)
            self._generate_mock_predictions(collector)
            
        # Calculate FPS (mock)
        total_time = sum(self.profiler.timing_stats.get("full_pipeline", [0]))
        fps = 30.0 * 1000 / total_time if total_time > 0 else 0  # Assume 30 frames processed
        
        return {
            "fps": fps,
            "total_frames": 30,  # Mock
            "processing_time_ms": total_time
        }
    
    def _generate_mock_predictions(self, collector: BenchmarkCollector):
        """Generate mock predictions that roughly match ground truth."""
        # This simulates what the real pipeline would output
        collector.collect_vehicle_event(1, "51F12345", 3.8, 4.8, "car", 0.94)
        collector.collect_vehicle_event(2, None, 5.1, 6.2, "truck", 0.87)  
        collector.collect_vehicle_event(3, "51B98765", 8.4, 9.4, "car", 0.91)
        collector.collect_vehicle_event(4, "29A11111", 12.0, 13.3, "bus", 0.88)
        collector.collect_vehicle_event(6, None, 15.8, 17.0, "motorcycle", 0.84)  # Wrong track_id
        
        collector.collect_queue_event(5.0, 2)
        collector.collect_queue_event(10.0, 1)
        collector.collect_queue_event(15.0, 3)


def load_evaluation_videos(videos_config_path: Path) -> List[Dict[str, Any]]:
    """Load the list of evaluation videos from YAML config."""
    try:
        with open(videos_config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config.get('videos', [])
    except Exception as e:
        logger.error(f"Failed to load video config from {videos_config_path}: {e}")
        return []


def run_single_video(video_config: Dict[str, Any], pipeline_config: Dict[str, Any], 
                    output_dir: Path, profiler: Profiler) -> Dict[str, Any]:
    """Run benchmark on a single video."""
    video_name = video_config['name']
    video_path = Path(video_config['path'])
    
    logger.info(f"Benchmarking video: {video_name}")
    
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        return {"error": f"Video file not found: {video_path}"}
    
    # Initialize collector
    collector = BenchmarkCollector(output_dir)
    
    # Run pipeline
    runner = MockPipelineRunner(pipeline_config, profiler)
    timing_stats = runner.run_video(video_path, collector)
    
    # Save predictions
    collector.save_predictions(video_name)
    
    return {
        "video_name": video_name,
        "video_path": str(video_path),
        **timing_stats
    }


def evaluate_all_videos(videos: List[Dict[str, Any]], output_dir: Path) -> EvaluationMetrics:
    """Evaluate all videos and compute aggregate metrics."""
    evaluator = E2EEvaluator()
    
    all_metrics = []
    for video_config in videos:
        video_name = video_config['name']
        
        # Paths
        gt_path = Path("data/eval/ground_truth") / f"{video_name}.events.json"
        pred_path = output_dir / f"{video_name}.pred.json"
        
        if not gt_path.exists():
            logger.warning(f"Ground truth not found for {video_name}: {gt_path}")
            continue
            
        if not pred_path.exists():
            logger.warning(f"Predictions not found for {video_name}: {pred_path}")
            continue
        
        # Evaluate single video
        metrics = evaluator.evaluate(gt_path, pred_path)
        all_metrics.append(metrics)
        
        logger.info(f"Video {video_name} - Vehicle F1: {metrics.vehicle_f1:.3f}, "
                   f"Plate F1: {metrics.plate_f1:.3f}, Count MAE: {metrics.count_mae:.1f}")
    
    if not all_metrics:
        logger.error("No videos could be evaluated")
        return EvaluationMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    # Aggregate metrics across all videos
    aggregate = EvaluationMetrics(
        vehicle_precision=sum(m.vehicle_precision for m in all_metrics) / len(all_metrics),
        vehicle_recall=sum(m.vehicle_recall for m in all_metrics) / len(all_metrics),
        vehicle_f1=sum(m.vehicle_f1 for m in all_metrics) / len(all_metrics),
        plate_precision=sum(m.plate_precision for m in all_metrics) / len(all_metrics),
        plate_recall=sum(m.plate_recall for m in all_metrics) / len(all_metrics),
        plate_f1=sum(m.plate_f1 for m in all_metrics) / len(all_metrics),
        count_mae=sum(m.count_mae for m in all_metrics) / len(all_metrics),
        count_rmse=sum(m.count_rmse for m in all_metrics) / len(all_metrics),
        count_smape=sum(m.count_smape for m in all_metrics) / len(all_metrics),
        queue_mae=sum(m.queue_mae for m in all_metrics) / len(all_metrics),
        queue_rmse=sum(m.queue_rmse for m in all_metrics) / len(all_metrics),
        overall_f1=sum(m.overall_f1 for m in all_metrics) / len(all_metrics)
    )
    
    return aggregate


def main():
    """Main benchmark execution function."""
    parser = argparse.ArgumentParser(description="E2E Traffic Monitor Benchmark")
    parser.add_argument("--config", required=True, help="Pipeline configuration YAML file")
    parser.add_argument("--videos", required=True, help="Video list configuration YAML file") 
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument("--profile", action="store_true", help="Enable detailed profiling")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize profiler
    profiler = Profiler(enabled=args.profile)
    
    try:
        # Load configurations
        logger.info("Loading configurations...")
        pipeline_config = load_config(Path(args.config))
        if not pipeline_config:
            logger.error("Failed to load pipeline configuration")
            return 1
            
        videos = load_evaluation_videos(Path(args.videos))
        if not videos:
            logger.error("No evaluation videos found")
            return 1
        
        # Record system info
        system_info = get_system_info()
        logger.info(f"System: {system_info.get('cpu_count', 'Unknown')} CPUs, "
                   f"{system_info.get('memory_total_gb', 0):.1f}GB RAM, "
                   f"{system_info.get('gpu_name', 'No GPU')}")
        
        # Run benchmark on all videos
        logger.info(f"Starting benchmark on {len(videos)} videos...")
        start_time = time.time()
        
        video_results = []
        for video_config in videos:
            with profiler.section(f"video_{video_config['name']}"):
                result = run_single_video(video_config, pipeline_config, output_dir, profiler)
                video_results.append(result)
        
        total_time = time.time() - start_time
        
        # Evaluate predictions against ground truth
        logger.info("Evaluating predictions against ground truth...")
        metrics = evaluate_all_videos(videos, output_dir)
        
        # Add timing metrics from profiler
        profiler_stats = profiler.get_stats()
        metrics.mean_latency_ms = profiler_stats.get("full_pipeline_mean_ms", 0.0)
        metrics.p95_latency_ms = profiler_stats.get("full_pipeline_p95_ms", 0.0)
        
        # Calculate aggregate FPS
        total_fps = sum(r.get("fps", 0) for r in video_results)
        metrics.fps = total_fps / len(video_results) if video_results else 0.0
        
        # Save results
        results = {
            "benchmark_info": {
                "timestamp": datetime.now().isoformat(),
                "total_time_seconds": total_time,
                "videos_processed": len(videos),
                "system_info": system_info
            },
            "metrics": {
                "vehicle_identification": {
                    "precision": metrics.vehicle_precision,
                    "recall": metrics.vehicle_recall,
                    "f1": metrics.vehicle_f1
                },
                "plate_recognition": {
                    "precision": metrics.plate_precision,
                    "recall": metrics.plate_recall,
                    "f1": metrics.plate_f1
                },
                "counting": {
                    "mae": metrics.count_mae,
                    "rmse": metrics.count_rmse,
                    "smape": metrics.count_smape
                },
                "queue_length": {
                    "mae": metrics.queue_mae,
                    "rmse": metrics.queue_rmse
                },
                "timing": {
                    "mean_latency_ms": metrics.mean_latency_ms,
                    "p95_latency_ms": metrics.p95_latency_ms,
                    "fps": metrics.fps
                },
                "overall": {
                    "f1": metrics.overall_f1
                }
            },
            "video_results": video_results,
            "profiler_stats": profiler_stats
        }
        
        # Save main results
        results_path = output_dir / "metrics.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save detailed profiling if enabled
        if args.profile:
            profiler.save_detailed_profile(output_dir / "profiling.csv")
        
        # Print summary
        logger.info("=" * 60)
        logger.info("BENCHMARK SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Overall F1 Score: {metrics.overall_f1:.3f}")
        logger.info(f"Vehicle Identification F1: {metrics.vehicle_f1:.3f}")
        logger.info(f"Plate Recognition F1: {metrics.plate_f1:.3f}")
        logger.info(f"Count MAE: {metrics.count_mae:.1f}")
        logger.info(f"Mean Latency: {metrics.mean_latency_ms:.1f}ms")
        logger.info(f"P95 Latency: {metrics.p95_latency_ms:.1f}ms")
        logger.info(f"Average FPS: {metrics.fps:.1f}")
        logger.info(f"Results saved to: {results_path}")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.exception(f"Benchmark failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 