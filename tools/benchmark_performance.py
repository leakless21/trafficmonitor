#!/usr/bin/env python3
"""
Performance-Only Benchmark Runner for Traffic Monitor.

This script focuses purely on performance metrics (timing, throughput, resource usage)
without requiring ground truth data. Ideal for development performance checks.
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
import statistics

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger
from traffic_monitor.utils.config_loader import load_config
from traffic_monitor.utils.profiler import Profiler, get_system_info


class PerformanceCollector:
    """Collects performance metrics during pipeline execution."""
    
    def __init__(self):
        self.frame_times = []
        self.throughput_data = []
        self.start_time = None
        self.end_time = None
        self.total_frames = 0
        
    def start_collection(self):
        """Start performance data collection."""
        self.start_time = time.time()
        self.frame_times = []
        self.throughput_data = []
        self.total_frames = 0
        
    def record_frame_processed(self, frame_time: float):
        """Record time taken to process a single frame."""
        self.frame_times.append(frame_time)
        self.total_frames += 1
        
        # Calculate rolling throughput (last 30 frames)
        if len(self.frame_times) >= 30:
            recent_times = self.frame_times[-30:]
            avg_frame_time = statistics.mean(recent_times)
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            self.throughput_data.append(fps)
    
    def finish_collection(self):
        """Finish collection and compute final metrics."""
        self.end_time = time.time()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get computed performance metrics."""
        if not self.frame_times:
            return {}
            
        total_time = self.end_time - self.start_time if self.end_time and self.start_time else 0
        
        # Frame timing statistics
        mean_frame_time = statistics.mean(self.frame_times)
        median_frame_time = statistics.median(self.frame_times)
        p95_frame_time = sorted(self.frame_times)[int(0.95 * len(self.frame_times))] if len(self.frame_times) > 1 else self.frame_times[0]
        min_frame_time = min(self.frame_times)
        max_frame_time = max(self.frame_times)
        
        # Throughput statistics
        overall_fps = self.total_frames / total_time if total_time > 0 else 0
        mean_fps = 1.0 / mean_frame_time if mean_frame_time > 0 else 0
        
        # Throughput stability
        throughput_std = statistics.stdev(self.throughput_data) if len(self.throughput_data) > 1 else 0
        
        return {
            "timing": {
                "total_time_seconds": total_time,
                "total_frames": self.total_frames,
                "mean_frame_time_ms": mean_frame_time * 1000,
                "median_frame_time_ms": median_frame_time * 1000,
                "p95_frame_time_ms": p95_frame_time * 1000,
                "min_frame_time_ms": min_frame_time * 1000,
                "max_frame_time_ms": max_frame_time * 1000
            },
            "throughput": {
                "overall_fps": overall_fps,
                "mean_fps": mean_fps,
                "peak_fps": max(self.throughput_data) if self.throughput_data else 0,
                "min_fps": min(self.throughput_data) if self.throughput_data else 0,
                "fps_stability_std": throughput_std
            },
            "efficiency": {
                "frames_per_second": overall_fps,
                "ms_per_frame": mean_frame_time * 1000,
                "realtime_factor": overall_fps / 30.0 if overall_fps > 0 else 0  # Assuming 30fps video
            }
        }


class MockPerformancePipeline:
    """Mock pipeline that simulates realistic performance characteristics."""
    
    def __init__(self, config: Dict[str, Any], profiler: Profiler):
        self.config = config
        self.profiler = profiler
        # Simulate different processing loads based on config
        self.base_frame_time = 0.05 if config.get("fast_mode", False) else 0.1
        
    def process_video(self, video_path: Path, collector: PerformanceCollector) -> Dict[str, Any]:
        """
        Process video with performance monitoring.
        
        In real implementation, this would:
        1. Run the actual traffic monitor pipeline
        2. Measure per-frame processing time
        3. Collect resource usage throughout
        """
        logger.info(f"Processing video: {video_path}")
        
        collector.start_collection()
        
        # Simulate video processing
        num_frames = 100  # Mock number of frames
        
        with self.profiler.section("video_processing"):
            for frame_idx in range(num_frames):
                frame_start = time.time()
                
                # Simulate frame processing components
                with self.profiler.section("frame_grabbing"):
                    time.sleep(0.001)  # Mock frame grab time
                
                with self.profiler.section("detection"):
                    # Simulate variable detection time
                    detection_time = self.base_frame_time * 0.6 + (frame_idx % 10) * 0.002
                    time.sleep(detection_time)
                
                with self.profiler.section("tracking"):
                    time.sleep(self.base_frame_time * 0.2)
                
                with self.profiler.section("plate_ocr"):
                    # Simulate occasional plate processing
                    if frame_idx % 20 == 0:  # Process plate every 20 frames
                        time.sleep(self.base_frame_time * 0.8)
                    else:
                        time.sleep(0.001)
                
                with self.profiler.section("counting"):
                    time.sleep(0.002)
                
                frame_time = time.time() - frame_start
                collector.record_frame_processed(frame_time)
                
                # Log progress periodically
                if frame_idx % 20 == 0:
                    logger.info(f"Processed frame {frame_idx}/{num_frames}")
        
        collector.finish_collection()
        
        return {
            "video_path": str(video_path),
            "frames_processed": num_frames,
            **collector.get_metrics()
        }


def load_performance_videos(videos_config_path: Path) -> List[Dict[str, Any]]:
    """Load video list for performance testing."""
    try:
        with open(videos_config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config.get('videos', [])
    except Exception as e:
        logger.error(f"Failed to load video config from {videos_config_path}: {e}")
        return []


def run_performance_benchmark(video_config: Dict[str, Any], pipeline_config: Dict[str, Any], 
                            profiler: Profiler) -> Dict[str, Any]:
    """Run performance benchmark on a single video."""
    video_name = video_config['name']
    video_path = Path(video_config['path'])
    
    logger.info(f"Performance benchmarking: {video_name}")
    
    if not video_path.exists():
        logger.warning(f"Video file not found (using mock): {video_path}")
    
    # Initialize collector and pipeline
    collector = PerformanceCollector()
    pipeline = MockPerformancePipeline(pipeline_config, profiler)
    
    # Run performance test
    with profiler.section(f"video_{video_name}"):
        result = pipeline.process_video(video_path, collector)
    
    return {
        "video_name": video_name,
        **result
    }


def compute_aggregate_metrics(video_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate performance metrics across all videos."""
    if not video_results:
        return {}
    
    # Collect timing data
    all_fps = [r.get("throughput", {}).get("overall_fps", 0) for r in video_results]
    all_frame_times = [r.get("timing", {}).get("mean_frame_time_ms", 0) for r in video_results]
    all_p95_times = [r.get("timing", {}).get("p95_frame_time_ms", 0) for r in video_results]
    
    # Filter out zeros
    valid_fps = [fps for fps in all_fps if fps > 0]
    valid_frame_times = [ft for ft in all_frame_times if ft > 0]
    valid_p95_times = [p95 for p95 in all_p95_times if p95 > 0]
    
    aggregate = {}
    
    if valid_fps:
        aggregate["fps"] = {
            "mean": statistics.mean(valid_fps),
            "median": statistics.median(valid_fps),
            "min": min(valid_fps),
            "max": max(valid_fps),
            "std": statistics.stdev(valid_fps) if len(valid_fps) > 1 else 0
        }
    
    if valid_frame_times:
        aggregate["frame_time_ms"] = {
            "mean": statistics.mean(valid_frame_times),
            "median": statistics.median(valid_frame_times),
            "min": min(valid_frame_times),
            "max": max(valid_frame_times),
            "std": statistics.stdev(valid_frame_times) if len(valid_frame_times) > 1 else 0
        }
    
    if valid_p95_times:
        aggregate["p95_latency_ms"] = {
            "mean": statistics.mean(valid_p95_times),
            "max": max(valid_p95_times)
        }
    
    return aggregate


def main():
    """Main performance benchmark execution."""
    parser = argparse.ArgumentParser(description="Performance-Only Traffic Monitor Benchmark")
    parser.add_argument("--config", required=True, help="Pipeline configuration YAML file")
    parser.add_argument("--videos", required=True, help="Video list configuration YAML file") 
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument("--profile", action="store_true", help="Enable detailed profiling")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--iterations", type=int, default=1, help="Number of iterations to run")
    parser.add_argument("--warmup", action="store_true", help="Run warmup iteration (not counted)")
    
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
        
        # Add fast mode flag for mock pipeline
        pipeline_config["fast_mode"] = "fast" in str(args.config)
            
        videos = load_performance_videos(Path(args.videos))
        if not videos:
            logger.error("No videos found for performance testing")
            return 1
        
        # Record system info
        system_info = get_system_info()
        logger.info(f"System: {system_info.get('cpu_count', 'Unknown')} CPUs, "
                   f"{system_info.get('memory_total_gb', 0):.1f}GB RAM")
        
        # Warmup run
        if args.warmup:
            logger.info("Running warmup iteration...")
            with profiler.section("warmup"):
                for video_config in videos:
                    run_performance_benchmark(video_config, pipeline_config, profiler)
            profiler.reset()  # Clear warmup data
            logger.info("Warmup completed")
        
        # Main benchmark iterations
        all_iterations = []
        total_start_time = time.time()
        
        for iteration in range(args.iterations):
            logger.info(f"Starting iteration {iteration + 1}/{args.iterations}")
            iteration_start = time.time()
            
            video_results = []
            for video_config in videos:
                result = run_performance_benchmark(video_config, pipeline_config, profiler)
                video_results.append(result)
            
            iteration_time = time.time() - iteration_start
            logger.info(f"Iteration {iteration + 1} completed in {iteration_time:.2f}s")
            
            all_iterations.append({
                "iteration": iteration + 1,
                "iteration_time_seconds": iteration_time,
                "video_results": video_results
            })
        
        total_time = time.time() - total_start_time
        
        # Compute aggregate metrics across all iterations
        all_video_results = []
        for iteration_data in all_iterations:
            all_video_results.extend(iteration_data["video_results"])
        
        aggregate_metrics = compute_aggregate_metrics(all_video_results)
        
        # Get profiler statistics
        profiler_stats = profiler.get_stats()
        
        # Compile final results
        results = {
            "benchmark_info": {
                "type": "performance_only",
                "timestamp": datetime.now().isoformat(),
                "total_time_seconds": total_time,
                "iterations": args.iterations,
                "videos_per_iteration": len(videos),
                "warmup_enabled": args.warmup,
                "system_info": system_info,
                "config_file": str(args.config)
            },
            "performance_metrics": aggregate_metrics,
            "profiler_stats": profiler_stats,
            "iterations": all_iterations
        }
        
        # Save results
        results_path = output_dir / "performance_metrics.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save detailed profiling if enabled
        if args.profile:
            profiler.save_detailed_profile(output_dir / "performance_profiling.csv")
        
        # Print performance summary
        logger.info("=" * 70)
        logger.info("PERFORMANCE BENCHMARK SUMMARY")
        logger.info("=" * 70)
        
        if aggregate_metrics:
            fps_data = aggregate_metrics.get("fps", {})
            if fps_data:
                logger.info(f"Average FPS: {fps_data.get('mean', 0):.1f} ± {fps_data.get('std', 0):.1f}")
                logger.info(f"FPS Range: {fps_data.get('min', 0):.1f} - {fps_data.get('max', 0):.1f}")
            
            frame_time_data = aggregate_metrics.get("frame_time_ms", {})
            if frame_time_data:
                logger.info(f"Average Frame Time: {frame_time_data.get('mean', 0):.1f}ms")
            
            p95_data = aggregate_metrics.get("p95_latency_ms", {})
            if p95_data:
                logger.info(f"P95 Latency: {p95_data.get('mean', 0):.1f}ms (max: {p95_data.get('max', 0):.1f}ms)")
        
        logger.info(f"Total Benchmark Time: {total_time:.1f}s")
        logger.info(f"Results saved to: {results_path}")
        logger.info("=" * 70)
        
        return 0
        
    except Exception as e:
        logger.exception(f"Performance benchmark failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 