"""
Lightweight profiler for E2E benchmarking.
Provides timing, CPU, and GPU resource monitoring.
"""

import time
import psutil
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Any
from loguru import logger

try:
    import pynvml
    NVIDIA_AVAILABLE = True
except ImportError:
    NVIDIA_AVAILABLE = False
    logger.warning("pynvml not available. GPU monitoring disabled.")


class Profiler:
    """Lightweight profiler for benchmarking with timing and resource monitoring."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.timing_stats = defaultdict(list)
        self.resource_stats = defaultdict(list)
        self.current_section = None
        
        if self.enabled and NVIDIA_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_available = True
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                logger.info("GPU monitoring enabled")
            except Exception as e:
                self.gpu_available = False
                logger.warning(f"GPU monitoring disabled: {e}")
        else:
            self.gpu_available = False
    
    @contextmanager
    def section(self, name: str):
        """Time a specific section and collect resource stats."""
        if not self.enabled:
            yield
            return
            
        self.current_section = name
        start_time = time.perf_counter_ns()
        
        # Collect initial resource stats
        self._collect_resources()
        
        try:
            yield
        finally:
            # Collect timing
            duration_ms = (time.perf_counter_ns() - start_time) / 1e6
            self.timing_stats[name].append(duration_ms)
            
            # Collect final resource stats
            end_resources = self._collect_resources()
            
            # Store resource delta or current state
            self.resource_stats[name].append({
                'duration_ms': duration_ms,
                **end_resources
            })
            
            self.current_section = None
    
    def _collect_resources(self) -> Dict[str, Any]:
        """Collect current resource usage."""
        resources = {}
        
        # CPU and Memory
        resources['cpu_percent'] = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        resources['memory_percent'] = memory.percent
        resources['memory_mb'] = memory.used / 1024 / 1024
        
        # GPU resources if available
        if self.gpu_available:
            try:
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                
                resources['gpu_util_percent'] = gpu_util.gpu
                resources['gpu_memory_percent'] = (memory_info.used / memory_info.total) * 100
                resources['gpu_memory_mb'] = memory_info.used / 1024 / 1024
            except Exception as e:
                logger.debug(f"GPU monitoring error: {e}")
        
        return resources
    
    def get_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics."""
        if not self.enabled:
            return {}
            
        stats = {}
        
        # Timing statistics
        for section, times in self.timing_stats.items():
            if times:
                stats[f"{section}_mean_ms"] = sum(times) / len(times)
                stats[f"{section}_p95_ms"] = sorted(times)[int(0.95 * len(times))] if len(times) > 1 else times[0]
                stats[f"{section}_min_ms"] = min(times)
                stats[f"{section}_max_ms"] = max(times)
        
        # Resource statistics  
        for section, resources in self.resource_stats.items():
            if resources:
                # Average resource usage during this section
                for key in ['cpu_percent', 'memory_percent', 'memory_mb', 'gpu_util_percent', 'gpu_memory_percent', 'gpu_memory_mb']:
                    values = [r.get(key, 0) for r in resources if key in r]
                    if values:
                        stats[f"{section}_{key}_avg"] = sum(values) / len(values)
                        stats[f"{section}_{key}_max"] = max(values)
        
        return stats
    
    def save_detailed_profile(self, output_path: Path):
        """Save detailed profiling data to CSV."""
        if not self.enabled:
            return
            
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Combine timing and resource data
        rows = []
        for section in self.timing_stats.keys():
            times = self.timing_stats[section]
            resources = self.resource_stats.get(section, [])
            
            for i, duration in enumerate(times):
                row = {
                    'section': section,
                    'iteration': i,
                    'duration_ms': duration
                }
                if i < len(resources):
                    row.update(resources[i])
                rows.append(row)
        
        # Write to CSV
        if rows:
            import pandas as pd
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
            logger.info(f"Detailed profile saved to {output_path}")
    
    def reset(self):
        """Reset all collected statistics."""
        self.timing_stats.clear()
        self.resource_stats.clear()
        self.current_section = None


@contextmanager 
def time_block(name: str, stats_dict: Dict[str, List[float]]):
    """Simple timing context manager for specific use cases."""
    start = time.perf_counter_ns()
    try:
        yield
    finally:
        duration_ms = (time.perf_counter_ns() - start) / 1e6
        stats_dict[name].append(duration_ms)


def get_system_info() -> Dict[str, Any]:
    """Get system information for benchmark reproducibility."""
    info = {
        'cpu_count': psutil.cpu_count(),
        'memory_total_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
        'python_version': __import__('sys').version,
        'platform': __import__('platform').platform()
    }
    
    if NVIDIA_AVAILABLE:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info['gpu_name'] = pynvml.nvmlDeviceGetName(handle).decode()
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            info['gpu_memory_gb'] = memory_info.total / 1024 / 1024 / 1024
        except Exception:
            pass
    
    return info 