"""
End-to-End Evaluation Logic for Traffic Monitor Benchmark.

This module evaluates the complete pipeline by matching predicted events
against ground truth and computing system-level metrics.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class VehicleEvent:
    """Represents a vehicle passing event."""
    track_id: int
    plate: Optional[str]
    ts_enter: float
    ts_exit: float
    vehicle_class: str
    confidence: float = 1.0


@dataclass 
class QueueEvent:
    """Represents a queue length measurement."""
    timestamp: float
    length: int


@dataclass
class EvaluationMetrics:
    """Container for all E2E evaluation metrics."""
    # Vehicle identification metrics
    vehicle_precision: float
    vehicle_recall: float
    vehicle_f1: float
    
    # Plate recognition metrics  
    plate_precision: float
    plate_recall: float
    plate_f1: float
    
    # Counting metrics
    count_mae: float
    count_rmse: float
    count_smape: float
    
    # Queue metrics
    queue_mae: float
    queue_rmse: float
    
    # Combined metrics
    overall_f1: float
    
    # Timing metrics (added by profiler)
    mean_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    fps: float = 0.0


class E2EEvaluator:
    """End-to-end evaluator for the traffic monitoring system."""

    def __init__(self, iou_threshold: float = 0.5, temporal_threshold: float = 1.0):
        """
        Initialize evaluator.
        
        Args:
            iou_threshold: IoU threshold for spatial matching
            temporal_threshold: Time threshold for temporal matching (seconds)
        """
        self.iou_threshold = iou_threshold
        self.temporal_threshold = temporal_threshold
    
    def load_ground_truth(self, gt_path: Path) -> Tuple[List[VehicleEvent], List[QueueEvent]]:
        """Load ground truth events from JSON file."""
        try:
            with open(gt_path, 'r') as f:
                events = json.load(f)
            
            vehicle_events = []
            queue_events = []
            
            for event in events:
                if event['event'] == 'VehiclePassed':
                    vehicle_events.append(VehicleEvent(
                        track_id=event['track_id'],
                        plate=event.get('plate'),
                        ts_enter=event['ts_enter'],
                        ts_exit=event['ts_exit'],
                        vehicle_class=event['vehicle_class'],
                        confidence=event.get('confidence', 1.0)
                    ))
                elif event['event'] == 'QueueLength':
                    queue_events.append(QueueEvent(
                        timestamp=event['timestamp'],
                        length=event['length']
                    ))
            
            logger.info(f"Loaded {len(vehicle_events)} vehicle events and {len(queue_events)} queue events")
            return vehicle_events, queue_events
            
        except Exception as e:
            logger.error(f"Failed to load ground truth from {gt_path}: {e}")
            return [], []
    
    def load_predictions(self, pred_path: Path) -> Tuple[List[VehicleEvent], List[QueueEvent]]:
        """Load predicted events from JSON file."""
        # Same format as ground truth for now
        return self.load_ground_truth(pred_path)
    
    def match_vehicles(self, gt_vehicles: List[VehicleEvent], pred_vehicles: List[VehicleEvent]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match predicted vehicles to ground truth vehicles.
        
        Returns:
            matches: List of (gt_idx, pred_idx) pairs
            unmatched_gt: List of unmatched ground truth indices  
            unmatched_pred: List of unmatched prediction indices
        """
        matches = []
        unmatched_gt = list(range(len(gt_vehicles)))
        unmatched_pred = list(range(len(pred_vehicles)))
        
        # Simple temporal matching for now (can be enhanced with spatial IoU)
        for i, gt_vehicle in enumerate(gt_vehicles):
            best_match = None
            best_overlap = 0.0
            
            for j, pred_vehicle in enumerate(pred_vehicles):
                if j not in unmatched_pred:
                    continue
                    
                # Calculate temporal overlap
                overlap = self._temporal_overlap(gt_vehicle, pred_vehicle)
                if overlap > best_overlap and overlap > self.temporal_threshold:
                    best_match = j
                    best_overlap = overlap
            
            if best_match is not None:
                matches.append((i, best_match))
                unmatched_gt.remove(i)
                unmatched_pred.remove(best_match)
        
        return matches, unmatched_gt, unmatched_pred
    
    def _temporal_overlap(self, gt_vehicle: VehicleEvent, pred_vehicle: VehicleEvent) -> float:
        """Calculate temporal overlap between two vehicle events."""
        gt_start = gt_vehicle.ts_enter
        gt_end = gt_vehicle.ts_exit
        pred_start = pred_vehicle.ts_enter
        pred_end = pred_vehicle.ts_exit

        # Inline local extrema for interval endpoints
        min_end = gt_end if gt_end < pred_end else pred_end
        max_start = gt_start if gt_start > pred_start else pred_start
        intersection = min_end - max_start
        if intersection <= 0:
            # Early exit if no overlap at all
            return 0.0
        max_end = gt_end if gt_end > pred_end else pred_end
        min_start = gt_start if gt_start < pred_start else pred_start
        union = max_end - min_start
        # No need to check union > 0, since intersection > 0 guarantees union > 0
        return intersection / union
    
    def evaluate_vehicle_identification(self, gt_vehicles: List[VehicleEvent], pred_vehicles: List[VehicleEvent]) -> Dict[str, float]:
        """Evaluate vehicle identification performance."""
        matches, unmatched_gt, unmatched_pred = self.match_vehicles(gt_vehicles, pred_vehicles)
        
        # Calculate precision, recall, F1
        true_positives = len(matches)
        false_positives = len(unmatched_pred)
        false_negatives = len(unmatched_gt)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall, 
            'f1': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    def evaluate_plate_recognition(self, gt_vehicles: List[VehicleEvent], pred_vehicles: List[VehicleEvent]) -> Dict[str, float]:
        """Evaluate license plate recognition performance."""
        matches, _, _ = self.match_vehicles(gt_vehicles, pred_vehicles)
        
        plate_correct = 0
        plate_total = 0
        gt_with_plates = 0
        pred_with_plates = 0
        
        # Count ground truth vehicles with plates
        for vehicle in gt_vehicles:
            if vehicle.plate is not None:
                gt_with_plates += 1
        
        # Evaluate matched vehicles
        for gt_idx, pred_idx in matches:
            gt_vehicle = gt_vehicles[gt_idx]
            pred_vehicle = pred_vehicles[pred_idx]
            
            if gt_vehicle.plate is not None:
                plate_total += 1
                if pred_vehicle.plate is not None and pred_vehicle.plate == gt_vehicle.plate:
                    plate_correct += 1
            
            if pred_vehicle.plate is not None:
                pred_with_plates += 1
        
        # Calculate plate-specific metrics
        plate_precision = plate_correct / pred_with_plates if pred_with_plates > 0 else 0.0
        plate_recall = plate_correct / gt_with_plates if gt_with_plates > 0 else 0.0  
        plate_f1 = 2 * plate_precision * plate_recall / (plate_precision + plate_recall) if (plate_precision + plate_recall) > 0 else 0.0
        
        return {
            'precision': plate_precision,
            'recall': plate_recall,
            'f1': plate_f1,
            'correct': plate_correct,
            'total_gt': gt_with_plates,
            'total_pred': pred_with_plates
        }
    
    def evaluate_counting(self, gt_vehicles: List[VehicleEvent], pred_vehicles: List[VehicleEvent]) -> Dict[str, float]:
        """Evaluate vehicle counting performance."""
        gt_count = len(gt_vehicles)
        pred_count = len(pred_vehicles)
        
        mae = abs(gt_count - pred_count)
        rmse = (gt_count - pred_count) ** 2
        smape = 2 * abs(gt_count - pred_count) / (gt_count + pred_count) * 100 if (gt_count + pred_count) > 0 else 0.0
        
        return {
            'mae': mae,
            'rmse': rmse,
            'smape': smape,
            'gt_count': gt_count,
            'pred_count': pred_count
        }
    
    def evaluate_queue_length(self, gt_queue: List[QueueEvent], pred_queue: List[QueueEvent]) -> Dict[str, float]:
        """Evaluate queue length estimation performance."""
        if not gt_queue or not pred_queue:
            return {'mae': 0.0, 'rmse': 0.0}
        
        # Match queue events by timestamp (simple nearest neighbor)
        errors = []
        for gt_event in gt_queue:
            # Find closest prediction in time
            closest_pred = min(pred_queue, key=lambda p: abs(p.timestamp - gt_event.timestamp))
            if abs(closest_pred.timestamp - gt_event.timestamp) <= 2.0:  # Within 2 seconds
                errors.append(abs(gt_event.length - closest_pred.length))
        
        if not errors:
            return {'mae': float('inf'), 'rmse': float('inf')}
        
        mae = np.mean(errors)
        rmse = np.sqrt(np.mean([e**2 for e in errors]))
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'matched_events': float(len(errors))
        }
    
    def evaluate(self, gt_path: Path, pred_path: Path) -> EvaluationMetrics:
        """
        Perform complete E2E evaluation.
        
        Args:
            gt_path: Path to ground truth events JSON
            pred_path: Path to predicted events JSON
            
        Returns:
            EvaluationMetrics object with all computed metrics
        """
        # Load data
        gt_vehicles, gt_queue = self.load_ground_truth(gt_path)
        pred_vehicles, pred_queue = self.load_predictions(pred_path)
        
        # Evaluate components
        vehicle_metrics = self.evaluate_vehicle_identification(gt_vehicles, pred_vehicles)
        plate_metrics = self.evaluate_plate_recognition(gt_vehicles, pred_vehicles)
        count_metrics = self.evaluate_counting(gt_vehicles, pred_vehicles)
        queue_metrics = self.evaluate_queue_length(gt_queue, pred_queue)
        
        # Calculate overall F1 (weighted average of vehicle and plate F1)
        overall_f1 = (vehicle_metrics['f1'] + plate_metrics['f1']) / 2
        
        return EvaluationMetrics(
            vehicle_precision=vehicle_metrics['precision'],
            vehicle_recall=vehicle_metrics['recall'],
            vehicle_f1=vehicle_metrics['f1'],
            plate_precision=plate_metrics['precision'],
            plate_recall=plate_metrics['recall'],
            plate_f1=plate_metrics['f1'],
            count_mae=count_metrics['mae'],
            count_rmse=count_metrics['rmse'],
            count_smape=count_metrics['smape'],
            queue_mae=queue_metrics['mae'],
            queue_rmse=queue_metrics['rmse'],
            overall_f1=overall_f1
        )
    
    def save_detailed_results(self, metrics: EvaluationMetrics, output_path: Path):
        """Save detailed evaluation results to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results = {
            'vehicle_identification': {
                'precision': metrics.vehicle_precision,
                'recall': metrics.vehicle_recall,
                'f1': metrics.vehicle_f1
            },
            'plate_recognition': {
                'precision': metrics.plate_precision,
                'recall': metrics.plate_recall,
                'f1': metrics.plate_f1
            },
            'counting': {
                'mae': metrics.count_mae,
                'rmse': metrics.count_rmse,
                'smape': metrics.count_smape
            },
            'queue_length': {
                'mae': metrics.queue_mae,
                'rmse': metrics.queue_rmse
            },
            'overall': {
                'f1': metrics.overall_f1
            },
            'timing': {
                'mean_latency_ms': metrics.mean_latency_ms,
                'p95_latency_ms': metrics.p95_latency_ms,
                'fps': metrics.fps
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Detailed results saved to {output_path}") 