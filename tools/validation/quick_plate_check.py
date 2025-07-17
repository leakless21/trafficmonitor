#!/usr/bin/env python3
"""
Quick Plate Check - Fast validation of plate display without OCR.

This tool provides a lightweight way to verify plate display by checking
the database records against the visualization service logs.
"""

import sys
import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

def get_db_connection():
    """Get database connection."""
    import sqlite3
    return sqlite3.connect("data/db/traffic_monitor.db")


def load_plates_from_db() -> List[Dict]:
    """Load all detected plates from database."""
    plates = []
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT frame_id, vehicle_id, plate_text, confidence, timestamp
                FROM ocr_results 
                WHERE plate_text IS NOT NULL AND plate_text != ''
                ORDER BY timestamp
            """)
            
            for row in cursor.fetchall():
                frame_id, vehicle_id, plate_text, confidence, timestamp = row
                plates.append({
                    'frame_id': frame_id,
                    'vehicle_id': vehicle_id,
                    'plate_text': plate_text,
                    'confidence': confidence,
                    'timestamp': timestamp
                })
                
    except Exception as e:
        print(f"❌ Error loading plates from database: {e}")
        return []
    
    return plates


def load_plates_from_logs(log_file: str = "logs/traffic_monitor.log") -> Set[str]:
    """Extract plate texts from visualization service logs."""
    displayed_plates = set()
    
    log_path = Path(log_file)
    if not log_path.exists():
        print(f"⚠️  Log file not found: {log_file}")
        return displayed_plates
    
    try:
        with open(log_path, 'r') as f:
            for line in f:
                # Look for visualization service logs mentioning plates
                if "VisualizationService" in line and "plate" in line.lower():
                    # Extract plate text from log line
                    # This is a simple heuristic - adjust based on actual log format
                    if "text:" in line:
                        parts = line.split("text:")
                        if len(parts) > 1:
                            plate_text = parts[1].split()[0].strip("'\"(),")
                            if plate_text and len(plate_text) >= 3:
                                displayed_plates.add(plate_text.upper())
                                
    except Exception as e:
        print(f"⚠️  Error reading log file: {e}")
    
    return displayed_plates


def check_summary_report(report_dir: str = "data/reports") -> Dict:
    """Check the latest summary report for plate statistics."""
    reports_path = Path(report_dir)
    
    if not reports_path.exists():
        return {}
    
    # Find the most recent summary report
    summary_files = list(reports_path.glob("*summary*.json"))
    if not summary_files:
        return {}
    
    latest_report = max(summary_files, key=lambda p: p.stat().st_mtime)
    
    try:
        with open(latest_report, 'r') as f:
            data = json.load(f)
        
        return {
            'file': latest_report.name,
            'plates_detected': data.get('license_plate_analysis', {}).get('plates_detected', 0),
            'plates_recognized': data.get('license_plate_analysis', {}).get('plates_successfully_recognized', 0),
            'ocr_success_rate': data.get('license_plate_analysis', {}).get('ocr_success_rate_percent', 0),
            'processing_time': data.get('performance_metrics', {}).get('processing_duration_seconds', 0),
            'total_frames': data.get('performance_metrics', {}).get('frames_processed', 0)
        }
        
    except Exception as e:
        print(f"⚠️  Error reading summary report: {e}")
        return {}


def quick_validation_check():
    """Perform quick validation check."""
    print("🚀 Quick Plate Display Validation")
    print("=" * 50)
    
    # Load plates from database
    print("📊 Loading detected plates from database...")
    db_plates = load_plates_from_db()
    
    if not db_plates:
        print("❌ No plates found in database")
        return False
    
    print(f"✅ Found {len(db_plates)} plates in database")
    
    # Group by vehicle for analysis
    plates_by_vehicle = defaultdict(list)
    unique_plates = set()
    
    for plate in db_plates:
        plates_by_vehicle[plate['vehicle_id']].append(plate)
        unique_plates.add(plate['plate_text'].upper().strip())
    
    print(f"📈 Statistics:")
    print(f"   - Unique vehicles with plates: {len(plates_by_vehicle)}")
    print(f"   - Unique plate texts: {len(unique_plates)}")
    print(f"   - Average confidence: {sum(p['confidence'] for p in db_plates) / len(db_plates):.2f}")
    
    # Show sample plates
    print(f"\n📝 Sample detected plates:")
    for i, plate_text in enumerate(sorted(unique_plates)[:10]):
        print(f"   {i+1:2d}. {plate_text}")
    
    if len(unique_plates) > 10:
        print(f"   ... and {len(unique_plates) - 10} more")
    
    # Check summary report
    print(f"\n📄 Checking summary report...")
    summary_data = check_summary_report()
    
    if summary_data:
        print(f"✅ Latest report: {summary_data['file']}")
        print(f"   - Plates detected: {summary_data['plates_detected']}")
        print(f"   - Plates recognized: {summary_data['plates_recognized']}")
        print(f"   - OCR success rate: {summary_data['ocr_success_rate']:.1f}%")
        print(f"   - Total frames: {summary_data['total_frames']}")
        
        # Validate consistency
        db_count = len(db_plates)
        report_count = summary_data['plates_recognized']
        
        if db_count == report_count:
            print(f"✅ Database and report counts match: {db_count}")
        else:
            print(f"⚠️  Count mismatch - DB: {db_count}, Report: {report_count}")
    else:
        print("⚠️  No summary report found")
    
    # Check logs for displayed plates
    print(f"\n📋 Checking visualization logs...")
    displayed_plates = load_plates_from_logs()
    
    if displayed_plates:
        print(f"✅ Found {len(displayed_plates)} plates mentioned in logs")
        
        # Check overlap with detected plates
        detected_set = set(p['plate_text'].upper().strip() for p in db_plates)
        overlap = detected_set.intersection(displayed_plates)
        
        print(f"📊 Overlap analysis:")
        print(f"   - Detected plates: {len(detected_set)}")
        print(f"   - Displayed plates: {len(displayed_plates)}")
        print(f"   - Overlap: {len(overlap)}")
        print(f"   - Display rate: {len(overlap) / len(detected_set) * 100:.1f}%")
        
        missing_from_display = detected_set - displayed_plates
        if missing_from_display:
            print(f"⚠️  Plates not found in logs: {list(missing_from_display)[:5]}")
        
    else:
        print("⚠️  No plates found in visualization logs")
    
    # Final assessment
    print(f"\n" + "=" * 50)
    
    if summary_data and summary_data['plates_recognized'] > 0:
        success_rate = summary_data['ocr_success_rate']
        if success_rate >= 80:
            print(f"✅ VALIDATION PASSED - {success_rate:.1f}% OCR success rate")
            return True
        else:
            print(f"⚠️  VALIDATION WARNING - {success_rate:.1f}% OCR success rate (below 80%)")
            return False
    else:
        print("❌ VALIDATION FAILED - No successful plate recognition found")
        return False


def main():
    """Main function."""
    success = quick_validation_check()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()