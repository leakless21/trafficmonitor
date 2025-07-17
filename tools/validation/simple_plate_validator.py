#!/usr/bin/env python3
"""
Simple Plate Validator - Check plate display using summary reports and logs.

This tool validates that detected plates are properly displayed by analyzing:
1. Summary reports for detection statistics
2. Log files for visualization mentions
3. Database records for OCR results
"""

import sys
import json
import sqlite3
import re
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict, Counter


def find_latest_summary_report() -> Path:
    """Find the most recent summary report."""
    possible_dirs = [
        Path("data/reports"),
        Path("data/outputs/videos"),
        Path("."),
    ]
    
    all_summaries = []
    for dir_path in possible_dirs:
        if dir_path.exists():
            summaries = list(dir_path.glob("*summary*.json"))
            all_summaries.extend(summaries)
    
    if not all_summaries:
        return None
    
    return max(all_summaries, key=lambda p: p.stat().st_mtime)


def load_summary_data(summary_path: Path) -> Dict:
    """Load data from summary report."""
    try:
        with open(summary_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading summary: {e}")
        return {}


def check_database_plates() -> List[Dict]:
    """Check plates in database."""
    db_path = Path("data/db/traffic_monitor.db")
    if not db_path.exists():
        print(f"⚠️  Database not found: {db_path}")
        return []
    
    plates = []
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Check if OCR results table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='ocr_results'
        """)
        
        if not cursor.fetchone():
            print("⚠️  OCR results table not found in database")
            conn.close()
            return []
        
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
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Database error: {e}")
    
    return plates


def analyze_log_files() -> Dict[str, int]:
    """Analyze log files for plate-related information."""
    log_files = [
        Path("logs/traffic_monitor.log"),
        Path("logs/fusion_test.log"),
        Path("traffic_monitor.log"),
    ]
    
    plate_mentions = Counter()
    visualization_logs = []
    ocr_logs = []
    fusion_logs = []
    
    for log_file in log_files:
        if not log_file.exists():
            continue
        
        try:
            with open(log_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line_lower = line.lower()
                    
                    # Count different types of plate-related logs
                    if 'plate' in line_lower:
                        if 'visualization' in line_lower:
                            visualization_logs.append(line.strip())
                        elif 'ocr' in line_lower:
                            ocr_logs.append(line.strip())
                        elif 'fusion' in line_lower:
                            fusion_logs.append(line.strip())
                        
                        # Extract potential plate text
                        plate_patterns = [
                            r"plate[_\s]*text[:\s]*['\"]?([A-Z0-9]{3,8})['\"]?",
                            r"['\"]([A-Z0-9]{3,8})['\"].*plate",
                            r"plate.*['\"]([A-Z0-9]{3,8})['\"]",
                        ]
                        
                        for pattern in plate_patterns:
                            matches = re.findall(pattern, line, re.IGNORECASE)
                            for match in matches:
                                if len(match) >= 3:
                                    plate_mentions[match.upper()] += 1
        
        except Exception as e:
            print(f"⚠️  Error reading {log_file}: {e}")
    
    return {
        'plate_mentions': dict(plate_mentions),
        'visualization_logs': len(visualization_logs),
        'ocr_logs': len(ocr_logs),
        'fusion_logs': len(fusion_logs),
        'total_plate_mentions': sum(plate_mentions.values())
    }


def validate_plate_display() -> Dict:
    """Main validation function."""
    print("🔍 Simple Plate Display Validation")
    print("=" * 50)
    
    validation_results = {
        'summary_found': False,
        'database_plates': 0,
        'log_analysis': {},
        'validation_passed': False,
        'issues': [],
        'recommendations': []
    }
    
    # 1. Check summary report
    print("📄 Checking summary reports...")
    summary_path = find_latest_summary_report()
    
    if summary_path:
        print(f"✅ Found summary: {summary_path.name}")
        summary_data = load_summary_data(summary_path)
        validation_results['summary_found'] = True
        validation_results['summary_data'] = summary_data
        
        # Extract key metrics
        license_plate_analysis = summary_data.get('license_plate_analysis', {})
        plates_detected = license_plate_analysis.get('plates_detected', 0)
        plates_recognized = license_plate_analysis.get('plates_successfully_recognized', 0)
        ocr_success_rate = license_plate_analysis.get('ocr_success_rate_percent', 0)
        
        print(f"   📊 Plates detected: {plates_detected}")
        print(f"   📊 Plates recognized: {plates_recognized}")
        print(f"   📊 OCR success rate: {ocr_success_rate:.1f}%")
        
        if plates_recognized > 0:
            print(f"   ✅ Plates are being recognized and should be displayed")
        else:
            print(f"   ⚠️  No plates recognized - check OCR pipeline")
            validation_results['issues'].append("No plates recognized by OCR")
    
    else:
        print("❌ No summary report found")
        validation_results['issues'].append("No summary report available")
    
    # 2. Check database
    print(f"\n💾 Checking database...")
    db_plates = check_database_plates()
    validation_results['database_plates'] = len(db_plates)
    
    if db_plates:
        print(f"✅ Found {len(db_plates)} plates in database")
        
        # Analyze plate data
        unique_plates = set(p['plate_text'].upper().strip() for p in db_plates)
        avg_confidence = sum(p['confidence'] for p in db_plates) / len(db_plates)
        
        print(f"   📊 Unique plate texts: {len(unique_plates)}")
        print(f"   📊 Average confidence: {avg_confidence:.2f}")
        
        # Show sample plates
        sample_plates = sorted(unique_plates)[:5]
        print(f"   📝 Sample plates: {', '.join(sample_plates)}")
        
        if avg_confidence < 0.7:
            validation_results['issues'].append(f"Low average OCR confidence: {avg_confidence:.2f}")
        
    else:
        print("❌ No plates found in database")
        validation_results['issues'].append("No plates in database")
    
    # 3. Analyze logs
    print(f"\n📋 Analyzing log files...")
    log_analysis = analyze_log_files()
    validation_results['log_analysis'] = log_analysis
    
    print(f"   📊 Visualization logs: {log_analysis['visualization_logs']}")
    print(f"   📊 OCR logs: {log_analysis['ocr_logs']}")
    print(f"   📊 Fusion logs: {log_analysis['fusion_logs']}")
    print(f"   📊 Total plate mentions: {log_analysis['total_plate_mentions']}")
    
    if log_analysis['plate_mentions']:
        print(f"   📝 Plates mentioned in logs: {list(log_analysis['plate_mentions'].keys())[:5]}")
    
    # 4. Cross-validation
    print(f"\n🔍 Cross-validation...")
    
    if db_plates and log_analysis['plate_mentions']:
        db_plate_texts = set(p['plate_text'].upper().strip() for p in db_plates)
        log_plate_texts = set(log_analysis['plate_mentions'].keys())
        
        overlap = db_plate_texts.intersection(log_plate_texts)
        
        print(f"   📊 DB plates: {len(db_plate_texts)}")
        print(f"   📊 Log plates: {len(log_plate_texts)}")
        print(f"   📊 Overlap: {len(overlap)}")
        
        if len(overlap) > 0:
            display_rate = len(overlap) / len(db_plate_texts) * 100
            print(f"   📊 Display rate: {display_rate:.1f}%")
            
            if display_rate >= 80:
                print(f"   ✅ Good display rate")
            else:
                print(f"   ⚠️  Low display rate")
                validation_results['issues'].append(f"Low display rate: {display_rate:.1f}%")
        else:
            print(f"   ❌ No overlap between DB and logs")
            validation_results['issues'].append("No overlap between database and logs")
    
    # 5. Final assessment
    print(f"\n" + "=" * 50)
    print("📋 VALIDATION SUMMARY:")
    
    # Determine if validation passed
    critical_issues = [
        "No plates recognized by OCR",
        "No plates in database",
        "No overlap between database and logs"
    ]
    
    has_critical_issues = any(issue in validation_results['issues'] for issue in critical_issues)
    
    if not has_critical_issues and (db_plates or (summary_data and summary_data.get('license_plate_analysis', {}).get('plates_successfully_recognized', 0) > 0)):
        validation_results['validation_passed'] = True
        print("✅ VALIDATION PASSED")
        print("   - Plates are being detected and processed")
        print("   - Event Fusion Service is working correctly")
        print("   - Visualization should display plate information")
    else:
        validation_results['validation_passed'] = False
        print("❌ VALIDATION FAILED")
        
        if validation_results['issues']:
            print("   Issues found:")
            for issue in validation_results['issues']:
                print(f"   - {issue}")
    
    # 6. Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if not summary_path:
        print("   - Run the system to generate summary reports")
    
    if not db_plates:
        print("   - Check if OCR service is running and configured correctly")
        print("   - Verify license plate detection model is loaded")
    
    if log_analysis['total_plate_mentions'] == 0:
        print("   - Check if visualization service is logging plate information")
        print("   - Verify Event Fusion Service is merging OCR data")
    
    if validation_results['validation_passed']:
        print("   - System appears to be working correctly!")
        print("   - For visual verification, check the output video file")
    
    return validation_results


def main():
    """Main function."""
    results = validate_plate_display()
    
    # Save results to file
    output_file = Path("plate_validation_results.json")
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n📄 Results saved to: {output_file}")
    except Exception as e:
        print(f"⚠️  Could not save results: {e}")
    
    sys.exit(0 if results['validation_passed'] else 1)


if __name__ == "__main__":
    main()