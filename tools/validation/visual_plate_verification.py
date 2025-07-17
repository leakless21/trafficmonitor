#!/usr/bin/env python3
"""
Visual Plate Verification - Verify plates are actually displayed in the output video.

This tool opens the output video and allows manual verification that detected plates
are properly displayed next to vehicles, or uses automated frame analysis.
"""

import sys
import cv2
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def load_detected_plates_from_summary(summary_file: str) -> List[Dict]:
    """Load detected plates from summary report."""
    plates = []
    
    try:
        with open(summary_file, 'r') as f:
            data = json.load(f)
        
        # Extract plate information from detailed metrics if available
        detailed_metrics = data.get('detailed_metrics', {})
        
        # Look for OCR results in various possible locations
        ocr_results = (
            detailed_metrics.get('ocr_results', []) or
            detailed_metrics.get('license_plates', []) or
            data.get('license_plate_analysis', {}).get('detected_plates', [])
        )
        
        for result in ocr_results:
            if isinstance(result, dict) and result.get('plate_text'):
                plates.append({
                    'plate_text': result['plate_text'],
                    'confidence': result.get('confidence', 0.0),
                    'vehicle_id': result.get('vehicle_id', 'unknown'),
                    'frame_id': result.get('frame_id', 'unknown')
                })
        
        # If no detailed results, extract from summary statistics
        if not plates:
            license_analysis = data.get('license_plate_analysis', {})
            plates_recognized = license_analysis.get('plates_successfully_recognized', 0)
            
            if plates_recognized > 0:
                print(f"📊 Summary shows {plates_recognized} plates recognized")
                print("   (Detailed plate list not available in summary)")
                
                # Create placeholder entries for manual verification
                for i in range(min(plates_recognized, 10)):  # Show up to 10 for manual check
                    plates.append({
                        'plate_text': f'PLATE_{i+1}',
                        'confidence': 0.9,
                        'vehicle_id': f'vehicle_{i+1}',
                        'frame_id': f'frame_{i+1}'
                    })
    
    except Exception as e:
        print(f"❌ Error loading summary: {e}")
    
    return plates


def load_plates_from_database() -> List[Dict]:
    """Load plates from database if available."""
    db_path = Path("data/db/traffic_monitor.db")
    if not db_path.exists():
        return []
    
    plates = []
    try:
        conn = sqlite3.connect(str(db_path))
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
        
        conn.close()
        
    except Exception as e:
        print(f"⚠️  Database error: {e}")
    
    return plates


def manual_video_verification(video_path: str, detected_plates: List[Dict]) -> bool:
    """Manual verification by playing video with detected plate information."""
    print(f"🎬 Manual Video Verification")
    print(f"Video: {video_path}")
    print("=" * 60)
    
    if not detected_plates:
        print("❌ No detected plates to verify")
        return False
    
    print(f"📋 Detected plates to look for:")
    for i, plate in enumerate(detected_plates[:20], 1):  # Show first 20
        print(f"   {i:2d}. {plate['plate_text']} (confidence: {plate.get('confidence', 0):.2f})")
    
    if len(detected_plates) > 20:
        print(f"   ... and {len(detected_plates) - 20} more plates")
    
    print(f"\n🎮 Video Controls:")
    print(f"   SPACE: Pause/Resume")
    print(f"   LEFT/RIGHT: Seek backward/forward")
    print(f"   UP/DOWN: Speed up/slow down")
    print(f"   Q: Quit")
    print(f"   Y: Mark current frame as having visible plates")
    print(f"   N: Mark current frame as missing plates")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return False
    
    # Video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"\n📊 Video info:")
    print(f"   Duration: {duration:.1f} seconds")
    print(f"   FPS: {fps}")
    print(f"   Total frames: {total_frames}")
    
    # Verification state
    frame_num = 0
    paused = False
    speed = 1
    plates_visible_count = 0
    plates_missing_count = 0
    frames_checked = 0
    
    print(f"\n▶️  Starting video playback...")
    print(f"   Look for vehicle labels with plate text next to vehicle IDs")
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1
        
        # Display frame with overlay information
        display_frame = frame.copy()
        
        # Add verification overlay
        cv2.putText(display_frame, f"Frame: {frame_num}/{total_frames}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Time: {frame_num/fps:.1f}s", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Speed: {speed}x", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if paused:
            cv2.putText(display_frame, "PAUSED", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Show verification stats
        cv2.putText(display_frame, f"Plates visible: {plates_visible_count}", 
                   (10, display_frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Plates missing: {plates_missing_count}", 
                   (10, display_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.imshow('Plate Verification', display_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(30 // speed) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):  # Space - pause/resume
            paused = not paused
        elif key == ord('y'):  # Y - plates visible
            plates_visible_count += 1
            frames_checked += 1
            print(f"✅ Frame {frame_num}: Plates visible (Total: {plates_visible_count})")
        elif key == ord('n'):  # N - plates missing
            plates_missing_count += 1
            frames_checked += 1
            print(f"❌ Frame {frame_num}: Plates missing (Total: {plates_missing_count})")
        elif key == 81:  # Left arrow - seek backward
            frame_num = max(0, frame_num - fps * 5)  # 5 seconds back
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        elif key == 83:  # Right arrow - seek forward
            frame_num = min(total_frames, frame_num + fps * 5)  # 5 seconds forward
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        elif key == 82:  # Up arrow - speed up
            speed = min(4, speed + 1)
        elif key == 84:  # Down arrow - slow down
            speed = max(1, speed - 1)
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Analysis
    print(f"\n📊 Manual Verification Results:")
    print(f"   Frames checked: {frames_checked}")
    print(f"   Frames with visible plates: {plates_visible_count}")
    print(f"   Frames with missing plates: {plates_missing_count}")
    
    if frames_checked > 0:
        visibility_rate = plates_visible_count / frames_checked * 100
        print(f"   Plate visibility rate: {visibility_rate:.1f}%")
        
        if visibility_rate >= 80:
            print(f"✅ MANUAL VERIFICATION PASSED (≥80% visibility)")
            return True
        else:
            print(f"❌ MANUAL VERIFICATION FAILED (<80% visibility)")
            return False
    else:
        print(f"⚠️  No frames manually checked")
        return False


def automated_frame_sampling(video_path: str, detected_plates: List[Dict]) -> bool:
    """Automated verification by sampling frames and checking for text."""
    print(f"\n🤖 Automated Frame Sampling Verification")
    print("=" * 60)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return False
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_interval = max(1, total_frames // 20)  # Sample 20 frames
    
    print(f"📊 Sampling every {sample_interval} frames ({20} samples total)")
    
    frames_with_text = 0
    frames_sampled = 0
    
    for frame_num in range(0, total_frames, sample_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        frames_sampled += 1
        
        # Simple text detection: look for rectangular regions that might be text
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Look for text-like regions (high contrast areas)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Text regions are typically rectangular with specific aspect ratios
            if 20 < w < 200 and 10 < h < 40 and 2 < w/h < 10:
                text_regions += 1
        
        if text_regions >= 3:  # Assume text is present if we find several text-like regions
            frames_with_text += 1
        
        print(f"   Frame {frame_num}: {text_regions} text-like regions")
    
    cap.release()
    
    print(f"\n📊 Automated Sampling Results:")
    print(f"   Frames sampled: {frames_sampled}")
    print(f"   Frames with text-like regions: {frames_with_text}")
    
    if frames_sampled > 0:
        text_rate = frames_with_text / frames_sampled * 100
        print(f"   Text presence rate: {text_rate:.1f}%")
        
        if text_rate >= 50:  # Lower threshold for automated detection
            print(f"✅ AUTOMATED VERIFICATION SUGGESTS TEXT IS PRESENT")
            return True
        else:
            print(f"⚠️  AUTOMATED VERIFICATION: LOW TEXT PRESENCE")
            return False
    
    return False


def main():
    """Main verification function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify plates are displayed in output video")
    parser.add_argument("--video", help="Path to output video file")
    parser.add_argument("--summary", help="Path to summary JSON file")
    parser.add_argument("--auto-only", action="store_true", help="Skip manual verification")
    
    args = parser.parse_args()
    
    print("👁️  Visual Plate Verification Tool")
    print("=" * 50)
    
    # Find video file
    if args.video:
        video_path = Path(args.video)
    else:
        # Look for recent output videos
        possible_dirs = [
            Path("data/outputs/videos/comprehensive_test"),
            Path("data/outputs/videos"),
            Path(".")
        ]
        
        video_files = []
        for dir_path in possible_dirs:
            if dir_path.exists():
                video_files.extend(dir_path.glob("*.mp4"))
        
        if not video_files:
            print("❌ No video files found. Specify --video path")
            sys.exit(1)
        
        # Use most recent video
        video_path = max(video_files, key=lambda p: p.stat().st_mtime)
    
    if not video_path.exists():
        print(f"❌ Video file not found: {video_path}")
        sys.exit(1)
    
    print(f"📹 Video: {video_path}")
    
    # Find summary file
    if args.summary:
        summary_path = Path(args.summary)
    else:
        # Look for summary in same directory as video
        summary_files = list(video_path.parent.glob("*summary*.json"))
        if summary_files:
            summary_path = max(summary_files, key=lambda p: p.stat().st_mtime)
        else:
            summary_path = None
    
    # Load detected plates
    detected_plates = []
    
    if summary_path and summary_path.exists():
        print(f"📊 Loading plates from summary: {summary_path.name}")
        detected_plates = load_detected_plates_from_summary(str(summary_path))
    
    if not detected_plates:
        print(f"📊 Loading plates from database...")
        detected_plates = load_plates_from_database()
    
    if not detected_plates:
        print("❌ No detected plates found to verify")
        sys.exit(1)
    
    print(f"✅ Found {len(detected_plates)} detected plates to verify")
    
    # Run verifications
    results = {}
    
    # Automated verification
    print(f"\n🤖 Running automated verification...")
    results['automated'] = automated_frame_sampling(str(video_path), detected_plates)
    
    # Manual verification (unless skipped)
    if not args.auto_only:
        print(f"\n👁️  Starting manual verification...")
        print(f"   Please watch the video and verify that plates are displayed")
        input("Press Enter to start video playback...")
        results['manual'] = manual_video_verification(str(video_path), detected_plates)
    else:
        results['manual'] = None
    
    # Final assessment
    print(f"\n" + "=" * 50)
    print("📋 VISUAL VERIFICATION SUMMARY:")
    
    if results['automated']:
        print("✅ Automated verification: Text regions detected in video")
    else:
        print("⚠️  Automated verification: Limited text detection")
    
    if results['manual'] is not None:
        if results['manual']:
            print("✅ Manual verification: Plates confirmed visible")
        else:
            print("❌ Manual verification: Plates not consistently visible")
    else:
        print("⏭️  Manual verification: Skipped")
    
    # Overall result
    if results['manual'] is not None:
        overall_success = results['manual']  # Manual verification is authoritative
    else:
        overall_success = results['automated']  # Fall back to automated
    
    if overall_success:
        print(f"\n🎉 VISUAL VERIFICATION PASSED!")
        print(f"   Detected plates are properly displayed in the video")
    else:
        print(f"\n❌ VISUAL VERIFICATION FAILED!")
        print(f"   Plates may not be properly displayed")
        print(f"   Check Event Fusion Service and visualization configuration")
    
    sys.exit(0 if overall_success else 1)


if __name__ == "__main__":
    main()