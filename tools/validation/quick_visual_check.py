#!/usr/bin/env python3
"""
Quick Visual Check - Simple tool to manually verify a few frames show plates.

This tool extracts a few sample frames from the video for quick manual inspection
to verify that plates are displayed next to vehicle IDs.
"""

import sys
import cv2
import json
from pathlib import Path


def extract_sample_frames(video_path: str, output_dir: str, num_samples: int = 5):
    """Extract sample frames from video for manual inspection."""
    print(f"📸 Extracting {num_samples} sample frames from video...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return False
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract frames at regular intervals
    frame_interval = total_frames // (num_samples + 1)
    
    extracted_frames = []
    
    for i in range(1, num_samples + 1):
        frame_num = i * frame_interval
        timestamp = frame_num / fps
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if ret:
            # Save frame
            frame_filename = f"sample_frame_{i:02d}_t{timestamp:.1f}s.jpg"
            frame_path = output_path / frame_filename
            cv2.imwrite(str(frame_path), frame)
            
            extracted_frames.append({
                'filename': frame_filename,
                'frame_number': frame_num,
                'timestamp': timestamp,
                'path': str(frame_path)
            })
            
            print(f"   ✅ Extracted frame {i}: {frame_filename} (t={timestamp:.1f}s)")
        else:
            print(f"   ❌ Failed to extract frame {i}")
    
    cap.release()
    
    if extracted_frames:
        # Create an HTML report for easy viewing
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Plate Display Verification - Sample Frames</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .frame {{ margin: 20px 0; border: 1px solid #ccc; padding: 10px; }}
        .frame img {{ max-width: 800px; height: auto; }}
        .instructions {{ background: #f0f8ff; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .checklist {{ background: #f9f9f9; padding: 15px; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>🔍 Plate Display Verification - Sample Frames</h1>
    
    <div class="instructions">
        <h3>📋 What to Look For:</h3>
        <ul>
            <li><strong>Vehicle Labels:</strong> Look for text above or near vehicles showing class name and ID</li>
            <li><strong>Plate Text:</strong> Check if license plate text appears next to vehicle ID (e.g., "car 42 ABC123")</li>
            <li><strong>Color Coding:</strong> 
                <ul>
                    <li>🟢 <strong>Green text:</strong> Successful OCR (plate text read)</li>
                    <li>🟡 <strong>Yellow text:</strong> Plate detected but not read</li>
                    <li>🔴 <strong>Red text:</strong> No plate detected</li>
                </ul>
            </li>
            <li><strong>Plate Bounding Boxes:</strong> Yellow rectangles around detected license plates</li>
        </ul>
    </div>
    
    <div class="checklist">
        <h3>✅ Verification Checklist:</h3>
        <p>For each frame below, check:</p>
        <ul>
            <li>□ Are vehicles properly labeled with class and ID?</li>
            <li>□ Do any vehicle labels include plate text?</li>
            <li>□ Are there yellow bounding boxes around license plates?</li>
            <li>□ Is the text color-coded based on detection status?</li>
        </ul>
    </div>
"""
        
        for frame_info in extracted_frames:
            html_content += f"""
    <div class="frame">
        <h3>Frame {frame_info['frame_number']} (t={frame_info['timestamp']:.1f}s)</h3>
        <img src="{frame_info['filename']}" alt="Frame {frame_info['frame_number']}">
        <p><strong>File:</strong> {frame_info['filename']}</p>
        <p><strong>Timestamp:</strong> {frame_info['timestamp']:.1f} seconds</p>
    </div>
"""
        
        html_content += """
    <div class="instructions">
        <h3>📊 Expected Results:</h3>
        <p>If the Event Fusion Service is working correctly, you should see:</p>
        <ul>
            <li>Vehicle labels that include license plate text when available</li>
            <li>Color-coded text indicating detection status</li>
            <li>Yellow bounding boxes around detected license plates</li>
            <li>Consistent labeling across frames</li>
        </ul>
        
        <h3>🚨 Red Flags:</h3>
        <ul>
            <li>All vehicle labels are red (no plates detected anywhere)</li>
            <li>No license plate text appears in any labels</li>
            <li>No yellow bounding boxes around visible license plates</li>
            <li>Inconsistent or missing vehicle labels</li>
        </ul>
    </div>
</body>
</html>
"""
        
        html_path = output_path / "verification_report.html"
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        print(f"\n📄 Created verification report: {html_path}")
        print(f"   Open this file in a web browser to inspect the frames")
        
        return True
    
    return False


def load_plate_statistics():
    """Load plate statistics from summary report."""
    summary_files = list(Path("data/outputs/videos/comprehensive_test").glob("*summary*.json"))
    if not summary_files:
        return None
    
    latest_summary = max(summary_files, key=lambda p: p.stat().st_mtime)
    
    try:
        with open(latest_summary, 'r') as f:
            data = json.load(f)
        
        license_analysis = data.get('license_plate_analysis', {})
        return {
            'plates_detected': license_analysis.get('plates_detected', 0),
            'plates_recognized': license_analysis.get('plates_successfully_recognized', 0),
            'success_rate': license_analysis.get('ocr_success_rate_percent', 0),
            'total_frames': data.get('performance_metrics', {}).get('frames_processed', 0),
            'total_vehicles': data.get('vehicle_analysis', {}).get('total_vehicles_tracked', 0)
        }
    except Exception as e:
        print(f"⚠️  Error loading summary: {e}")
        return None


def main():
    """Main function."""
    print("👁️  Quick Visual Check Tool")
    print("Extracts sample frames for manual plate display verification")
    print("=" * 60)
    
    # Find video file
    video_files = list(Path("data/outputs/videos/comprehensive_test").glob("*.mp4"))
    if not video_files:
        print("❌ No output video found")
        sys.exit(1)
    
    video_path = max(video_files, key=lambda p: p.stat().st_mtime)
    print(f"📹 Video: {video_path}")
    
    # Load statistics
    stats = load_plate_statistics()
    if stats:
        print(f"📊 Processing Statistics:")
        print(f"   - Total frames: {stats['total_frames']}")
        print(f"   - Total vehicles: {stats['total_vehicles']}")
        print(f"   - Plates detected: {stats['plates_detected']}")
        print(f"   - Plates recognized: {stats['plates_recognized']}")
        print(f"   - OCR success rate: {stats['success_rate']:.1f}%")
        
        if stats['plates_recognized'] > 0:
            print(f"   ✅ Plates were recognized - should be visible in frames")
        else:
            print(f"   ⚠️  No plates recognized - frames may not show plate text")
    
    # Extract sample frames
    output_dir = "data/outputs/videos/comprehensive_test/sample_frames"
    success = extract_sample_frames(str(video_path), output_dir, num_samples=8)
    
    if success:
        print(f"\n✅ Sample frames extracted successfully!")
        print(f"📁 Location: {output_dir}")
        print(f"📄 Open verification_report.html in a web browser to inspect frames")
        
        print(f"\n🔍 Manual Verification Steps:")
        print(f"   1. Open the HTML report in your browser")
        print(f"   2. Look for vehicle labels with license plate text")
        print(f"   3. Check for color-coded text (green = OCR success)")
        print(f"   4. Verify yellow bounding boxes around license plates")
        print(f"   5. Confirm consistent labeling across frames")
        
        if stats and stats['plates_recognized'] > 0:
            print(f"\n🎯 Expected: You should see license plate text in vehicle labels")
            print(f"   With {stats['plates_recognized']} plates recognized, some frames should show plate text")
        else:
            print(f"\n⚠️  Note: No plates were recognized, so you may not see plate text")
            print(f"   But you should still see proper vehicle labeling and bounding boxes")
        
        sys.exit(0)
    else:
        print(f"❌ Failed to extract sample frames")
        sys.exit(1)


if __name__ == "__main__":
    main()