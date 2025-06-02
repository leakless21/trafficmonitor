#!/usr/bin/env python3
import cv2
import time
import argparse
from ultralytics import YOLO
import onnxruntime as ort
import os

# Get the absolute path to the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up two levels to the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

def main():
    parser = argparse.ArgumentParser(description="Traffic monitoring with YOLO")
    parser.add_argument('--source', type=str, default=0, 
                        help='Source for detection (0 for webcam, or path to video file)')
    parser.add_argument('--model', type=str, default=os.path.join(PROJECT_ROOT, 'data/models/yolo11s.onnx'),
                        help='Model path (ONNX or PT)')
    parser.add_argument('--plate-model', type=str, default=os.path.join(PROJECT_ROOT, 'data/models/plate_v8n.pt'),
                        help='Path to plate detection model (used when --plate is specified)')
    parser.add_argument('--conf', type=float, default=0.25, 
                        help='Confidence threshold for detections')
    parser.add_argument('--plate', action='store_true',
                        help='Use plate detection model (shortcut for using the plate model)')
    args = parser.parse_args()

    # Apply the plate shortcut if specified
    if args.plate:
        args.model = args.plate_model
        print(f"Using plate detection model: {args.model}")
        
    # Construct absolute path for model if a relative path was given via CLI
    if not os.path.isabs(args.model):
        args.model = os.path.join(PROJECT_ROOT, args.model)
        print(f"Resolved model path to: {args.model}")
    
    # Define the classes we're interested in (for COCO dataset)
    target_classes = ['person', 'car', 'motorcycle', 'bus', 'truck']
    
    # Load the model (will automatically handle PT or ONNX format)
    if args.model.endswith('.onnx'):
        print(f"Loading ONNX model: {args.model}")
        # Just verify ONNX model can be loaded
        ort_session = ort.InferenceSession(args.model)
        # But we'll use the YOLO wrapper for consistent API
        model = YOLO(args.model)
    else:
        print(f"Loading PyTorch model: {args.model}")
        model = YOLO(args.model)
    
    # Get model class names mapping
    class_names = model.names if hasattr(model, 'names') else None
    if not class_names:
        if 'plate' in args.model:
            # Plate model only has one class
            class_names = {0: 'plate'}
        else:
            # Fallback COCO class names if not available in model
            class_names = {0: 'person', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
    
    # Print all available classes in the model for debugging
    print(f"Model class names: {class_names}")
    
    # If this is a plate model (either detected from name or specified by arg)
    if 'plate' in args.model or args.plate:
        # For plate detection, we want to show all detected plates
        target_class_ids = list(class_names.keys())
        print("Plate detection mode: Showing all detected plates")
    else:
        # For COCO model, filter for target transportation classes
        target_class_ids = [i for i, name in class_names.items() if name in target_classes]
    
    print(f"Monitoring for classes: {[class_names[i] for i in target_class_ids if i in class_names]}")
    
    # Start video capture
    source = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video source: {width}x{height} @ {fps}fps")
    
    # FPS calculation variables
    frame_count = 0
    start_time = time.time()
    fps_display = 0
    
    print("Press 'q' to quit")
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("End of video stream")
            break
        
        # Run inference
        results = model(frame, conf=args.conf)
        
        # Process results
        for result in results:
            boxes = result.boxes.cpu().numpy()
            
            for box in boxes:
                # Get class ID, confidence and box coordinates
                cls_id = int(box.cls[0])
                
                # Check if this class should be displayed
                if cls_id not in target_class_ids and cls_id not in class_names:
                    continue
                    
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add class name and confidence
                class_name = class_names.get(cls_id, f"class_{cls_id}")
                label = f"{class_name} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Calculate and display FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        
        # Update FPS every second
        if elapsed_time >= 1.0:
            fps_display = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
        
        # Display FPS on frame
        cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display the frame
        cv2.imshow('Traffic Monitor', frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()