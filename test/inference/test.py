import cv2
import time

from ultralytics import YOLO

# Load the YOLO model
model = YOLO("best(8).pt")

# Open the video file
video_path = "IMG_3311.MOV"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Start time for FPS calculation
        start_time = time.time()

        # Run YOLO inference on the frame
        results = model(frame, conf=0.6)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Calculate FPS
        end_time = time.time()
        fps = 1 / (end_time - start_time)

        # Display FPS on the frame
        cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the annotated frame
        cv2.imshow("YOLO Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()