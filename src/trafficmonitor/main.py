import cv2
import sys # To exit the program

def run_video_display(video_source):
    """
    Opens a video source and displays it frame by frame.
    """
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"Error: Could not open video source: {video_source}")
        return

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read() # Read a frame

        if not ret:
            print("Info: End of video stream or error reading frame.")
            break

        cv2.imshow("Live Feed", frame) # Display the frame

        # Wait for 1 millisecond, and check if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release() # Release the video capture object
    cv2.destroyAllWindows() # Close all OpenCV windows
    print("Video display finished.")

if __name__ == "__main__":
    # You'll need a video file. For testing, download a short sample video
    # and place it in your project directory or provide the full path.
    # Example: video_path = "sample_traffic.mp4"
    # Or use 0 for webcam: video_path = 0
    video_path = 0 # Try your webcam first! If not, use a video file path.

    # For a video file, make sure it exists or change the path:
    # video_path = "path/to/your/video.mp4"

    print(f"Attempting to open video source: {video_path}")
    run_video_display(video_path)
    print("Program finished.")
    sys.exit(0)