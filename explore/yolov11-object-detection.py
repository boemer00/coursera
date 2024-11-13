import cv2
import numpy as np
import time
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")

# Initialize the webcam (using the default camera)
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open webcam.")
    exit()

try:
    while True:
        # Capture frame from the webcam
        ret, frame = camera.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Perform inference
        results = model(frame)

        # Render results on the frame
        for result in results:
            rendered_frame = result.plot()

        # Display the frame with detections
        cv2.imshow('YOLOv11 Real-Time Object Detection', rendered_frame)

        # Exit condition - Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Limit FPS (for older MacBook compatibility)
        time.sleep(0.05)  # ~20 FPS

except KeyboardInterrupt:
    print("Stopped by User")

finally:
    # Release resources
    camera.release()
    cv2.destroyAllWindows()
