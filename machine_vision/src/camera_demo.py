import sys
import torch
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

print("Python path:", sys.executable)
print("All imports successful!")
print("GPU available:", torch.cuda.is_available())


# Processes an image using YOLO object detection
def process_frame(color_image, model):

    # Apply YOLO object detection using GPU
    results = model(color_image, device=0, verbose=False)

    # Draw bounding boxes and labels
    result = results[0].plot()

    return result


def main():
    # Load the pretrained YOLO model
    model = YOLO("yolo26n.pt")

    # Set up the RealSense camera
    pipeline = rs.pipeline()
    config = rs.config()

    # Configure the color and depth streams
    config.enable_stream(
        rs.stream.color, 640, 480, rs.format.bgr8, 30
    )

    config.enable_stream(
        rs.stream.depth, 640, 480, rs.format.z16, 30
    )

    # Align depth frames with color frames
    align_to_color = rs.align(rs.stream.color)

    # Start the camera
    profile = pipeline.start(config)

    try:
        while True:

            # Get the latest camera frames
            frames = pipeline.wait_for_frames()

            # Align the depth and color frames
            aligned_frames = align_to_color.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            # Skip invalid frames
            if not color_frame or not depth_frame:
                continue

            # Convert frames into NumPy arrays
            color_image = np.asanyarray(
                color_frame.get_data()
            )

            depth_image = np.asanyarray(
                depth_frame.get_data()
            )

            # Process the color frame using YOLO
            result = process_frame(color_image, model)

            # Display the processed frame
            cv2.imshow("YOLO RealSense Detection", result)

            # Press Q to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Release camera resources
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()