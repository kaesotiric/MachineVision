import cv2
import numpy as np
import pyrealsense2 as rs

# Draws a green box around a nearby object

# def process_frame(color_image, depth_meters):
#     # Keep pixels w a valid depth reading up to 1 metre away
#     nearby = (depth_meters > 0) & (depth_meters <= 1.0)
#
#     result = color_image.copy()
#     result[~nearby] = 0  # Turn everything outside that range black
#     return result
def process_frame(color_image, depth_meters):
    # Select pixels between 25 cm and 1 metre from the camera.
    nearby = (depth_meters >= 0.25) & (depth_meters <= 1.0)

    # Make a black-and-white image: nearby pixels are white (255).
    mask = nearby.astype(np.uint8) * 255

    # Find the outer outline of each white region.
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    result = color_image.copy()

    for contour in contours:
        if cv2.contourArea(contour) >= 1000:  # Ignore small specks.
            x, y, width, height = cv2.boundingRect(contour)
            cv2.rectangle(result, (x, y), (x + width, y + height), (0, 255, 0), 2)

    return result

def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    align_to_color = rs.align(rs.stream.color)
    profile = pipeline.start(config)

    try:
        sensor = profile.get_device().first_depth_sensor()
        depth_scale = sensor.get_depth_scale()

        while True:
            frames = pipeline.wait_for_frames()
            aligned = align_to_color.process(frames)

            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())
            depth_meters = depth_raw.astype(np.float32) * depth_scale

            result = process_frame(color_image, depth_meters)
            cv2.imshow("Live color | Within 1 metre", np.hstack((color_image, result)))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()