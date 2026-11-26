# MachineVision

A machine vision project that integrates the Intel RealSense D435 depth camera with Ultralytics YOLO26 for real-time object detection using Python, OpenCV, and GPU acceleration.

## Features

- Real-time video streaming from the Intel RealSense D435.
- Real-time object detection using YOLO26.
- Object classification, bounding boxes, and confidence scores.
- Color and depth frame acquisition.
- Depth-to-color frame alignment.
- NVIDIA GPU acceleration using PyTorch and CUDA.
- OpenCV-based visualization.

## 1. Prerequisites

Ensure the following are available on your local machine:

- Windows 10 or 11 (64-bit)
- Python 3.13
- Git
- IntelliJ IDEA, PyCharm, or another Python IDE
- Intel RealSense D435 depth camera
- USB 3.0 connection
- NVIDIA GPU with compatible drivers for GPU acceleration

### Required Libraries

| Library | Purpose |
|---|---|
| PyTorch | Neural network inference and GPU acceleration |
| Torchvision | Computer vision utilities |
| Ultralytics | YOLO26 object detection |
| OpenCV | Image processing and visualization |
| NumPy | Image and array manipulation |
| pyrealsense2 | Communication with the RealSense camera |

## 2. Install Python

Download Python 3.13 from:

https://www.python.org/downloads/

During installation, enable **Add Python to PATH** if available.

Verify the installation by opening Command Prompt:

```bash
py -3.13 --version
```

Update pip:

```bash
py -3.13 -m pip install --upgrade pip
```

## 3. Clone the Repository

```bash
git clone <repository-url>
cd MachineVision
```

Replace `<repository-url>` with the URL of this GitHub repository.

## 4. Install PyTorch with GPU Support

Check whether your NVIDIA GPU and drivers are recognized:

```bash
nvidia-smi
```

Install PyTorch and Torchvision with CUDA 12.6 support:

```bash
py -3.13 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

The CUDA 12.6 build was used for the initial development environment. Other GPUs or driver versions may require a different PyTorch build.

Refer to the official PyTorch installation guide if necessary:

https://pytorch.org/get-started/locally/

Verify GPU availability:

```bash
py -3.13 -c "import torch; print(torch.cuda.is_available())"
```

Expected output:

```text
True
```

If the output is `False`, verify that your NVIDIA drivers and PyTorch installation support CUDA.

## 5. Install Project Dependencies

Install Ultralytics:

```bash
py -3.13 -m pip install -U ultralytics
```

Install the RealSense Python SDK:

```bash
py -3.13 -m pip install pyrealsense2
```

Install OpenCV and NumPy:

```bash
py -3.13 -m pip install opencv-python numpy
```

## 6. Configure the Python Interpreter

Ensure your IDE uses the same Python interpreter where the dependencies were installed.

To locate the Python executable, run:

```bash
py -3.13 -c "import sys; print(sys.executable)"
```

### IntelliJ IDEA

1. Navigate to **File > Project Structure > SDKs**.
2. Click the `+` button to add a Python SDK.
3. Choose **Select existing**.
4. Select Python as the interpreter type.
5. Enter the Python executable path obtained above.
6. Assign the interpreter to the project and module.
7. Open **Run > Edit Configurations**.
8. Ensure the run configuration uses the same Python interpreter.
9. Apply the changes.

A separate virtual environment may not have access to packages installed through the system Python interpreter.

## 7. Verify the Installation

Run the following Python script to ensure all required libraries are installed correctly:

```python
import sys
import torch
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

print("Python path:", sys.executable)
print("All imports successful!")
print("GPU available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

Example output:

```text
All imports successful!
GPU available: True
GPU: NVIDIA GeForce RTX 2060
```

## 8. Connect the RealSense Camera

1. Connect the Intel RealSense D435 to a USB 3.0 port.
2. Ensure the camera is recognized by your computer.
3. Close any other applications using the camera, including RealSense Viewer.
4. Run the application.

RealSense SDK documentation:

https://github.com/IntelRealSense/librealsense

## 9. Run the Application

From the root directory of the project:

```bash
py -3.13 machine_vision/src/camera_demo.py
```

The application will:

1. Initialize the pretrained YOLO26 nano model.
2. Start the RealSense color and depth streams.
3. Capture incoming camera frames.
4. Align the depth frames with the color frames.
5. Convert the color frames into NumPy arrays.
6. Perform object detection using YOLO.
7. Display bounding boxes, object labels, and confidence scores.

The pretrained `yolo26n.pt` model will automatically download during the first execution if it is not already available locally.

Press **Q** while the OpenCV window is focused to stop the application.

## 10. System Overview

```text
Intel RealSense D435
        |
        v
Color and Depth Frame Acquisition
        |
        v
Depth-to-Color Alignment
        |
        v
NumPy Image Conversion
        |
        v
YOLO26 Object Detection
        |
        v
Bounding Boxes and Classification
        |
        v
OpenCV Visualization
```

## 11. Camera Configuration

| Setting | Value |
|---|---|
| Color resolution | 640 x 480 |
| Depth resolution | 640 x 480 |
| Camera frame rate | 30 FPS |
| Color format | BGR8 |
| Depth format | Z16 |
| YOLO model | YOLO26 Nano |
| Inference device | NVIDIA GPU (device 0) |

The actual processing frame rate depends on the available hardware and inference speed.

## 12. Troubleshooting

### ModuleNotFoundError

Example:

```text
ModuleNotFoundError: No module named 'ultralytics'
```

Install the missing dependency:

```bash
py -3.13 -m pip install ultralytics
```

If the dependency is already installed, ensure your IDE and run configuration are using the correct Python interpreter.

### CUDA is unavailable

Run:

```bash
nvidia-smi
```

Verify PyTorch GPU support:

```bash
py -3.13 -c "import torch; print(torch.cuda.is_available())"
```

Ensure a CUDA-enabled version of PyTorch is installed.

For CPU-only execution, change the inference device in the source code:

```python
results = model(color_image, device="cpu", verbose=False)
```

CPU inference may be slower than GPU inference.

### RealSense camera is not detected

- Verify that the camera is connected.
- Use a USB 3.0 port.
- Disconnect and reconnect the camera.
- Close other applications using the camera.
- Verify camera functionality using RealSense Viewer.

### Virtual environment or pip errors

If the project environment produces errors involving pip or Python standard libraries, verify the system Python installation:

```bash
py -3.13 -c "import ctypes; print('Python is working')"
```

Configure the IDE to use a working interpreter or recreate the virtual environment using a healthy Python installation.

## 13. Future Development since this is just for testing 

- Integrate YOLO object predictions with depth measurements.
- Filter detected objects based on their distance from the camera.
- Estimate three-dimensional object coordinates.
- Implement real-time object tracking.
- Measure inference latency and processing FPS.
- Improve depth measurement accuracy.
- Train custom object detection models.
- Optimize the system for robotics applications

## References

- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [Ultralytics YOLO26 Tutorial](https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Intel RealSense SDK](https://github.com/IntelRealSense/librealsense)
- [OpenCV Documentation](https://docs.opencv.org/)
- [NumPy Documentation](https://numpy.org/doc/)
