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
git clone https://github.com/kaesotiric/MachineVision.git
cd MachineVision
```

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

If the output is `False`, verify that your NVIDIA drivers and