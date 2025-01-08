# Environment Mapping with Object Detection and Point Cloud Generation

This repository contains a Python project for performing environment mapping, object detection, and point cloud generation from a `.mov` video file using OpenCV and YOLOv3. The system processes video frames, detects objects, estimates the camera pose, and generates a 3D point cloud.

## Features
- **Video Processing:** Reads a video file and extracts frames.
- **Object Detection:** Utilizes YOLOv3 for object detection with a pre-trained model.
- **Feature Matching:** ORB feature matching between consecutive frames.
- **Pose Estimation:** Estimates camera pose using essential matrix and RANSAC.
- **3D Point Cloud Generation:** Triangulates 3D points from feature matches.
- **Visualization:** Visualizes the 3D point cloud using Matplotlib.

## Dependencies
- Python 3.8+
- OpenCV
- NumPy
- SciPy
- Matplotlib
- YOLOv3 weights and configuration files

## Setup
1. Clone the repository:
   ```bash
   git clone <repo-link>
   cd <repo-folder>
   ```
2. Install dependencies:
   ```bash
   pip install opencv-python numpy scipy matplotlib
   ```
3. Download YOLOv3 weights and configuration files:
   - Place `yolov3-openimages.cfg`, `yolov3-openimages.weights`, and `openimages.names` in the root directory.

## Usage
```python
from environment import Environment

# Initialize the Environment with a video file
video_path = "path/to/video.mov"
env = Environment(video_path)

# Access detected objects and generated data
objects = env.get_objects()
keypoints = env.get_keypoints()
```

## Example Output
- **Video Output:** Annotated video with detected objects.
- **3D Point Cloud:** Scatter plot visualization of the point cloud.

## Future Improvements
- Implement Open3D for better point cloud visualization.
- Add support for real-time video streaming.
- Improve pose estimation robustness.

---
Created by Varun Trivedi.
