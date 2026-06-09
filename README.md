# Drone Detection & Tracking with YOLOv8 + Kalman Filter

A computer vision pipeline that detects drones in video footage using a fine-tuned YOLOv8 model and tracks their trajectory frame-by-frame using a Kalman filter. Built as a graduate project for DS681 at NJIT Ying Wu College of Computing.

## Results

| Metric | Score |
|--------|-------|
| F1 Score (test set) | **0.91** |
| Base model | YOLOv8s |
| Training dataset | [Seraphim Drone Detection Dataset](https://huggingface.co/datasets/lgrzybowski/seraphim-drone-detection-dataset) |

## Demo

The pipeline processes raw video, runs the detector on every frame, and overlays bounding boxes and a tracked trajectory polyline on each detection.

![Drone detection with trajectory tracking overlay]

> Green box: YOLOv8 detection bounding box  
> Red line: Kalman filter estimated trajectory

## Overview

The system has two stages:

**Stage 1 — Detection**  
YOLOv8s fine-tuned on a drone detection dataset processes each video frame. Frames containing at least one detection are saved for tracking.

**Stage 2 — Tracking**  
A Kalman filter with state vector `[x, y, vx, vy]` and measurement vector `[x, y]` smooths the detections into a continuous trajectory. The filter is initialized on the first detection and updated on each subsequent frame.

## Approach

**Model selection:** YOLOv8s was chosen as a balance between inference speed and accuracy — large enough to generalize well, small enough to train overnight on a laptop GPU.

**Dataset selection:** The [Seraphim dataset](https://huggingface.co/datasets/lgrzybowski/seraphim-drone-detection-dataset) was selected from Hugging Face as the largest available drone image corpus. Finding images *of* drones (rather than images *from* drones) proved to be a meaningful constraint — larger training corpus was prioritized to maximize fine-tuning performance.

**Kalman filter design:**
```
State:       [x, y, vx, vy]   — position and velocity
Measurement: [x, y]           — bounding box center
Init state:  [x0, y0, 0, 0]   — first detection, assuming drone at rest
R (measurement noise): I * 5  — 5 pixel measurement uncertainty
Q (process noise):     I * 0.1
P (initial covariance): I * 1000
```

## Failure Analysis

Honest assessment of where the system breaks down:

**Out-of-frame reacquisition:** When the drone leaves and re-enters the frame, the Kalman filter draws a straight-line trajectory between the last known position and the new detection. This is geometrically incorrect but the filter reacquires the target quickly.

**Background confusion:** Against a dynamic sky background, cloud voids with similar contrast to a drone cause false detections. When the drone exits frame, the tracker latches onto the nearest cloud void. When the drone reappears, the tracker snaps back — but the interval produces spurious trajectory segments.

**Distance degradation:** Detection reliability decreases as the drone moves farther from the camera, which is expected given the training distribution.

**Root cause:** The system performs well tracking a hovering or in-frame drone against a stationary monochrome background. Dynamic backgrounds with similarly-shaped low-contrast objects are the primary failure mode.

## Tech Stack

- **Python 3** — core language
- **YOLOv8 (Ultralytics)** — object detection
- **filterpy** — Kalman filter implementation
- **OpenCV** — frame processing and visualization
- **ffmpeg-python** — video I/O
- **PyTorch** — model inference
- **Docker / Docker Compose** — containerized dev environment (NVIDIA GPU on Linux, CPU on macOS)
- **ROS 2 Jazzy** — robotics middleware (included in dev environment)

## Project Structure

```
KalmanFilters/
├── project/              # Core pipeline source code
├── assignments/          # Course assignment notebooks
├── notebooks/            # Exploratory notebooks
├── docs/                 # Quarto documentation source
├── docker/               # Docker configuration
├── ros_ws/               # ROS 2 workspace
├── index.py              # Main pipeline entry point
└── README.md
```

## Running the Pipeline

This project uses a Docker-based dev environment. NVIDIA GPU support is available on Linux; macOS runs CPU-only.

**Prerequisites**
- Docker and Docker Compose
- VS Code with the Dev Containers extension
- NVIDIA Container Toolkit (Linux, for GPU support)

**Quick start**
```bash
# Clone the repo
git clone https://github.com/DanielAyer/KalmanFilters.git
cd KalmanFilters

# Copy environment file
cp .env.example .env

# Start the GPU container (Linux)
docker compose up -d torch.dev.gpu
docker compose exec torch.dev.gpu bash

# Inside the container
make start
source .venv/bin/activate
```

**macOS**
```bash
docker compose -f docker-compose-mac.yml up -d torch.dev.mac
docker compose -f docker-compose-mac.yml exec torch.dev.mac bash
```

See the [full environment setup docs](https://aegean.ai/aiml-common/resources/environment/) for additional configuration.

## Context

This project was completed as part of DS681 (AI Agents and Computer Vision) at NJIT Ying Wu College of Computing, Spring 2026. The dev environment template is based on [pantelis/eng-ai-agents](https://github.com/pantelis/eng-ai-agents).

## Author

**Daniel Ayer**  
M.S. Artificial Intelligence Candidate, NJIT  
[LinkedIn](https://linkedin.com/in/danielayer) · [GitHub](https://github.com/DanielAyer) · [Hugging Face](https://huggingface.co/danielayer)
