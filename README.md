Dataset Choice:
    I chose to use: https://huggingface.co/datasets/lgrzybowski/seraphim-drone-detection-dataset for the fine tuning dataset as it was the largest dataset of images of drones on Hugging Face.  It was also challenging to find images OF drones instead of images taken BY drones.  I thought the larger the training corpus, the better fine tuning performance potential.

Detector Configuration:
    The base detector selected was: yolov8s.pt .  This was selected as small enough to train overnight, and large enough to maximize performance constrained by the training time available.  It was set to train overnight.  In the morning I accidentally unplugged my laptop, and that shut the entire system off immediately.  Fortunately, YOLO is well designed to tolerate such interruptions, and the training was resumed and completed.
    
    Training achieved an F1 of 0.91 on the test dataset.  

Kalman Filter State Design
    The Kalman filter was initialized with four state variables: [x, y, vx, vy], and two measurement variables: [x, y].  Since we're analyzing flat images that's what we have to measure.  The initial state was set as [x, y, 0, 0] using the first detection's x and y and assuming the drone was at rest.  The noise parameters were set using the documentation: https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html .  

Failure Cases

Video 1
    In video 1 at 0:49 the drone leaves the field of view.  It re-enters a second later in the top of the frame.  The system quickly reacquires the drone, but infers a trajectory line that the drone did not travel.  It connects the last known detection point to the current detection point with a straight line.
    The drone once again leaves the field of view around 1:25.  The detector reacquires the drone quickly, but the inferred trajectory line takes until about 1:28 to actually connect to the center of the drone’s bounding box again.  Though, it does lag behind by about a frame, and does seem to track the drone well.
     In the final shot of video 1 the drone leaves the field of view again, and the inferred trajectory moves off at about a 315 degree angle.

Video 2
     The system tracks the drone well through to about 0.07 despite a disjoint at around 0.05.  In 0.06 the drone appears in an entirely different position.  Perhaps this is aliasing due to the capture rate of the frames or the source video.  Once the drone leaves the field of view the system begins tracking voids in the cloud cover that is moving in parallax in the background.
     At around 0:13 the drone reappears but much farther away.  The system is unable to identify it and continues to to track the cloud voids.
     The drone continues to move about and is eventually detected and tracked.  Though the system moves between drone and cloud void frequently.  While the drone is in frame the system seems to track it well up to a certain distance.  When it leaves frame some of the cloud voids are traced instead.
     When the system fails to detect the drone in frame, it seems to track a cloud void.  However, when it reaquires the drone it snaps back to that object.

Failure Case Discussion

    The system seems well able to hand a hovering drone, or a drone tracked within frame, against a stationary relatively monochrome background.  However, some cloud voids seem to look a bit like drones, and without a superior target to track the system seems to track the voids.




# Introduction

## What is this repository?

This is a template docker-based dev environment. It supports NVIDIA GPUs on Linux, and provides CPU-based containers for macOS (Apple Silicon and Intel).

It currently includes the following tools:

- An `assignments` directory with notebooks where you populate your code
- A `project` directory for your project source code. The documentation for the project is stored separately in the `docs` directory
- A `docs` directory that contains the source code of [Quarto](https://quarto.org/) markdown (qmd) and ipynb notebooks content. You use the docs folder to publish your project work

## How to Launch the Development Container in VS Code

This repository includes a VS Code development container configuration that can be launched with either CPU or GPU support.

### Prerequisites

1. **Install VS Code** with the "Dev Containers" extension
2. **Install Docker** and ensure it's running
3. **For GPU support**: Install NVIDIA Container Toolkit (for Linux) or Docker Desktop with GPU support

**IMPORTANT:** After the container is launched, you must run the following commands **inside the container** to set up the environment:

```bash
make start                    # Must run inside container (uses uv package manager)
source .venv/bin/activate     # Activate the virtual environment
```

**Critical Note:** The `make start` command (and `make venv-recreate`) must be executed from within the devcontainer, not on the host machine. The Makefile uses the `uv` package manager which is only available inside the container and respects container-specific constraints.

## Running with Docker Compose (without VS Code)

You can also run the containers directly with Docker Compose:

```bash
# Copy environment file
cp .env.example .env

# Build and start the PyTorch GPU container
docker compose up -d torch.dev.gpu

# Exec into the running container
docker compose exec torch.dev.gpu bash

# Or for ROS development
docker compose up -d ros.dev.gpu
docker compose exec ros.dev.gpu bash
```

## macOS Development (Apple Silicon & Intel)

This repository includes dedicated support for macOS users, including both Apple Silicon (M1/M2/M3/M4) and Intel-based Macs.

### Important Limitations

**GPU Acceleration (MPS) is NOT available inside Docker on macOS.** Docker Desktop uses virtualization (HyperKit/Apple Virtualization Framework) which lacks access to Apple's Metal GPU APIs. For GPU-accelerated PyTorch training using MPS, you must run PyTorch natively on macOS, not inside Docker.

For more details, see:

- [Apple Silicon GPUs, Docker and Ollama: Pick Two](https://chariotsolutions.com/blog/post/apple-silicon-gpus-docker-and-ollama-pick-two/)
- [PyTorch MPS Backend Documentation](https://docs.pytorch.org/docs/stable/notes/mps.html)

### macOS Services

The repository provides macOS-specific services via `docker-compose-mac.yml`:

- **`torch.dev.mac`**: PyTorch development environment (CPU-only)
- **`ros.dev.mac`**: ROS 2 Jazzy development environment

### Running on macOS

```bash
# Copy environment file
cp .env.example .env

# Build and start the PyTorch container for macOS
docker compose -f docker-compose-mac.yml up -d torch.dev.mac

# Exec into the running container
docker compose -f docker-compose-mac.yml exec torch.dev.mac bash

# Or for ROS 2 development
docker compose -f docker-compose-mac.yml up -d ros.dev.mac
docker compose -f docker-compose-mac.yml exec ros.dev.mac bash
```

### VS Code Dev Container for macOS

To use the macOS containers with VS Code, update `.devcontainer/devcontainer.json`:

```json
{
  "dockerComposeFile": ["../docker-compose-mac.yml"],
  "service": "torch.dev.mac",
  "runServices": ["torch.dev.mac"]
}
```

### GUI Applications (RViz2, etc.) on macOS

Running GUI applications like RViz2 requires XQuartz:

1. Install XQuartz:

   ```bash
   brew install --cask xquartz
   ```

2. Open XQuartz and go to **Preferences > Security**
3. Enable **"Allow connections from network clients"**
4. **Reboot your Mac** (required for changes to take effect)
5. After reboot, allow connections:

   ```bash
   xhost +localhost
   ```

**Troubleshooting:** If you see `Error: Can't open display: host.docker.internal:0`, ensure you have:

- Completed all XQuartz configuration steps above
- Rebooted your Mac after enabling network clients
- Run `xhost +localhost` in a terminal on the host

For more details, see:

- [Installing ROS 2 on macOS with Docker (Foxglove)](https://foxglove.dev/blog/installing-ros2-on-macos-with-docker)
- [Setup ROS 2 Dev Docker with Emacs in macOS](https://qurobotics.de/blog/2024-01-11-setup-ros2-dev-docker-with-emacs-in-macos/)
- [Development Container for ROS 2 on ARM64 Mac](https://github.com/tatsuyai713/Development-Container-for-ROS2-on-Arm64-Mac)

### Native PyTorch with MPS (Recommended for GPU Training)

If you need GPU acceleration on macOS, install PyTorch natively (outside Docker):

```bash
# Create a virtual environment
python3 -m venv .venv-native
source .venv-native/bin/activate

# Install PyTorch with MPS support
pip install torch torchvision torchaudio

# Verify MPS is available
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

See [Apple's PyTorch Metal documentation](https://developer.apple.com/metal/pytorch/) for more details.

### Port Customization

You can customize the exposed ports by modifying the `.env` file. Each service has its own set of port variables:

**Dev Service Ports:**

- `DEV_QUARTO_PORT`: Quarto preview server (default: 4100)
- `DEV_JUPYTER_PORT`: Jupyter notebook server (default: 8888)
- `DEV_PORT`: Additional development server (default: 8000)

**ROS Service Ports:**

- `ROS_PORT`: ROS master port (default: 11311)
- `ROS_QUARTO_PORT`: Quarto preview server (default: 4180)
- `ROS_JUPYTER_PORT`: Jupyter notebook server (default: 8880)
- `ROS_DEV_PORT`: Additional development server (default: 8078)
- `FOXGLOVE_PORT`: Foxglove bridge WebSocket (default: 8765)

Note: The actual ports exposed will be the values from your `.env` file.

### Service Selection

The repository supports multiple container configurations:

- **`torch.dev.gpu`**: PyTorch development environment with GPU support (Linux)
- **`ros.dev.gpu`**: ROS 2 (Jazzy) development environment with GPU support (Linux)
- **`torch.dev.mac`**: PyTorch development environment for macOS (CPU-only)
- **`ros.dev.mac`**: ROS 2 (Jazzy) development environment for macOS

#### Switching Services

To switch between services, modify the `service` field in `.devcontainer/devcontainer.json`:

```json
{
  "service": "torch.dev.gpu"  // or "ros.dev.gpu", "torch.dev.mac", "ros.dev.mac"
}
```

After changing the service configuration, rebuild the container using VS Code's "Dev Containers: Rebuild Container" command.

#### Why Two Containers Launch by Default

When VS Code opens the Dev Container, it runs `docker-compose up` which starts **all services** defined in `docker-compose.yml` by default. The `"service"` field in `devcontainer.json` only specifies which container VS Code attaches to—it doesn't limit which containers are started.

Both containers share the same network (`ai-agents-network`), allowing inter-container communication if needed.

**To start only a single container**, add the `runServices` property to `.devcontainer/devcontainer.json`:

```json
{
  "service": "torch.dev.gpu",
  "runServices": ["torch.dev.gpu"]
}
```

This explicitly tells VS Code to only start the specified service(s) rather than all services in the compose file.

## What should I do with it?

- Follow all instructions under [resources in the class website](https://aegean.ai/aiml-common/resources/environment/) as you will need it to submit your work
- Familiarize yourself with the `uv` package manager as you will use it to build and manage all your dependencies
- Follow the instructions in the course web site under resources to [submit your GitHub repo to the course's LMS system](https://aegean.ai/aiml-common/resources/environment/assignment-submission.html) (Canvas/Brightspace)

### Additional Notes for ROS Development

**ROS 2 Discovery Settings (macOS):** The macOS containers are configured with `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST` and `ROS_DOMAIN_ID=42` to ensure proper ROS 2 node discovery within Docker's network isolation.

**Foxglove Bridge:** Connect to the Foxglove app using:

```bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml
```

**Building ROS Packages:** Use the provided aliases:

```bash
# Install dependencies
rosdi

# Build with symlink install
cbuild

# Source the workspace
ssetup
```


```python
# Import dependencies.
import os

import torch
from ultralytics import YOLO
import shutil
import ffmpeg
import numpy as np
import cv2

# Deliverable
# Split each video into frames and run your detector on every frame. 
# Save all frames that contain at least one detection to a folder called detections/. 
# Write your code so that it processes all .mp4 files in a given directory, not just the two test videos.


# Set device.                                                                                                                                                                                    
if torch.cuda.is_available():                                                                                                                                                               
    device = 0  
else:                                                                                                                                                                                       
    device = "cpu"
    print("WARNING: No GPU detected, training on CPU. This will be very slow.")

# Closure to run on each frame.
def store_detections_closure(model, frames_dir, detections_dir):
    def store_detections(predictor):
    # This function stores the results when a frame contains the object.

        for result in predictor.results:

            # # Testing.
            # print(f"Image: {result.path}")                                                                                                                                                      
            # print(f"Boxes: {result.boxes}")
            # print(f"Confidences: {result.boxes.conf}")
            # print(f"Classes: {result.boxes.cls}")

            # Check detection.
            if len(result.boxes) > 0:

                ### Save images with detections.  ###
                
                # Keep same folder convention as source frame folder.
                src = result.path

                # Get video name without extension.
                video_name, _ = os.path.splitext(os.path.basename(os.path.dirname(result.path)))

                # Set destination directory.
                dst_dir = os.path.join(detections_dir, video_name)

                # Make the directory.
                os.makedirs(dst_dir, exist_ok=True)

                dst = os.path.join(dst_dir, os.path.basename(result.path)) 

                # Copy into detections.
                shutil.copy(src, dst)

    
    return store_detections

# Fucntion to parse the videos into frames.
def parse_videos(vidoes_dir, frames_dir):
    # This function parses the videos in the path into frames stored in subfolders labeled for the video.

    # Split each video in folder into frames.
    videos_list = os.listdir(vidoes_dir)

    for video in videos_list:

        # Name destination folder to store frames.
        target = f"{frames_dir}/{video}" 

        try:
            # Make the directory
            os.mkdir(target)

        except FileExistsError:

            # Folder exists.  Check if contents.
            if len(os.listdir(target)) == 0:

                # Folder needs frames.
                print(f"Making frames of {target}")

            else:

                # Report and move on.
                print(f"It look's like {target} already exists and has frames.  \nPlease delete the folder and re-reun if needed.") 

        # Extract frames.
        output_pattern = os.path.join(target, "frame_%04d.jpg")

        (
            ffmpeg
            .input(f"{vidoes_dir}/{video}")
            .filter("fps", fps=5)
            .output(output_pattern)
            .run()
        )

def task_1():

    # Set videos uri.
    videos_dir = "/home/johnsmith/Desktop/njit/workspaces/ds681/eng-ai-agents-main/assignments/assignment-3/videos/"

    # Create videos folder.
    os.makedirs(videos_dir, exist_ok=True) 

    # Set frames uri.
    frames_dir = "/home/johnsmith/Desktop/njit/workspaces/ds681/eng-ai-agents-main/assignments/assignment-3/frames/"

    # Create frames folder.
    os.makedirs(frames_dir, exist_ok=True) 

    # Set detections uri.
    detections_dir = "/home/johnsmith/Desktop/njit/workspaces/ds681/eng-ai-agents-main/assignments/assignment-3/detections"

    # Create detections folder.
    os.makedirs(detections_dir, exist_ok=True)   

    # Parse videos.
    # parse_videos(videos_dir, frames_dir)


    """ 
        Model fine tuned in train_yolo.py .

    """
    # Set model uri.
    model_uri = "/home/johnsmith/Desktop/njit/workspaces/ds681/eng-ai-agents-main/assignments/assignment-3/runs/yolo-drone-20260306-185910/weights/best.pt"

    # Load fine tuned model.
    model = YOLO(model_uri, task = 'detect')

    # Add callback to the model.
    model.add_callback("on_predict_batch_end", store_detections_closure(model, frames_dir, detections_dir))

    # Run the model on all frames of each video.
    for video in os.listdir(frames_dir):

        # Run model on each folder of images.  Because of the added callback we don't need to store the results.
        with torch.no_grad():
            model(f"{frames_dir}{video}")

# End of task_1().

from filterpy.kalman import KalmanFilter  

def task_2():

    """ 
        Model fine tuned in train_yolo.py .

    """
    # Set model uri.
    model_uri = "/home/johnsmith/Desktop/njit/workspaces/ds681/eng-ai-agents-main/assignments/assignment-3/runs/yolo-drone-20260306-185910/weights/best.pt"

    # Load fine tuned model.
    model = YOLO(model_uri, task = 'detect')

    # Set frames uri.
    frames_dir = "/home/johnsmith/Desktop/njit/workspaces/ds681/eng-ai-agents-main/assignments/assignment-3/frames/"

    # Set detections folder.
    detections_dir = "/home/johnsmith/Desktop/njit/workspaces/ds681/eng-ai-agents-main/assignments/assignment-3/detections"

    # Set make_videos folder.
    make_videos_dir = "/home/johnsmith/Desktop/njit/workspaces/ds681/eng-ai-agents-main/assignments/assignment-3/make_videos"

    # Make videos folder.
    os.makedirs(make_videos_dir, exist_ok = True)

    # The frames_dir contains a folder for each processed video.
    for video in os.listdir(frames_dir):

        # Initialize vars.

        # Set kf to None for first pass detection.
        kf = None

        # Build list to track trajectories.
        trajectory = []
        
        # Join path to the video in question.
        detection_dir_path = os.path.join(detections_dir, video)

        # If the folder doesn't exist then no detections were made in the video; skip.
        if not os.path.exists(detection_dir_path):
            continue

        # Get sorted list of all frames in video folder.
        all_frames = sorted(os.listdir(detection_dir_path))

        # Iterate over frames.
        for frame in all_frames:

            # Run detector on detection frame.
            result = model(os.path.join(detection_dir_path, frame))[0]

            # Create list of the x, y, width and height values.
            x, y, w, h = result.boxes.xywh[0].tolist()

            if kf is None:
                # Initialize on first detection.

                # Create Kalman Filter with four state variables [x, y, vx, vy] and two measurement variables [x,y].
                kf = KalmanFilter(dim_x=4, dim_z=2)

                # Initialize state.  Velocity is all zero assuming an at rest drone.
                kf.x = np.array([x, y, 0., 0.])

                # State transition matrix.
                kf.F = np.array([[1, 0, 1, 0],
                                 [0, 1, 0, 1],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]], dtype=float)
                
                # Measurement Matrix
                kf.H = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0]], dtype=float)
                
                # Measuremment noise of 5 pixels.
                kf.R = np.eye(2) * 5

                # Process noise.
                kf.Q = np.eye(4) * 0.1

                # Initial state covariance.
                kf.P = np.eye(4) * 1000

            else:
                kf.predict()
                kf.update(np.array([x, y]))

            # Append estimated position to trajectory.
            trajectory.append((int(kf.x[0]), int(kf.x[1])))

            # Draw overlays.
            dst_dir = os.path.join(make_videos_dir, video)
            os.makedirs(dst_dir, exist_ok=True)

            img = cv2.imread(os.path.join(detection_dir_path, frame))

            # Draw bounding box.
            x1, y1 = int(x - w/2), int(y - h/2)
            x2, y2 = int(x + w/2), int(y + h/2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw trajectory polyline.
            if len(trajectory) > 1:
                cv2.polylines(img, [np.array(trajectory, dtype=np.int32)], False, (0, 0, 255), 2)

            cv2.imwrite(os.path.join(dst_dir, frame), img)

            print(f"{video}/{frame}: estimated position {kf.x[:2]}")

def stitch_frames_to_video():
    # This function stiches all the images in a folder into a video.
    # I am so tired man.  

    videos_dir = "/home/johnsmith/Desktop/njit/workspaces/ds681/eng-ai-agents-main/assignments/assignment-3/make_videos"

    for video in os.listdir(videos_dir):

        (                                                                                                                                                                                           
      ffmpeg                                                                                                                                                                                  
      .input(os.path.join(videos_dir, video, "frame_*.jpg"), pattern_type="glob", framerate=5)
      .output(os.path.join(videos_dir, f"{video}.mp4"))
      .run()
  )

    
# Run Task.
stitch_frames_to_video()



```
