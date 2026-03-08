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
def parse_videos(vidoes_dir):
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
            .input(f"{videos_dir}/{video}")
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
    # parse_videos(videos_dir)


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
    os.makedirs(make_videos_dir, exist_ok=True)

    for video in os.listdir(frames_dir):

        # Initialize vars.
        kf = None
        trajectory = []

        detection_dir_path = os.path.join(detections_dir, video)
        if not os.path.exists(detection_dir_path):
            continue

        all_frames = sorted(os.listdir(detection_dir_path))

        for frame in all_frames:

            # Run detector on detection frame.
            result = model(os.path.join(detection_dir_path, frame))[0]
            x, y, w, h = result.boxes.xywh[0].tolist()

            if kf is None:
                # Initialize on first detection.
                kf = KalmanFilter(dim_x=4, dim_z=2)
                kf.x = np.array([x, y, 0., 0.])
                kf.F = np.array([[1, 0, 1, 0],
                                 [0, 1, 0, 1],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]], dtype=float)
                kf.H = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0]], dtype=float)
                kf.R = np.eye(2) * 5
                kf.Q = np.eye(4) * 0.1
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


