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