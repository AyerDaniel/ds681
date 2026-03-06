from ultralytics import YOLO

import numpy as np

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOGGER

from datetime import datetime

import torch                                                                                                                                                                                
                                                                                                                                                                                              
if torch.cuda.is_available():                                                                                                                                                               
    device = 0  
else:                                                                                                                                                                                       
    device = "cpu"
    print("WARNING: No GPU detected, training on CPU. This will be very slow.")

# This code from: https://docs.ultralytics.com/guides/custom-trainer/#logging-custom-metrics  -Daniel Ayer -03062026

class MetricsTrainer(DetectionTrainer):
    """Custom trainer that computes and logs F1 score at the end of each epoch."""

    def validate(self):

        """Run validation and compute per-class F1 scores."""
        metrics, fitness = super().validate()

        if metrics is None:
            return metrics, fitness

        if hasattr(self.validator, "metrics") and hasattr(self.validator.metrics, "box"):

            box = self.validator.metrics.box
            f1_per_class = box.f1
            class_indices = box.ap_class_index
            names = self.validator.names

            valid_f1 = f1_per_class[f1_per_class > 0]
            mean_f1 = np.mean(valid_f1) if len(valid_f1) > 0 else 0.0

            LOGGER.info(f"Mean F1 Score: {mean_f1:.4f}")
            per_class_str = [
                f"{names[i]}: {f1_per_class[j]:.3f}" for j, i in enumerate(class_indices) if f1_per_class[j] > 0
            ]
            LOGGER.info(f"Per-class F1: {per_class_str}")

        return metrics, fitness


# Load a pretrained YOLO model (you can choose n, s, m, l, or x versions)
model = YOLO("yolo26n.pt")

# Store datasets.yaml uri.
datasets = "/home/johnsmith/Desktop/njit/workspaces/ds681/eng-ai-agents-main/assignments/assignment-3/datasetts.yaml"

# Set device.

# Dynamically name each run.
run_name = f"yolo-drone-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

# Start training on your custom dataset
model.train(
    data = datasets, 
    epochs = 100, 
    batch = -1,
    device = device,
    optimizer = 'Adam',
    patience = 10,
    name = run_name,
    imgsz = 640,
    trainer=MetricsTrainer)
