# I accidentally unplugged my laptop at about 80% training complete.  This script is to resume that training.

import os                                                                                                                                                                                   
os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                                                                                                                                                                                            
import numpy as np
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOGGER


class MetricsTrainer(DetectionTrainer):
    """Custom trainer that computes and logs F1 score at the end of each epoch."""

    def validate(self):
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
                f"{names[i]}: {f1_per_class[j]:.3f}"
                for j, i in enumerate(class_indices)
                if f1_per_class[j] > 0
            ]
            LOGGER.info(f"Per-class F1: {per_class_str}")
        return metrics, fitness


model = YOLO("/home/johnsmith/Desktop/njit/workspaces/ds681/eng-ai-agents-main/assignments/assignment-3/runs/yolo-drone-20260306-185910/weights/last.pt")
model.train(resume=True, trainer=MetricsTrainer)
