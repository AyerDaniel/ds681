# DS681 Daniel Ayer  Assignment_2 Spring 2026

## Dependencies ##

from ultralytics import YOLO


## Flags ##

# Variable to flag for training or not.
train_ = True
eval_ = True

## Run Code ##

# Load a pretrained segmentation model like YOLO26n-seg
model = YOLO("yolo26n-seg.pt")  # load a pretrained model (recommended for training)

# Train if wanted.
if train_:

    # Train the model on the Carparts Segmentation dataset
    results = model.train(data="carparts-seg.yaml", epochs=100, imgsz=640)

if eval_:
    # After training, you can validate the model's performance on the validation set
    results = model.val()

# Or perform prediction on new images or videos
#results = model.predict("path/to/your/image.jpg")