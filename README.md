# ZenML Object Detection Pipeline

This is a simple project that shows how to use ZenML to transform a COCO dataset and train a model on it.

This project is mostly meant to showcase the capabilities of ZenML to handle complex data pipelines and model training. 

## Setup

```bash
pip install -r requirements.txt
```

## Run the training pipeline

The pipeline first loads the COCO dataset as part of the create_coco_dataset step.

In the next step all the images of the dataset are processed. In our case we just resize the images to 224x224 pixels and adjust the annotations accordingly.

The combine_step combines the processed chunks and converts the dataset to the YOLO format.

The split_dataset step splits the dataset into a training and validation set.

The training step trains a YOLO model on the training set.

```bash
python process_and_training.py
```

## Run the inference pipeline

This pipeline loads the trained model and uses it to make a prediction on an image.

```bash
python inference.py
```
