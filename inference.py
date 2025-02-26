from typing import Annotated, Tuple, Any
import requests
import tempfile
import os
from PIL import Image, ImageDraw

from zenml import step, pipeline, Model, get_step_context
from zenml.config import DockerSettings
from zenml.enums import ModelStages


def visualize_yolo_prediction(
    image: Image.Image,
    prediction,
    box_color: str = "red",
    line_width: int = 2
) -> Image.Image:
    """
    Draw YOLO predictions on a PIL Image.
    
    Args:
        image: PIL Image to draw on
        prediction: YOLO prediction result
        box_color: Color for bounding boxes
        line_width: Width of the lines to draw
        
    Returns:
        PIL Image with predictions drawn
    """
    draw = ImageDraw.Draw(image)
    
    # Process each detection
    for result in prediction:
        boxes = result.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            # Draw bounding box
            draw.rectangle(
                [(x1, y1), (x2, y2)],
                outline=box_color,
                width=line_width
            )
            
            # Add class name and confidence
            class_name = result.names[int(box.cls)]
            confidence = float(box.conf)
            label = f"{class_name}: {confidence:.2f}"
            draw.text((x1, y1 - 10), label, fill=box_color)
    
    return image

@step
def inference(
    model_artifact_name: str = "yolo_model",
    image_url: str = "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg"
) -> Annotated[Image.Image, "prediction_visualization"]:
    """
    Run inference on an image and visualize the results.
    
    Args:
        model_artifact_name: Name of the model artifact to use
        image_url: URL of the image to process
        
    Returns:
        Tuple of (prediction results, visualized image with predictions)
    """
    # Download image from URL
    response = requests.get(image_url)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        tmp_file.write(response.content)
        temp_image_path = tmp_file.name

    # Load model and run inference
    zenml_model = get_step_context().model
    yolo_model = (
        zenml_model.get_model_artifact(model_artifact_name)
                   .load()             
    )
    
    prediction = yolo_model(source=temp_image_path)
    
    # Load image and visualize predictions
    image = Image.open(temp_image_path)
    visualized_image = visualize_yolo_prediction(image, prediction)

    return visualized_image


@pipeline(
    enable_cache=False,
    model=Model(
        name="SimpleYOLO",
        version=ModelStages.LATEST
    ),
    settings={
        "docker": DockerSettings(
            requirements="requirements.txt",
        ),
    },
)
def yolo_inference():
    inference()

yolo_inference()