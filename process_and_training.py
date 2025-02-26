import json
from pathlib import Path
import random
import tempfile
from typing import Dict, List, Tuple
from urllib.request import urlopen
from io import BytesIO
import os
import logging
from uuid import uuid4
from zipfile import ZipFile
from PIL import Image as PILImage
from tqdm import tqdm
from typing_extensions import Annotated
from pydantic import BaseModel
from ultralytics import YOLO
import yaml
from zenml import ArtifactConfig, Model, get_pipeline_context, get_step_context, pipeline, step, register_artifact
from zenml.client import Client 
from zenml.enums import ArtifactType
from zenml.types import HTMLString
from zenml.io import fileio
from zenml.config import DockerSettings
from annotation import COCODataset, draw_annotations_on_image
from ultralytics_materializer import UltralyticsMaterializer
from utils import visualize_folder_structure, copy_recursive



def do_transform(
    image_base_path: str,
    image_name: str,
    annotations: COCODataset
) -> Tuple[PILImage.Image, COCODataset]:
    """
    Transforms an image and its corresponding annotations by resizing them to 224x224.

    Args:
        image_base_path: Path to the directory containing the source images
        image_name: Name of the image file to transform
        annotations: COCO format annotations for the image

    Returns:
        tuple: (
            transformed_image: PIL Image resized to 224x224 in RGB format,
            transformed_annotations: Updated annotations matching the new image dimensions
        )
    """
    logging.debug(f"Transforming image: {image_name}")
    
    # Load and process image
    image_path = os.path.join(image_base_path, image_name)
    original_image = PILImage.open(fileio.open(image_path, "rb"))
    original_width, original_height = original_image.size
    logging.debug(f"Original image size: {original_width}x{original_height}")

    # Convert to RGB and resize
    transformed_image = original_image.convert("RGB")
    transformed_image = transformed_image.resize((224, 224))
    logging.debug("Image resized to 224x224")

    # Update annotations to match new dimensions
    transformed_annotations = annotations.resize_annotations_for_image(image_name, 224, 224)
    logging.debug(f"Updated annotations for {image_name}")

    return transformed_image, transformed_annotations


@step(enable_cache=True)
def create_coco_dataset(output_root_dir: str) -> Tuple[
    Annotated[str, "coco_dataset"],
    Annotated[HTMLString, "coco_dataset_structure"],
    Annotated[PILImage.Image, "example_image"]
    ]:
    """
    Loads the tiniest_coco dataset from the internet and saves it to the artifact store.

    The dataset has the following structure:
    tiniest_coco-main/
    ├── annotations/
    │   ├── instances.json
    ├── images/
    │   ├── 0000001.png
    │   ├── 0000002.png

    Returns:
        The path to the created COCO dataset in the artifact store.
    """
    logging.info("Starting COCO dataset creation")
    step_context = get_step_context()

    logging.info(f"Output directory: {output_root_dir}")

    url = "https://github.com/AlexejPenner/tiniest_coco/archive/refs/heads/main.zip"
    logging.info(f"Downloading dataset from {url}")

    # Create temp dir for extraction - this tmp dir 
    tmp_dir = tempfile.mkdtemp()
    
    # Download and extract to temp dir
    http_response = urlopen(url)
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path=tmp_dir)
    
    # Copy extracted contents to output dir
    src_path = os.path.join(tmp_dir, "tiniest_coco-main")
    copy_recursive(src_path, output_root_dir)
    logging.info("Dataset extracted successfully")

    images_dir = os.path.join(output_root_dir, "images")
    num_images = len(fileio.listdir(images_dir))
    logging.info(f"Dataset contains {num_images} images")

    annotations_path = os.path.join(output_root_dir, "annotations", "instances.json")
    json_data = json.load(fileio.open(annotations_path, "r"))
    num_annotations = len(json_data["annotations"])
    logging.info(f"Dataset contains {num_annotations} annotations")
    
    zenml_artifact = register_artifact(
        output_root_dir,
        name="initial_coco_dataset",
        artifact_metadata={
            "description": "Tiniest COCO dataset",
            "source": url, 
            "num_images": num_images,
            "num_annotations": num_annotations
        }
    )

    annotations = COCODataset.model_validate_json(
        fileio.open(annotations_path, "r").read()
    )

    # Draw annotations on a random image as a visual check
    random_image = random.choice(annotations.images)
    image = PILImage.open(fileio.open(os.path.join(images_dir, random_image.file_name), "rb"))

    annotated_image = draw_annotations_on_image(
        image=image,
        coco_dataset=annotations,
        image_name=random_image.file_name,
        draw_boxes=True,
        draw_segments=True
    )
    
    return zenml_artifact.uri, visualize_folder_structure(output_root_dir), annotated_image


@step(enable_cache=True)
def transform(
    input_dataset_path: str,
    output_dataset_path: str,
    chunk: int,
    splits: int = 5
) -> Tuple[
    Annotated[str, "transformed_dataset_path"],
    Annotated[COCODataset, "transformed_annotations"],
    Annotated[PILImage.Image, "example_image"]
]:
    """
    Transforms a subset of images from the dataset by resizing them and their annotations.

    Args:
        input_dataset_path: Source directory containing the COCO dataset
        output_dataset_path: Target directory for the transformed dataset
        chunk: Current chunk number being processed (0-based index)
        splits: Total number of chunks to divide the dataset into

    Returns:
        tuple: (
            output_dataset_path: Path to the directory containing transformed images,
            transformed_annotations: Updated COCO annotations for the transformed chunk,
            example_image: Random image from the chunk with annotations visualized
        )
    """
    step_run_id = get_step_context().step_run.original_step_run_id
    logging.info(f"Starting transformation of chunk {chunk + 1}/{splits}")
    
    # Calculate chunk boundaries
    source_images_dir = os.path.join(input_dataset_path, "images")
    all_image_names = fileio.listdir(source_images_dir)
    chunk_start = chunk * len(all_image_names) // splits
    chunk_end = (chunk + 1) * len(all_image_names) // splits
    chunk_image_names = all_image_names[chunk_start:chunk_end]
    logging.info(f"Processing {len(chunk_image_names)} images in chunk {chunk + 1}")

    # Load and filter annotations for this chunk
    annotations_path = os.path.join(input_dataset_path, "annotations", "instances.json")
    chunk_annotations = COCODataset.model_validate_json(
        fileio.open(annotations_path, "r").read()
    )
    chunk_annotations = chunk_annotations.filter_by_image_ids(chunk_image_names)
    logging.info(f"Loaded annotations for chunk {chunk + 1} successfully")
    
    transformed_images: Dict[str, PILImage.Image] = {}
    for image_name in tqdm(chunk_image_names, desc="Transforming images"):
        image, annotations = do_transform(
            image_base_path=source_images_dir, 
            image_name=image_name, 
            annotations=chunk_annotations
            )
        transformed_images[image_name] = image
    
    output_images_dir = os.path.join(output_dataset_path, "images")
    fileio.makedirs(output_images_dir)

    for image_name, image in tqdm(transformed_images.items(), desc="Saving transformed images"):
        temp_path = os.path.join("/tmp", "temp_image.jpg")
        image.save(temp_path)
        fileio.copy(temp_path, os.path.join(output_images_dir, image_name), overwrite=False)

    random_image = random.choice(list(transformed_images.keys()))
    annotated_image = draw_annotations_on_image(
        image=transformed_images[random_image],
        coco_dataset=chunk_annotations,
        image_name=random_image,
        draw_boxes=True,
        draw_segments=True
    )

    return output_dataset_path, chunk_annotations, annotated_image


@step(enable_cache=True)
def combine_step(
    step_prefix: str,
    yolo_dataset_dir: str
    ) -> Tuple[
        Annotated[str, "yolo_dataset"],
        Annotated[HTMLString, "yolo_dataset_structure"]
        ]:
    """
    Combines the processed chunks and converts to YOLO format.

    Args:
        step_prefix: The prefix of the steps to combine.
        yolo_dataset_dir: The directory to save the YOLO dataset.
    Returns:
        The path to the created YOLO dataset in the artifact store.
    """
    logging.info("Starting combination of processed chunks")
    run_name = get_step_context().pipeline_run.name
    run = Client().get_pipeline_run(run_name)

    # Fetch all results from parallel processing steps
    output_dataset_paths: List[str] = []
    processed_annotations: List[COCODataset] = []
    for step_name, step_info in run.steps.items():
        if step_name.startswith(step_prefix):
            logging.info(f"Processing step: {step_name}")
            transformed_dataset_path = step_info.outputs["transformed_dataset_path"][0].load()
            output_dataset_paths.append(transformed_dataset_path)
            processed_annotations.append(step_info.outputs["transformed_annotations"][0].load())
            logging.info(f"Added annotations from {step_name}")
    
    # Ensure all paths are the same
    if len(set(output_dataset_paths)) != 1:
        raise ValueError("All transform steps must write to the same transformed dataset path")
    output_path = output_dataset_paths[0]
    
    # Merge annotations
    annotation = processed_annotations[0]
    for i, annotations in enumerate(processed_annotations[1:], 1):
        logging.info(f"Merging chunk {i}")
        annotation.merge(annotations)

    # Create images directory and copy images
    images_dir = os.path.join(yolo_dataset_dir, "images")
    fileio.makedirs(images_dir)
    
    source_images_dir = os.path.join(output_path, "images")
    for image_name in fileio.listdir(source_images_dir):
        src_path = os.path.join(source_images_dir, image_name)
        dst_path = os.path.join(images_dir, image_name)
        fileio.copy(src_path, dst_path)
    
    logging.info("Copied images to YOLO dataset directory")

    # Create labels directory
    labels_dir = os.path.join(yolo_dataset_dir, "labels")
    fileio.makedirs(labels_dir)

    # Get category mapping
    categories = {cat.id: idx for idx, cat in enumerate(annotation.categories)}
    
    

    # Create YOLO format annotations
    for image in annotation.images:
        image_annotations = annotation.get_annotations_for_image(image.file_name)
        
        # Create YOLO format strings
        yolo_annotations = []
        for ann in image_annotations:
            # Get bounding box in YOLO format (x_center, y_center, width, height)
            x, y, w, h = ann.bbox
            x_center = (x + w/2) / image.width
            y_center = (y + h/2) / image.height
            width = w / image.width
            height = h / image.height
            
            # Convert category ID to index
            class_idx = categories[ann.category_id]
            
            # YOLO format: class_idx x_center y_center width height
            yolo_annotations.append(f"{class_idx} {x_center} {y_center} {width} {height}")
        
        # Write annotations to file
        label_filename = os.path.splitext(image.file_name)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_filename)
        with fileio.open(label_path, "w") as f:
            f.write("\n".join(yolo_annotations))

    # Create data.yaml
    yaml_content = {
        "path": ".",
        "train": "images",  # relative to path
        "val": "images",    # using same images for validation in this example
        "names": {idx: cat.name for idx, cat in enumerate(annotation.categories)}
    }
    
    yaml_path = os.path.join(yolo_dataset_dir, "data.yaml")
    with fileio.open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    logging.info("YOLO format dataset created successfully")
    return register_artifact(yolo_dataset_dir, name="yolo_dataset").uri, visualize_folder_structure(yolo_dataset_dir)


@step(enable_cache=True)
def split_dataset(
    yolo_dataset_dir: str,
    destination_dir: str,
    train_ratio: float = 0.8
) -> Tuple[
    Annotated[str, "split_dataset"],
    Annotated[HTMLString, "split_dataset_structure"]
]:
    """
    Splits a YOLO dataset into train and validation sets.

    Args:
        yolo_dataset_dir: Source directory containing the YOLO dataset
        destination_dir: Target directory for the split dataset
        train_ratio: Ratio of images to use for training (default: 0.8)

    Returns:
        tuple: (
            destination_dir: Path to the split dataset directory,
            dataset_structure: Visualization of the dataset structure
        )
    """
    logging.info("Starting dataset splitting")
    
    # Create train and val directories
    train_images_dir = os.path.join(destination_dir, "train", "images")
    train_labels_dir = os.path.join(destination_dir, "train", "labels")
    val_images_dir = os.path.join(destination_dir, "val", "images")
    val_labels_dir = os.path.join(destination_dir, "val", "labels")
    
    for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        fileio.makedirs(dir_path)

    # Get list of all images and their corresponding labels
    source_images_dir = os.path.join(yolo_dataset_dir, "images")
    source_labels_dir = os.path.join(yolo_dataset_dir, "labels")
    
    all_images = fileio.listdir(source_images_dir)
    random.shuffle(all_images)
    
    # Calculate split point
    split_idx = int(len(all_images) * train_ratio)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    
    logging.info(f"Splitting dataset: {len(train_images)} training, {len(val_images)} validation images")

    # Copy training files
    for image_name in train_images:
        # Copy image
        src_image = os.path.join(source_images_dir, image_name)
        dst_image = os.path.join(train_images_dir, image_name)
        fileio.copy(src_image, dst_image)
        
        # Copy corresponding label
        label_name = os.path.splitext(image_name)[0] + ".txt"
        src_label = os.path.join(source_labels_dir, label_name)
        dst_label = os.path.join(train_labels_dir, label_name)
        fileio.copy(src_label, dst_label)

    # Copy validation files
    for image_name in val_images:
        # Copy image
        src_image = os.path.join(source_images_dir, image_name)
        dst_image = os.path.join(val_images_dir, image_name)
        fileio.copy(src_image, dst_image)
        
        # Copy corresponding label
        label_name = os.path.splitext(image_name)[0] + ".txt"
        src_label = os.path.join(source_labels_dir, label_name)
        dst_label = os.path.join(val_labels_dir, label_name)
        fileio.copy(src_label, dst_label)

    # Copy and modify data.yaml
    src_yaml = os.path.join(yolo_dataset_dir, "data.yaml")
    dst_yaml = os.path.join(destination_dir, "data.yaml")
    
    with fileio.open(src_yaml) as f:
        yaml_content = yaml.safe_load(f)
    
    yaml_content.update({
        "path": ".",  # Relative path
        "train": "train/images",  # Path to train images
        "val": "val/images",      # Path to validation images
    })
    
    with fileio.open(dst_yaml, "w") as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    logging.info("Dataset splitting completed")
    return register_artifact(destination_dir, name="split_dataset").uri, visualize_folder_structure(destination_dir)


@step(
    output_materializers={"yolo_model": UltralyticsMaterializer},
)
def training(yolo_dataset_dir: str, epochs: int = 1) -> Annotated[
    YOLO, ArtifactConfig(
        name="yolo_model",
        artifact_type=ArtifactType.MODEL,
    )
    ]:
    """
    Trains a model on the given data source.

    Args:
        data_source: The path to the COCO dataset in the artifact store.

    Returns:
        The path to the created model in the artifact store.
    """
    logging.info(f"Starting training with data from: {yolo_dataset_dir}")    
    

    temp_dir = tempfile.mkdtemp()
    copy_recursive(yolo_dataset_dir, temp_dir)

    # Load and modify data.yaml path
    yaml_path = os.path.join(temp_dir, "data.yaml")
    with fileio.open(yaml_path) as f:
        data_config = yaml.safe_load(f)
    
    data_config["path"] = temp_dir
    
    with fileio.open(yaml_path, "w") as f:
        yaml.dump(data_config, f, sort_keys=False)

    # traing yolo model
    model = YOLO("yolo11n.pt")
    model.train(data=os.path.join(temp_dir, "data.yaml"), epochs=epochs)

    return model


@pipeline(
    enable_cache=True,
    model=Model(
        name="SimpleYOLO",
        description="A simple YOLO model trained on the tiniest_coco dataset.",
    ),
    settings={
        "docker": DockerSettings(
            requirements="requirements.txt",
            apt_packages=[
                'libgl1-mesa-glx',
                'libglib2.0-0'
            ]
        ),
    },
)
def yolo_training_pipeline(num_chunks: int = 5, unique_id: str = uuid4().hex):
    """
    A simple pipeline to transform a COCO dataset and train a model on it.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info(f"Starting pipeline with {num_chunks} chunks")
    prefix = Client().active_stack.artifact_store.path
    output_root_dir = os.path.join(prefix, unique_id)
    fileio.makedirs(output_root_dir)
    initial_coco_dataset_dir = os.path.join(output_root_dir, "initial_coco_dataset")
    fileio.makedirs(initial_coco_dataset_dir)

    ### ------------- First Step: Create COCO Dataset -------------
    coco_dataset, _, _ = create_coco_dataset(initial_coco_dataset_dir)

    ### ------------- Second Step: Transform COCO Dataset in parallel -------------
    transformed_dataset_coco_dataset_dir = os.path.join(output_root_dir, "transformed_coco_dataset")
    fileio.makedirs(transformed_dataset_coco_dataset_dir)
    after = []
    for i in range(num_chunks):
        transformed_dataset = transform(input_dataset_path=coco_dataset, output_dataset_path=transformed_dataset_coco_dataset_dir, chunk=i, splits=num_chunks, id=f"transform_{i}")
        after.append(f"transform_{i}")

    ### ------------- Third Step: Combine results from all parallel branches -------------
    yolo_dataset_dir = os.path.join(output_root_dir, "yolo_dataset")
    fileio.makedirs(yolo_dataset_dir)
    full_yolo_dataset, _ = combine_step(step_prefix="transform", yolo_dataset_dir=yolo_dataset_dir, after=after)

    ### ------------- Fourth Step: Split dataset -------------
    split_dataset_dir = os.path.join(output_root_dir, "split_dataset")
    fileio.makedirs(split_dataset_dir)
    split_yolo_dataset, _ = split_dataset(yolo_dataset_dir=full_yolo_dataset, destination_dir=split_dataset_dir)

    ### ------------- Fifth Step: Train model -------------
    training(split_yolo_dataset, epochs=1)


if __name__ == "__main__":
    yolo_training_pipeline()