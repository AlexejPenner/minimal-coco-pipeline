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
from typing_extensions import Annotated
from pydantic import BaseModel
from zenml import get_pipeline_context, get_step_context, pipeline, step, register_artifact
from zenml.client import Client 
from zenml.enums import ArtifactType
from zenml.types import HTMLString
from zenml.io import fileio
from annotation import COCODataset, draw_annotations_on_image
from utils import visualize_folder_structure, copy_recursive



def do_transform(
    image_base_path: str,
    image_name: str,
    annotations: COCODataset
    ) -> Tuple[PILImage.Image, COCODataset]:
    """
    Transforms the image and the corresponding annotations.

    Args:
        image_base_path: The path to the base directory of the image.
        image_name: The name of the image to transform.
        annotations: The annotations of the image.

    Returns:
        A tuple containing:
            - The transformed image.
            - The transformed annotations.
    """
    logging.info(f"Transforming image: {image_name}")
    # do some transfomration on the image and the corresponding annotations
    image = PILImage.open(fileio.open(os.path.join(image_base_path, image_name), "rb"))
    # get original image size
    original_width, original_height = image.size
    logging.debug(f"Original image size: {original_width}x{original_height}")

    # convert the image to RGB
    image = image.convert("RGB")
    image = image.resize((224, 224))
    logging.debug("Image resized to 224x224")

    # update the annotations of this specific image to the new image size
    annotations.resize_annotations_for_image(image_name, 224, 224)
    logging.debug(f"Updated annotations for {image_name}")

    return image, annotations


@step(enable_cache=False)
def create_coco_dataset() -> Tuple[
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

    prefix = Client().active_stack.artifact_store.path
    output_root_dir = os.path.join(prefix, uuid4().hex)
    fileio.makedirs(output_root_dir)
    logging.info(f"Output directory: {output_root_dir}")

    url = "https://github.com/AlexejPenner/tiniest_coco/archive/refs/heads/main.zip"
    logging.info(f"Downloading dataset from {url}")

    # Create temp dir for extraction
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
        name="tiniest_coco",
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


@step()
def transform(
    dataset_path: str,
    chunk: int,
    splits: int = 5
    ) -> Tuple[
        Annotated[Dict[str, PILImage.Image], "transformed_images"],
        Annotated[COCODataset, "transformed_annotations"],
        Annotated[PILImage.Image, "example_image"]
        ]:
    """
    Transforms the dataset by resizing the images and resizing the annotations.

    Args:
        dataset_path: The path to the COCO dataset in the artifact store.
        chunk: The current chunk number.
        splits: The total number of chunks to split the dataset into.

    Returns:
        A tuple containing:
            - A dictionary of transformed images.
            - The transformed annotations.
            - An example image with annotations.
    """
    logging.info(f"Starting transformation of chunk {chunk}/{splits}")
    
    images_dir = os.path.join(dataset_path, "images")
    list_image_names = fileio.listdir(images_dir)
    start_index = (chunk) * len(list_image_names) // splits
    end_index = (chunk+1) * len(list_image_names) // splits
    current_chunk_image_names = list_image_names[start_index:end_index]
    logging.info(f"Processing {len(current_chunk_image_names)} images in chunk {chunk}")

    annotations_path = os.path.join(os.path.join(dataset_path, "annotations"), "instances.json")
    logging.info(f"Loading annotations from {annotations_path}")
    # load cocodataset from annotations_path
    annotations = COCODataset.model_validate_json(
        fileio.open(annotations_path, "r").read()
    )
    logging.debug("Loaded annotations successfully")
    
    transformed_images: Dict[str, PILImage.Image] = {}
    for image_name in current_chunk_image_names:
        image, annotations = do_transform(
            image_base_path=images_dir, 
            image_name=image_name, 
            annotations=annotations
            )
        transformed_images[image_name] = image
    
    # Draw annotations on a random image as a visual check
    random_image = random.choice(list(transformed_images.keys()))

    annotated_image = draw_annotations_on_image(
        image=transformed_images[random_image],
        coco_dataset=annotations,
        image_name=random_image,
        draw_boxes=True,
        draw_segments=True
    )

    return transformed_images, annotations, annotated_image

@step(enable_cache=False)
def combine_step(
    step_prefix: str
    ) -> Tuple[
        Annotated[str, "full_coco_dataset"],
        Annotated[HTMLString, "full_coco_dataset_structure"]
        ]:
    """
    Combines the processed chunks into a single COCO dataset.

    Args:
        step_prefix: The prefix of the steps to combine.

    Returns:
        The path to the created COCO dataset in the artifact store.
    """
    logging.info("Starting combination of processed chunks")
    run_name = get_step_context().pipeline_run.name
    run = Client().get_pipeline_run(run_name)

    # Fetch all results from parallel processing steps
    processed_image_sources: List[str] = []
    processed_annotations: List[COCODataset] = []
    for step_name, step_info in run.steps.items():
        if step_name.startswith(step_prefix):
            logging.info(f"Processing step: {step_name}")
            processed_image_sources.append(
                os.path.join(
                    step_info.outputs["transformed_images"][0].body.uri,
                    "metadata.json"
                )
            )
            processed_annotations.append(step_info.outputs["transformed_annotations"][0].load())
            logging.info(f"Added annotations from {step_name}")
    
    logging.info(f"Found {len(processed_annotations)} chunks to combine")
    
    # Merge the annotations into one single COCODataset
    annotation = processed_annotations[0]
    logging.info(f"Starting with base annotations containing {len(annotation.images)} images")
    for i, annotations in enumerate(processed_annotations[1:], 1):
        logging.info(f"Merging chunk {i}")
        annotation.merge(annotations)

    prefix = Client().active_stack.artifact_store.path
    output_root_dir = os.path.join(prefix, uuid4().hex)
    logging.info(f"Created output directory: {output_root_dir}")

    # Copy all images to a new directory
    images_dir = os.path.join(output_root_dir, "images")
    fileio.makedirs(images_dir)
    logging.info(f"Processing {len(processed_image_sources)} image source files")

    for idx, image_sources_json in enumerate(processed_image_sources):
        logging.info(f"Processing image source {idx + 1}/{len(processed_image_sources)}")
        
        # Load json from file
        with fileio.open(image_sources_json, "r") as f:
            dict_data = json.load(f)

        original_file_names_json_path = os.path.join(dict_data[0]["path"], "data.json")

        # Load json from local file
        with fileio.open(original_file_names_json_path, "r") as f:
            original_file_names = json.load(f)

        image_count = 0
        image_folder_base_path = dict_data[1]["path"]
        for i in range(len(original_file_names)):
            image_path = os.path.join(image_folder_base_path, str(i), "image_file.PNG")
            # move all folder contents to the new directory
            dest_path = os.path.join(output_root_dir, "images", original_file_names[i])
            fileio.copy(image_path, dest_path)
            image_count += 1
        logging.info(f"Copied {image_count} images from source {idx + 1}")

    total_images = len(fileio.listdir(images_dir))
    logging.info(f"Total images copied to output directory: {total_images}")

    # save the annotations to a json file
    annotations_dir = os.path.join(output_root_dir, "annotations")
    fileio.makedirs(annotations_dir)
    annotations_file = os.path.join(annotations_dir, "instances.json")
    logging.debug(f"Saving annotations to {annotations_file}")
    with fileio.open(annotations_file, "w") as f:
        f.write(annotation.model_dump_json(indent=2))
    logging.info("Annotations saved successfully")

    return register_artifact(output_root_dir, name="full_coco_dataset").uri, visualize_folder_structure(output_root_dir)

@step
def training(data_source: str) -> Annotated[str, "model"]:
    """
    Trains a model on the given data source.

    Args:
        data_source: The path to the COCO dataset in the artifact store.

    Returns:
        The path to the created model in the artifact store.
    """
    logging.info(f"Starting training with data from: {data_source}")    
    images_dir = os.path.join(data_source, "images")
    list_image_names = fileio.listdir(images_dir)

    annotations_path = os.path.join(os.path.join(data_source, "annotations"), "instances.json")
    logging.info(f"Loading annotations from {annotations_path}")
    annotations = COCODataset.model_validate_json(
        fileio.open(annotations_path, "r").read()
    )

    model = "blah" # Dummy model
    return model


@pipeline
def simple_cv_pipeline(num_chunks: int = 5):
    """
    A simple pipeline to transform a COCO dataset and train a model on it.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info(f"Starting pipeline with {num_chunks} chunks")
    
    coco_dataset, _, _ = create_coco_dataset()

    # Fan out: Run the transform step in parallel
    after = []
    for i in range(num_chunks):
        transformed_dataset = transform(dataset_path=coco_dataset, chunk=i, splits=num_chunks, id=f"transform_{i}")
        after.append(f"transform_{i}")

    # Fan in: Combine results from all parallel branches
    full_coco_dataset, _ = combine_step(step_prefix="transform", after=after)
    training(full_coco_dataset)


if __name__ == "__main__":
    simple_cv_pipeline()