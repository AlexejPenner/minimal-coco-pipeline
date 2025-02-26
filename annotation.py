from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime
from PIL import Image as PILImage, ImageDraw
import os
from zenml.io import fileio


class License(BaseModel):
    url: str
    id: int
    name: str


class Image(BaseModel):
    license: int
    file_name: str
    coco_url: str
    height: int
    width: int
    date_captured: str
    flickr_url: Optional[str] = None
    id: int


class Category(BaseModel):
    supercategory: str
    id: int
    name: str

class Annotation(BaseModel):
    segmentation: Union[List[List[float]], Dict[str, Any]]
    area: float
    iscrowd: int
    image_id: int
    bbox: List[float]  # [x, y, width, height]
    category_id: int
    id: int


class Info(BaseModel):
    description: str
    url: str
    version: str
    year: int
    contributor: str
    date_created: str


class COCODataset(BaseModel):
    images: List[Image]
    annotations: List[Annotation]
    categories: List[Category]
    type: str = "instances"
    info: Info
    licenses: List[License]

    def get_image_id(self, image_name: str) -> int:
        """
        Get the ID of an image by its file name.
        
        Args:
            image_name: The name of the image to find   

        Returns:
            The ID of the image
        """
        return next((img.id for img in self.images if img.file_name == image_name), None)

    def filter_by_image_ids(self, image_file_names: List[int]) -> 'COCODataset':
        """
        Creates a new COCODataset containing only the specified images and their annotations.
        
        Args:
            image_file_names: List of image file names to keep
            
        Returns:
            A new COCODataset instance with filtered images and annotations
        """
        # Filter images
        filtered_images = [img for img in self.images if img.file_name in image_file_names]

        image_ids = [img.id for img in filtered_images]
        # Filter annotations that correspond to the kept images
        filtered_annotations = [
            ann for ann in self.annotations 
            if ann.image_id in image_ids
        ]
        
        # Create new dataset with filtered data
        reduced_dataset = COCODataset(
            images=filtered_images,
            annotations=filtered_annotations,
            categories=self.categories,
            type=self.type,
            info=self.info,
            licenses=self.licenses
        )
        assert len(reduced_dataset.images) == len(image_file_names)
        return reduced_dataset

    def resize_annotations_for_image(self, image_file_name: str, new_width: int, new_height: int) -> None:
        """
        Resizes all annotations for a specific image based on new dimensions.
        
        Args:
            image_id: ID of the image whose annotations should be resized
            new_width: New width of the image
            new_height: New height of the image
        """
        # Find the original image dimensions
        image = next((img for img in self.images if img.file_name == image_file_name), None)
        if not image:
            raise ValueError(f"Image with ID {image_file_name} not found")
            
        # Calculate scaling factors
        width_scale = new_width / image.width
        height_scale = new_height / image.height
        
        # Update annotations for this image
        for ann in self.annotations:
            if ann.image_id == image.id:
                # Update bbox [x, y, width, height]
                ann.bbox[0] *= width_scale  # x
                ann.bbox[1] *= height_scale  # y
                ann.bbox[2] *= width_scale  # width
                ann.bbox[3] *= height_scale  # height
                
                # Update segmentation coordinates
                if isinstance(ann.segmentation, list):
                    # Handle list of polygons format
                    for polygon in ann.segmentation:
                        for i in range(0, len(polygon), 2):
                            polygon[i] *= width_scale     # x coordinates
                            polygon[i+1] *= height_scale  # y coordinates
                elif isinstance(ann.segmentation, dict):
                    # Handle RLE format if present
                    # Note: RLE format requires more complex handling and might need 
                    # to be recomputed from the polygon format
                    # raise NotImplementedError("RLE segmentation format resizing not implemented")
                    pass
                
                # Update area
                ann.area *= (width_scale * height_scale)
        
        # Update image dimensions
        image.width = new_width
        image.height = new_height

    def merge(self, other: 'COCODataset') -> None:
        """Merge another COCODataset into this one.
        
        Args:
            other: Another COCODataset instance to merge into this one
        """
        # Merge images
        self.images.extend(other.images)
        
        # Merge categories (if there are any new ones)
        existing_category_ids = {cat.id for cat in self.categories}
        for category in other.categories:
            if category.id not in existing_category_ids:
                self.categories.append(category)
                existing_category_ids.add(category.id)
        
        # Merge annotations
        self.annotations.extend(other.annotations)

def draw_annotations_on_image(
    image: PILImage.Image,
    coco_dataset: COCODataset,
    image_name: str,
    draw_boxes: bool = True,
    draw_segments: bool = True,
    box_color: str = "red",
    segment_color: str = "blue",
    line_width: int = 2
) -> Image:
    """
    Draw annotations on a Pillow Image for a specific image in the COCO dataset.
    
    Args:
        image_path: Path to the image file
        coco_dataset: COCODataset instance containing annotations
        image_file_name: Name of the image file to process
        draw_boxes: Whether to draw bounding boxes
        draw_segments: Whether to draw segmentation polygons
        box_color: Color for bounding boxes
        segment_color: Color for segmentation polygons
        line_width: Width of the lines to draw
        
    Returns:
        PIL Image with annotations drawn
    """
    # Open and convert image to RGB
    draw = ImageDraw.Draw(image)
    
    # Get annotations for this image
    image_id = coco_dataset.get_image_id(image_name)
    annotations = [ann for ann in coco_dataset.annotations if ann.image_id == image_id]
    
    for ann in annotations:
        # Draw bounding box
        if draw_boxes:
            x, y, w, h = ann.bbox
            draw.rectangle(
                [(x, y), (x + w, y + h)],
                outline=box_color,
                width=line_width
            )
        
        # Draw segmentation
        if draw_segments and isinstance(ann.segmentation, list):
            for polygon in ann.segmentation:
                # Convert flat list of coordinates to list of tuples
                points = [(polygon[i], polygon[i+1]) 
                         for i in range(0, len(polygon), 2)]
                draw.polygon(points, outline=segment_color, width=line_width)
        
        # Optionally add category name
        category = next(cat for cat in coco_dataset.categories 
                       if cat.id == ann.category_id)
        draw.text((ann.bbox[0], ann.bbox[1] - 10), category.name, 
                 fill=box_color)
    
    return image