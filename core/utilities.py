import csv
import yaml
import torch
import argparse
from pathlib import Path
from typing import Union, List
from PIL import Image
from torchvision import transforms
import numpy as np
from omegaconf import OmegaConf


# ------------------------------------------------------------
# Image lists loading utility
# ------------------------------------------------------------

def get_list_files(image_list_path: Path):
    """
    Get list files from input path.
    
    Parameters
    ----------
    image_list_path : Path
        Path to a image list file or a directory containing image list files.
    
    Returns
    -------
    list_files : List[Path]
        List of image list file paths.
    """
    image_list_path = Path(image_list_path)
    
    if image_list_path.is_file():
        # Single image list file
        if image_list_path.suffix.lower() == ".csv":
            return [image_list_path]
        else:
            raise ValueError(f"File must be a image list file: {image_list_path}")
    elif image_list_path.is_dir():
        # Directory containing image list files
        list_files = sorted(image_list_path.glob("*.csv"))
        if not list_files:
            raise FileNotFoundError(f"No image list files found in directory: {image_list_path}")
        return list_files
    else:
        raise FileNotFoundError(f"Path does not exist: {image_list_path}")

# ------------------------------------------------------------
# Image loading utility
# ------------------------------------------------------------

def load_image(path: Path, device: torch.device) -> torch.Tensor:
    """
    Load image and normalize to [-1, 1].

    Returns
    -------
    image : torch.Tensor
        Shape: (1, 3, H, W)
    """
    img = Image.open(path).convert("RGB")
    tfm = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return tfm(img).unsqueeze(0).to(device)


def load_image_list(csv_file: Path, base_path: Path = None):
    """
    Load image paths from a csv file.
    
    Parameters
    ----------
    csv_file : Path
        Path to CSV file containing image paths.
    base_path : Path, optional
        Base directory path to prepend to relative image paths.
        If None, returns paths as-is.
    """
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file does not exist: {csv_file}")
    
    arch = csv_file.stem.split('_')[0]
    image_files = []

    with csv_file.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = row["imagepath"]
            img_path = Path(img_path.replace('\\', '/'))
            
            if base_path:
                img_path = str(base_path / img_path)
            else:
                img_path = str(img_path)
            image_files.append(img_path)
    return image_files, arch



# ------------------------------------------------------------
# Convert images utility
# ------------------------------------------------------------

def tensor_to_pil(tensor: torch.Tensor) -> Union[Image.Image, List[Image.Image]]:
    """
    Convert tensor images to PIL Image objects.
    
    Converts a batch or single image from tensor format (range [-1, 1])
    to PIL Image format (range [0, 255]). Supports both single images
    and batches.
    
    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor with shape (3, H, W) for single image or (B, 3, H, W) for batch.
    
    Returns
    -------
    Union[Image.Image, List[Image.Image]]
        Single PIL Image object if input shape is (3, H, W),
        List of PIL Image objects if input shape is (B, 3, H, W).
    
    """
    tensor = tensor.detach().cpu()
    
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
        is_single = True
    elif tensor.ndim == 4:
        is_single = False
    else:
        raise ValueError(
            f"Expected tensor with 3 or 4 dimensions, got {tensor.ndim}. "
            f"Shape: {tensor.shape}"
        )
    
    tensor = (tensor * 0.5 + 0.5).clamp(0, 1)
    tensor_np = tensor.numpy()  # (B, 3, H, W)
    tensor_np = (tensor_np * 255).round().astype(np.uint8)
    
    # Convert to PIL Images
    pil_images = []
    for i in range(tensor_np.shape[0]):
        # Shape: (3, H, W) -> transpose to (H, W, 3)
        img_array = np.transpose(tensor_np[i], (1, 2, 0))
        pil_img = Image.fromarray(img_array, 'RGB')
        pil_images.append(pil_img)
    
    if is_single:
        return pil_images[0]
    else:
        return pil_images

