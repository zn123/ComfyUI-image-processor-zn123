"""
Utility functions for image processing
"""

import torch
import numpy as np
from PIL import Image
import comfy.utils


def tensor_to_pil(image_tensor):
    """Convert tensor to PIL Image"""
    if len(image_tensor.shape) == 4:
        # Batch of images
        images = []
        for i in range(image_tensor.shape[0]):
            img_np = image_tensor[i].cpu().numpy()
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
            images.append(img_pil)
        return images
    else:
        # Single image
        img_np = image_tensor.cpu().numpy()
        img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
        return img_pil


def pil_to_tensor(pil_image):
    """Convert PIL Image to tensor"""
    if isinstance(pil_image, list):
        # List of images
        tensors = []
        for img in pil_image:
            img_np = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np)
            tensors.append(img_tensor)
        return torch.stack(tensors)
    else:
        # Single image
        img_np = np.array(pil_image).astype(np.float32) / 255.0
        return torch.from_numpy(img_np)


def validate_image_dimensions(image, min_size=1, max_size=8192):
    """Validate image dimensions"""
    if len(image.shape) == 4:
        height, width = image.shape[2], image.shape[3]
    else:
        height, width = image.shape[0], image.shape[1]
    
    if height < min_size or width < min_size:
        raise ValueError(f"Image dimensions too small: {height}x{width}")
    
    if height > max_size or width > max_size:
        raise ValueError(f"Image dimensions too large: {height}x{width}")
    
    return True


def clamp_value(value, min_val, max_val):
    """Clamp value between min and max"""
    return max(min_val, min(value, max_val))


def calculate_aspect_ratio(width, height):
    """Calculate aspect ratio"""
    if height == 0:
        return float('inf')
    return width / height


def resize_to_fit_within(image, max_width, max_height, method="bicubic"):
    """Resize image to fit within specified dimensions while maintaining aspect ratio"""
    if len(image.shape) == 4:
        current_height, current_width = image.shape[2], image.shape[3]
    else:
        current_height, current_width = image.shape[0], image.shape[1]
    
    # Calculate scaling factor
    width_ratio = max_width / current_width
    height_ratio = max_height / current_height
    scale_factor = min(width_ratio, height_ratio, 1.0)  # Don't upscale
    
    if scale_factor >= 1.0:
        return image  # No resize needed
    
    new_width = int(current_width * scale_factor)
    new_height = int(current_height * scale_factor)
    
    # Use ComfyUI's upscale function
    if len(image.shape) == 4:
        samples = image.movedim(-1, 1)
        resized = comfy.utils.common_upscale(samples, new_width, new_height, method, "disabled")
        return resized.movedim(1, -1)
    else:
        # For single images, convert to PIL and resize
        img_pil = tensor_to_pil(image)
        if isinstance(img_pil, list):
            img_pil = img_pil[0]
        
        resized_pil = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return pil_to_tensor(resized_pil)


def create_grid(images, cols=None, padding=10):
    """Create a grid from multiple images"""
    if not isinstance(images, list):
        images = [images]
    
    num_images = len(images)
    if num_images == 0:
        return None
    
    if cols is None:
        cols = int(np.ceil(np.sqrt(num_images)))
    
    rows = int(np.ceil(num_images / cols))
    
    # Get dimensions of first image
    first_img = tensor_to_pil(images[0])
    if isinstance(first_img, list):
        first_img = first_img[0]
    
    img_width, img_height = first_img.size
    
    # Calculate grid dimensions
    grid_width = cols * img_width + (cols - 1) * padding
    grid_height = rows * img_height + (rows - 1) * padding
    
    # Create blank canvas
    grid = Image.new('RGB', (grid_width, grid_height), (0, 0, 0))
    
    # Paste images
    for i, img_tensor in enumerate(images):
        row = i // cols
        col = i % cols
        
        img_pil = tensor_to_pil(img_tensor)
        if isinstance(img_pil, list):
            img_pil = img_pil[0]
        
        x = col * (img_width + padding)
        y = row * (img_height + padding)
        
        grid.paste(img_pil, (x, y))
    
    return pil_to_tensor(grid)