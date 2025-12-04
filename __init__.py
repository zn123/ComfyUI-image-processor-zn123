"""
ComfyUI Image Processor Plugin
A comprehensive image processing plugin for ComfyUI
"""

__version__ = "1.0.0"
__author__ = "zn123"

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def load_nodes():
    """Load all node classes from the plugin"""
    from .nodes import ZNImageProcessorNodes
    return ZNImageProcessorNodes

def register_nodes():
    """Register nodes with ComfyUI"""
    nodes_module = load_nodes()
    
    # Import all node classes
    from .nodes import (
        ZNImageScaleNode,
        ZNImageResizeNode,
        # ZNImageRotateNode,
        # ZNImageFlipNode,
        # ZNImageCropNode,
        ZNImageAdjustBrightnessNode,
        ZNImageAdjustContrastNode,
        # ZNImageBlurNode,
        # ZNImageSharpenNode,
        ZNImageGrayscaleNode,
        ZNImageScaleByShortSideNode,
        ZNImageScaleByLongSideNode,
    )
    
    # Register node mappings
    NODE_CLASS_MAPPINGS.update({
        "ImageUpscale": ZNImageScaleNode,
        "ImageResize": ZNImageResizeNode,
        # "ImageRotate": ZNImageRotateNode,
        # "ImageFlip": ZNImageFlipNode,
        # "ImageCrop": ZNImageCropNode,
        "ImageAdjustBrightness": ZNImageAdjustBrightnessNode,
        "ImageAdjustContrast": ZNImageAdjustContrastNode,
        # "ImageBlur": ZNImageBlurNode,
        # "ImageSharpen": ZNImageSharpenNode,
        "ImageGrayscale": ZNImageGrayscaleNode,
        "ImageScaleByShortSide": ZNImageScaleByShortSideNode,
        "ImageScaleByLongSide": ZNImageScaleByLongSideNode,
    })
    
    NODE_DISPLAY_NAME_MAPPINGS.update({
        "ImageUpscale": "Image Upscale @zn123",
        "ImageResize": "Image Resize @zn123",
        # "ImageRotate": "Image Rotate @zn123",
        # "ImageFlip": "Image Flip @zn123",
        # "ImageCrop": "Image Crop @zn123",
        "ImageAdjustBrightness": "Adjust Brightness @zn123",
        "ImageAdjustContrast": "Adjust Contrast @zn123",
        # "ImageBlur": "Image Blur @zn123",
        # "ImageSharpen": "Image Sharpen @zn123",
        "ImageGrayscale": "Image Grayscale @zn123",
        "ImageScaleByShortSide": "Scale By Short Side @zn123",
        "ImageScaleByLongSide": "Scale By Long Side @zn123",
    })
    
    return NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Auto-register when imported
register_nodes()

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']