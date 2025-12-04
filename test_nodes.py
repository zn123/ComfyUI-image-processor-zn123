"""
Test script for image processing nodes
"""

import torch
import numpy as np
from PIL import Image
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nodes import (
    ImageUpscaleNode,
    ImageRotateNode,
    ImageFlipNode,
    ImageCropNode,
    ImageAdjustBrightnessNode,
    ImageAdjustContrastNode,
    ImageBlurNode,
    ImageSharpenNode,
    ImageGrayscaleNode,
    ImageResizeNode
)


def create_test_image(width=512, height=512):
    """Create a test image with some patterns"""
    # Create a simple gradient test image
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    X, Y = np.meshgrid(x, y)
    
    # Create RGB channels
    R = X
    G = Y
    B = (X + Y) / 2
    
    # Stack to create RGB image
    img_array = np.stack([R, G, B], axis=-1)
    
    return torch.from_numpy(img_array.astype(np.float32)).unsqueeze(0)  # Add batch dimension


def test_upscale_node():
    """Test the upscale node"""
    print("Testing ImageUpscaleNode...")
    
    node = ImageUpscaleNode()
    test_img = create_test_image(256, 256)
    
    # Test different upscale methods
    methods = ["nearest", "bilinear", "bicubic", "area", "lanczos"]
    for method in methods:
        result = node.upscale(test_img, method, 2.0)
        print(f"  {method}: {result[0].shape} (expected: [1, 512, 512, 3])")
        assert result[0].shape == (1, 512, 512, 3), f"Upscale failed for {method}"
    
    print("  ✓ Upscale node tests passed")


def test_rotate_node():
    """Test the rotate node"""
    print("Testing ImageRotateNode...")
    
    node = ImageRotateNode()
    test_img = create_test_image(256, 256)
    
    # Test rotation
    result = node.rotate(test_img, 90, True)
    print(f"  90° rotation: {result[0].shape}")
    assert result[0].shape == (1, 256, 256, 3), "Rotation failed"
    
    # Test 45 degree rotation with expansion
    result = node.rotate(test_img, 45, True)
    print(f"  45° rotation with expansion: {result[0].shape}")
    assert result[0].shape[0] == 1, "Rotation failed"
    assert result[0].shape[3] == 3, "Rotation failed"
    
    print("  ✓ Rotate node tests passed")


def test_flip_node():
    """Test the flip node"""
    print("Testing ImageFlipNode...")
    
    node = ImageFlipNode()
    test_img = create_test_image(256, 256)
    
    # Test horizontal flip
    result = node.flip(test_img, True, False)
    print(f"  Horizontal flip: {result[0].shape}")
    assert result[0].shape == (1, 256, 256, 3), "Horizontal flip failed"
    
    # Test vertical flip
    result = node.flip(test_img, False, True)
    print(f"  Vertical flip: {result[0].shape}")
    assert result[0].shape == (1, 256, 256, 3), "Vertical flip failed"
    
    # Test both flips
    result = node.flip(test_img, True, True)
    print(f"  Both flips: {result[0].shape}")
    assert result[0].shape == (1, 256, 256, 3), "Both flips failed"
    
    print("  ✓ Flip node tests passed")


def test_crop_node():
    """Test the crop node"""
    print("Testing ImageCropNode...")
    
    node = ImageCropNode()
    test_img = create_test_image(512, 512)
    
    # Test crop
    result = node.crop(test_img, 100, 100, 256, 256)
    print(f"  Crop: {result[0].shape} (expected: [1, 256, 256, 3])")
    assert result[0].shape == (1, 256, 256, 3), "Crop failed"
    
    # Test edge cases
    result = node.crop(test_img, 0, 0, 512, 512)
    assert result[0].shape == (1, 512, 512, 3), "Full image crop failed"
    
    print("  ✓ Crop node tests passed")


def test_adjustment_nodes():
    """Test brightness and contrast adjustment nodes"""
    print("Testing adjustment nodes...")
    
    brightness_node = ImageAdjustBrightnessNode()
    contrast_node = ImageAdjustContrastNode()
    test_img = create_test_image(256, 256)
    
    # Test brightness
    result = brightness_node.adjust_brightness(test_img, 1.5)
    print(f"  Brightness adjustment: {result[0].shape}")
    assert result[0].shape == (1, 256, 256, 3), "Brightness adjustment failed"
    
    # Test contrast
    result = contrast_node.adjust_contrast(test_img, 1.2)
    print(f"  Contrast adjustment: {result[0].shape}")
    assert result[0].shape == (1, 256, 256, 3), "Contrast adjustment failed"
    
    print("  ✓ Adjustment node tests passed")


def test_effect_nodes():
    """Test blur and sharpen nodes"""
    print("Testing effect nodes...")
    
    blur_node = ImageBlurNode()
    sharpen_node = ImageSharpenNode()
    test_img = create_test_image(256, 256)
    
    # Test blur
    result = blur_node.blur(test_img, 2.0)
    print(f"  Blur: {result[0].shape}")
    assert result[0].shape == (1, 256, 256, 3), "Blur failed"
    
    # Test sharpen
    result = sharpen_node.sharpen(test_img, 1.5)
    print(f"  Sharpen: {result[0].shape}")
    assert result[0].shape == (1, 256, 256, 3), "Sharpen failed"
    
    print("  ✓ Effect node tests passed")


def test_grayscale_node():
    """Test grayscale node"""
    print("Testing ImageGrayscaleNode...")
    
    node = ImageGrayscaleNode()
    test_img = create_test_image(256, 256)
    
    result = node.to_grayscale(test_img)
    print(f"  Grayscale: {result[0].shape}")
    assert result[0].shape == (1, 256, 256, 3), "Grayscale conversion failed"
    
    print("  ✓ Grayscale node tests passed")


def test_resize_node():
    """Test resize node"""
    print("Testing ImageResizeNode...")
    
    node = ImageResizeNode()
    test_img = create_test_image(512, 256)
    
    # Test resize to square
    result = node.resize(test_img, 256, 256, "bilinear")
    print(f"  Resize to 256x256: {result[0].shape}")
    assert result[0].shape == (1, 256, 256, 3), "Resize failed"
    
    # Test resize with different methods
    methods = ["nearest", "bilinear", "bicubic", "lanczos"]
    for method in methods:
        result = node.resize(test_img, 128, 128, method)
        assert result[0].shape == (1, 128, 128, 3), f"Resize failed for {method}"
    
    print("  ✓ Resize node tests passed")


def run_all_tests():
    """Run all tests"""
    print("Running Image Processor Plugin Tests...\n")
    
    try:
        test_upscale_node()
        test_rotate_node()
        test_flip_node()
        test_crop_node()
        test_adjustment_nodes()
        test_effect_nodes()
        test_grayscale_node()
        test_resize_node()
        
        print("\n✅ All tests passed! The plugin is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()