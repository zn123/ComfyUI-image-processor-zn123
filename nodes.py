"""
Image Processing Nodes for ComfyUI
Comprehensive image processing functionality
"""

import torch
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import comfy.utils
import comfy.model_management


class ZNImageProcessorNodes:
    """Base class for image processing nodes"""
    pass


class ZNImageScaleNode:
    """Image upscale node with multiple methods"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale_by": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "upscale_method": (["nearest", "bilinear", "bicubic", "area", "lanczos"], {"default": "bicubic"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("upscaled_image",)
    FUNCTION = "upscale"
    CATEGORY = "image/processor"
    
    def upscale(self, image, upscale_method, scale_by):
        if scale_by == 1.0:
            return (image,)

        samples = image.movedim(-1, 1)
        width = round(samples.shape[3] * scale_by)
        height = round(samples.shape[2] * scale_by)
        s = comfy.utils.common_upscale(samples, width, height, upscale_method, "disabled")
        s = s.movedim(1, -1)
        return (s,)


class ZNImageScaleByShortSideNode:
    """Image upscale node by short side maintaining aspect ratio"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "short_side_target": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 1}),
                "upscale_method": (["nearest", "bilinear", "bicubic", "area", "lanczos"], {"default": "bicubic"}),
                "only_upscale": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("scaled_image", "width", "height")
    FUNCTION = "scale_by_short_side"
    CATEGORY = "image/processor"

    def scale_by_short_side(self, image, short_side_target, upscale_method, only_upscale):
        # Get image dimensions
        batch_size, img_height, img_width, channels = image.shape

        # Determine current short side
        current_short_side = min(img_width, img_height)

        # Calculate scale factor based on short side
        scale_factor = short_side_target / current_short_side

        # Calculate new dimensions maintaining aspect ratio
        if img_width <= img_height:
            # Width is short side
            new_width = short_side_target
            new_height = round(img_height * scale_factor)
        else:
            # Height is short side
            new_height = short_side_target
            new_width = round(img_width * scale_factor)
        # If only_upscale is True and image is already large enough, return original
        if only_upscale and current_short_side >= short_side_target:
            return (image, img_width, img_height)

        # If the calculated dimensions are the same as the original, return original image
        if new_width == img_width and new_height == img_height:
            return (image, img_width, img_height)

        # Perform the scaling
        samples = image.movedim(-1, 1)
        s = comfy.utils.common_upscale(samples, new_width, new_height, upscale_method, "disabled")
        s = s.movedim(1, -1)

        return (s, new_width, new_height)


class ZNImageScaleByLongSideNode:
    """Image upscale node by long side maintaining aspect ratio"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "long_side_target": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 1}),
                "upscale_method": (["nearest", "bilinear", "bicubic", "area", "lanczos"], {"default": "bicubic"}),
                "only_upscale": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("scaled_image", "width", "height")
    FUNCTION = "scale_by_long_side"
    CATEGORY = "image/processor"

    def scale_by_long_side(self, image, long_side_target, upscale_method, only_upscale):
        # Get image dimensions
        batch_size, img_height, img_width, channels = image.shape

        # Determine current long side
        current_long_side = max(img_width, img_height)

        # Calculate scale factor based on long side
        scale_factor = long_side_target / current_long_side

        # Calculate new dimensions maintaining aspect ratio
        if img_width >= img_height:
            # Width is long side
            new_width = long_side_target
            new_height = round(img_height * scale_factor)
        else:
            # Height is long side
            new_height = long_side_target
            new_width = round(img_width * scale_factor)
        # If only_upscale is True and image is already large enough, return original
        if only_upscale and current_long_side >= long_side_target:
            return (image, img_width, img_height)

        # If the calculated dimensions are the same as the original, return original image
        if new_width == img_width and new_height == img_height:
            return (image, img_width, img_height)
        # Perform the scaling
        samples = image.movedim(-1, 1)
        s = comfy.utils.common_upscale(samples, new_width, new_height, upscale_method, "disabled")
        s = s.movedim(1, -1)

        return (s, new_width, new_height)


class ZNImageScaleByShortSideFactorNode:
    """
    根据输入图片的短边尺寸自动调整缩放比例的节点。
    用户可以设置多个短边尺寸范围，每个范围对应不同的缩放比例。
    例如：
      - 短边小于512像素的图片放大4倍
      - 短边在512-1024之间的图片放大2倍
      - 短边大于1024的图片不缩放
    """


    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_method": (["nearest", "bilinear", "bicubic", "area", "lanczos"], {"default": "bicubic"}),
                # 短边尺寸阈值，小于此尺寸将使用scale_factor1
                "threshold1": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 1}),
                # 短边尺寸阈值，小于此尺寸将使用scale_factor2，大于等于threshold1且小于threshold2
                "threshold2": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 1}),
                # 不同范围的缩放因子
                "scale_factor1": ("FLOAT", {"default": 4.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "scale_factor2": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "scale_factor3": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                # 是否仅放大而不缩小
                "only_upscale": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("scaled_image", "width", "height", "scale_factor", "scale_info")
    FUNCTION = "scale_by_short_side_factor"
    CATEGORY = "image/processor"
    # 添加这一行使节点显示输出值
    OUTPUT_NODE = True

    def scale_by_short_side_factor(self, image, upscale_method, threshold1, threshold2,
                                   scale_factor1, scale_factor2, scale_factor3, only_upscale):
        # Get image dimensions
        batch_size, img_height, img_width, channels = image.shape

        # Determine current short side
        current_short_side = min(img_width, img_height)

        # Determine scale factor based on short side
        if current_short_side < threshold1:
            scale_factor = scale_factor1
            range_info = f"short side < {threshold1}px"
        elif current_short_side < threshold2:
            scale_factor = scale_factor2
            range_info = f"{threshold1}px ≤ short side < {threshold2}px"
        else:
            scale_factor = scale_factor3
            range_info = f"short side ≥ {threshold2}px"

        # If only_upscale is True and scale_factor < 1.0, set to 1.0 (no downscaling)
        if only_upscale and scale_factor < 1.0:
            scale_factor = 1.0
            range_info += " (only upscale mode)"

        # If scale_factor is 1.0, return original image
        if scale_factor == 1.0:
            info = f"Original: {img_width}×{img_height}, Short side: {current_short_side}px, Scale: 1.0x, {range_info}"
            # return (image, img_width, img_height, scale_factor, info)
            return {"ui": {"text": (info,)}, "result": (image, img_width, img_height, scale_factor, info)}
        # Calculate new dimensions
        new_width = round(img_width * scale_factor)
        new_height = round(img_height * scale_factor)

        # Perform the scaling
        samples = image.movedim(-1, 1)
        s = comfy.utils.common_upscale(samples, new_width, new_height, upscale_method, "disabled")
        s = s.movedim(1, -1)

        # Create info string
        info = f"Original: {img_width}×{img_height}, Short side: {current_short_side}px, Scale: {scale_factor:.1f}x, New: {new_width}×{new_height}, {range_info}"
        # return (s, new_width, new_height, scale_factor, info)
        return {"ui": {"text": (info,)}, "result": (s, new_width, new_height, scale_factor, info)}

class ZNImageRotateNode:
    """Image rotation node"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "angle": ("FLOAT", {"default": 90.0, "min": -360.0, "max": 360.0, "step": 1.0}),
                "expand": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rotated_image",)
    FUNCTION = "rotate"
    CATEGORY = "image/processor"
    
    def rotate(self, image, angle, expand):
        batch_size = image.shape[0]
        result = []
        
        for i in range(batch_size):
            img_tensor = image[i]
            img_np = img_tensor.cpu().numpy()
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
            
            rotated = img_pil.rotate(angle, expand=expand, fillcolor=(0, 0, 0))
            rotated_np = np.array(rotated).astype(np.float32) / 255.0
            rotated_tensor = torch.from_numpy(rotated_np)
            
            result.append(rotated_tensor)
        
        result_tensor = torch.stack(result)
        return (result_tensor,)


class ZNImageFlipNode:
    """Image flip node"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "flip_horizontal": ("BOOLEAN", {"default": True}),
                "flip_vertical": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("flipped_image",)
    FUNCTION = "flip"
    CATEGORY = "image/processor"
    
    def flip(self, image, flip_horizontal, flip_vertical):
        if not flip_horizontal and not flip_vertical:
            return (image,)
        
        result = image.clone()
        
        if flip_horizontal:
            result = torch.flip(result, dims=[3])  # flip width dimension
        
        if flip_vertical:
            result = torch.flip(result, dims=[2])  # flip height dimension
        
        return (result,)


class ZNImageCropNode:
    """Image crop node"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "x": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "y": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "width": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("cropped_image",)
    FUNCTION = "crop"
    CATEGORY = "image/processor"
    
    def crop(self, image, x, y, width, height):
        batch_size, img_height, img_width, channels = image.shape
        
        # Clamp values to valid ranges
        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))
        x2 = min(x + width, img_width)
        y2 = min(y + height, img_height)
        
        cropped = image[:, y:y2, x:x2, :]
        return (cropped,)


class ZNImageAdjustBrightnessNode:
    """Image brightness adjustment node"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "brightness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("adjusted_image",)
    FUNCTION = "adjust_brightness"
    CATEGORY = "image/processor"
    
    def adjust_brightness(self, image, brightness):
        batch_size = image.shape[0]
        result = []
        
        for i in range(batch_size):
            img_tensor = image[i]
            img_np = img_tensor.cpu().numpy()
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
            
            enhancer = ImageEnhance.Brightness(img_pil)
            enhanced = enhancer.enhance(brightness)
            enhanced_np = np.array(enhanced).astype(np.float32) / 255.0
            enhanced_tensor = torch.from_numpy(enhanced_np)
            
            result.append(enhanced_tensor)
        
        result_tensor = torch.stack(result)
        return (result_tensor,)


class ZNImageAdjustContrastNode:
    """Image contrast adjustment node"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("adjusted_image",)
    FUNCTION = "adjust_contrast"
    CATEGORY = "image/processor"
    
    def adjust_contrast(self, image, contrast):
        batch_size = image.shape[0]
        result = []
        
        for i in range(batch_size):
            img_tensor = image[i]
            img_np = img_tensor.cpu().numpy()
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
            
            enhancer = ImageEnhance.Contrast(img_pil)
            enhanced = enhancer.enhance(contrast)
            enhanced_np = np.array(enhanced).astype(np.float32) / 255.0
            enhanced_tensor = torch.from_numpy(enhanced_np)
            
            result.append(enhanced_tensor)
        
        result_tensor = torch.stack(result)
        return (result_tensor,)


class ZNImageBlurNode:
    """Image blur node"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "blur_radius": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 20.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("blurred_image",)
    FUNCTION = "blur"
    CATEGORY = "image/processor"
    
    def blur(self, image, blur_radius):
        if blur_radius <= 0:
            return (image,)
        
        batch_size = image.shape[0]
        result = []
        
        for i in range(batch_size):
            img_tensor = image[i]
            img_np = img_tensor.cpu().numpy()
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
            
            blurred = img_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            blurred_np = np.array(blurred).astype(np.float32) / 255.0
            blurred_tensor = torch.from_numpy(blurred_np)
            
            result.append(blurred_tensor)
        
        result_tensor = torch.stack(result)
        return (result_tensor,)


class ZNImageSharpenNode:
    """Image sharpen node"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "sharpen_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("sharpened_image",)
    FUNCTION = "sharpen"
    CATEGORY = "image/processor"
    
    def sharpen(self, image, sharpen_factor):
        batch_size = image.shape[0]
        result = []
        
        for i in range(batch_size):
            img_tensor = image[i]
            img_np = img_tensor.cpu().numpy()
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
            
            enhancer = ImageEnhance.Sharpness(img_pil)
            sharpened = enhancer.enhance(sharpen_factor)
            sharpened_np = np.array(sharpened).astype(np.float32) / 255.0
            sharpened_tensor = torch.from_numpy(sharpened_np)
            
            result.append(sharpened_tensor)
        
        result_tensor = torch.stack(result)
        return (result_tensor,)


class ZNImageGrayscaleNode:
    """Image grayscale conversion node"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("grayscale_image",)
    FUNCTION = "to_grayscale"
    CATEGORY = "image/processor"
    
    def to_grayscale(self, image):
        batch_size = image.shape[0]
        result = []
        
        for i in range(batch_size):
            img_tensor = image[i]
            img_np = img_tensor.cpu().numpy()
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
            
            grayscale = img_pil.convert('L')
            # Convert back to RGB for consistency
            grayscale_rgb = grayscale.convert('RGB')
            grayscale_np = np.array(grayscale_rgb).astype(np.float32) / 255.0
            grayscale_tensor = torch.from_numpy(grayscale_np)
            
            result.append(grayscale_tensor)
        
        result_tensor = torch.stack(result)
        return (result_tensor,)


class ZNImageResizeNode:
    """Image resize node with exact dimensions"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "resize_method": (["nearest", "bilinear", "bicubic", "lanczos"], {"default": "bicubic"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("resized_image",)
    FUNCTION = "resize"
    CATEGORY = "image/processor"
    
    def resize(self, image, width, height, resize_method):
        samples = image.movedim(-1, 1)
        s = comfy.utils.common_upscale(samples, width, height, resize_method, "disabled")
        s = s.movedim(1, -1)
        return (s,)