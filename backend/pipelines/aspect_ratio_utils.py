"""
Aspect ratio utilities for ensuring consistent 16:9 output across all pipelines.
"""

from .common_imports import *
from PIL import Image
import cv2

def enforce_16_9_aspect_ratio(image_path: str, output_path: str = None) -> str:
    """
    Enforce 16:9 aspect ratio on an image by cropping or padding.
    
    Args:
        image_path: Path to input image
        output_path: Path for output image (optional, defaults to input_path)
        
    Returns:
        Path to the processed image
    """
    if output_path is None:
        output_path = image_path
        
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            target_ratio = 16 / 9
            current_ratio = width / height
            
            if abs(current_ratio - target_ratio) < 0.01:
                if output_path != image_path:
                    img.save(output_path)
                return output_path
            
            if current_ratio > target_ratio:
                new_width = int(height * target_ratio)
                left = (width - new_width) // 2
                img_cropped = img.crop((left, 0, left + new_width, height))
            else:
                new_height = int(width / target_ratio)
                top = (height - new_height) // 2
                img_cropped = img.crop((0, top, width, top + new_height))
            
            img_cropped.save(output_path)
            logger.info(f"Enforced 16:9 aspect ratio: {image_path} -> {output_path}")
            return output_path
            
    except Exception as e:
        logger.error(f"Failed to enforce aspect ratio on {image_path}: {e}")
        return image_path

def enforce_16_9_video_aspect_ratio(video_path: str, output_path: str = None) -> str:
    """
    Enforce 16:9 aspect ratio on a video by cropping or padding.
    
    Args:
        video_path: Path to input video
        output_path: Path for output video (optional, defaults to input_path)
        
    Returns:
        Path to the processed video
    """
    if output_path is None:
        output_path = video_path
        
    try:
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        target_ratio = 16 / 9
        current_ratio = width / height
        
        if abs(current_ratio - target_ratio) < 0.01:
            cap.release()
            if output_path != video_path:
                import shutil
                shutil.copy2(video_path, output_path)
            return output_path
        
        if current_ratio > target_ratio:
            new_width = int(height * target_ratio)
            x_offset = (width - new_width) // 2
            y_offset = 0
        else:
            new_height = int(width / target_ratio)
            x_offset = 0
            y_offset = (height - new_height) // 2
            new_width = width
            new_height = new_height
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if current_ratio > target_ratio:
                cropped_frame = frame[:, x_offset:x_offset + new_width]
            else:
                cropped_frame = frame[y_offset:y_offset + new_height, :]
                
            out.write(cropped_frame)
        
        cap.release()
        out.release()
        
        logger.info(f"Enforced 16:9 aspect ratio on video: {video_path} -> {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to enforce video aspect ratio on {video_path}: {e}")
        return video_path

def get_16_9_resolution(base_height: int = 1080) -> Tuple[int, int]:
    """
    Get 16:9 resolution based on height.
    
    Args:
        base_height: Target height
        
    Returns:
        Tuple of (width, height) for 16:9 aspect ratio
    """
    width = int(base_height * 16 / 9)
    return (width, base_height)

def validate_16_9_aspect_ratio(width: int, height: int, tolerance: float = 0.01) -> bool:
    """
    Validate if dimensions are close to 16:9 aspect ratio.
    
    Args:
        width: Image/video width
        height: Image/video height
        tolerance: Acceptable deviation from exact 16:9 ratio
        
    Returns:
        True if within tolerance of 16:9 ratio
    """
    target_ratio = 16 / 9
    current_ratio = width / height
    return abs(current_ratio - target_ratio) <= tolerance
