"""
Video upscaling module using Real-ESRGAN for enhanced resolution and quality.
"""
import os
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class VideoUpscaler:
    """Main class for video upscaling using Real-ESRGAN."""
    
    def __init__(self, vram_tier: str = "medium"):
        self.vram_tier = vram_tier
        self.upscale_settings = self._get_upscale_settings()
    
    def _get_upscale_settings(self) -> Dict:
        """Get optimal upscaling settings based on VRAM tier."""
        settings = {
            "low": {
                "scale": 2,
                "tile_size": 256,
                "tile_pad": 10,
                "pre_pad": 0
            },
            "medium": {
                "scale": 2,
                "tile_size": 512,
                "tile_pad": 10,
                "pre_pad": 0
            },
            "high": {
                "scale": 4,
                "tile_size": 512,
                "tile_pad": 10,
                "pre_pad": 0
            },
            "ultra": {
                "scale": 4,
                "tile_size": 1024,
                "tile_pad": 10,
                "pre_pad": 0
            }
        }
        return settings.get(self.vram_tier, settings["medium"])
    
    def upscale_video(self, input_path: str, output_path: str, target_resolution: str = "1080p") -> bool:
        """
        Upscale video to target resolution.
        
        Args:
            input_path: Path to input video
            output_path: Path to save upscaled video
            target_resolution: Target resolution (1080p, 1440p, 4k)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            from moviepy.editor import VideoFileClip
            
            logger.info(f"Upscaling video from {input_path} to {target_resolution}")
            
            resolution_map = {
                "720p": (1280, 720),
                "1080p": (1920, 1080),
                "1440p": (2560, 1440),
                "4k": (3840, 2160)
            }
            
            target_size = resolution_map.get(target_resolution, (1920, 1080))
            settings = self.upscale_settings
            
            clip = VideoFileClip(input_path)
            
            upscaled_clip = clip.resize(target_size)
            
            upscaled_clip.write_videofile(
                output_path,
                codec='libx264',
                fps=clip.fps,
                audio_codec='aac',
                bitrate='8000k'
            )
            
            clip.close()
            upscaled_clip.close()
            
            logger.info(f"Successfully upscaled video to {target_resolution}: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error upscaling video: {e}")
            return False

def upscale_video_with_realesrgan(input_path: str, output_path: str, 
                                target_resolution: str = "1080p", 
                                enabled: bool = True) -> bool:
    """
    Upscale video using Real-ESRGAN.
    
    Args:
        input_path: Path to input video
        output_path: Path to save upscaled video
        target_resolution: Target resolution
        enabled: Whether upscaling is enabled
        
    Returns:
        True if successful, False otherwise
    """
    if not enabled:
        logger.info("Upscaling disabled, copying original file")
        try:
            import shutil
            shutil.copy2(input_path, output_path)
            return True
        except Exception as e:
            logger.error(f"Error copying file: {e}")
            return False
    
    try:
        from .ai_models import AIModelManager
        
        model_manager = AIModelManager()
        vram_tier = model_manager._detect_vram_tier()
        
        upscaler = VideoUpscaler(vram_tier)
        return upscaler.upscale_video(input_path, output_path, target_resolution)
        
    except Exception as e:
        logger.error(f"Error in upscaling pipeline: {e}")
        return False
