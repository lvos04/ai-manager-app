"""
Frame interpolation utility for AI Project Manager.
Uses AI models to generate intermediate frames for smoother video output.
"""
import cv2
import numpy as np
import hashlib
import shutil
from pathlib import Path
from typing import List, Optional
import logging
try:
    from moviepy.editor import VideoFileClip, ImageSequenceClip
except ImportError:
    VideoFileClip = None
    ImageSequenceClip = None
import shutil

logger = logging.getLogger(__name__)

class FrameInterpolator:
    """AI-powered frame interpolation for video enhancement."""
    
    def __init__(self, vram_tier: str = "medium"):
        self.vram_tier = vram_tier
        self.interpolation_settings = self._get_interpolation_settings()
        self.cache_dir = Path("cache/frame_interpolation")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_interpolation_settings(self):
        """Get optimal interpolation settings based on VRAM tier."""
        settings = {
            "low": {"method": "opencv", "quality": "fast"},
            "medium": {"method": "deforum", "quality": "balanced"},
            "high": {"method": "deforum", "quality": "high"},
            "ultra": {"method": "deforum", "quality": "ultra"}
        }
        return settings.get(self.vram_tier, settings["medium"])
    
    def _get_cache_key(self, input_path: str, render_fps: int, output_fps: int) -> str:
        """Generate cache key for interpolation results."""
        content_hash = hashlib.md5(f"{input_path}_{render_fps}_{output_fps}_{self.vram_tier}".encode()).hexdigest()
        return f"interpolation_{content_hash}.mp4"
    
    def _check_cache(self, cache_key: str) -> Optional[str]:
        """Check if cached interpolation result exists."""
        cache_path = self.cache_dir / cache_key
        return str(cache_path) if cache_path.exists() else None
    
    def _save_to_cache(self, output_path: str, cache_key: str):
        """Save interpolation result to cache."""
        cache_path = self.cache_dir / cache_key
        shutil.copy2(output_path, cache_path)
    
    def interpolate_video(self, input_path: str, output_path: str, 
                        render_fps: int, output_fps: int) -> bool:
        """
        Interpolate frames to increase video FPS.
        
        Args:
            input_path: Path to input video
            output_path: Path to output interpolated video
            render_fps: Original render FPS
            output_fps: Target output FPS
            
        Returns:
            bool: Success status
        """
        try:
            if output_fps <= render_fps:
                logger.info("No interpolation needed, output FPS <= render FPS")
                return self._copy_video(input_path, output_path)
            
            cache_key = self._get_cache_key(input_path, render_fps, output_fps)
            cached_result = self._check_cache(cache_key)
            if cached_result:
                logger.info("Using cached interpolation result")
                shutil.copy2(cached_result, output_path)
                return True
            
            interpolation_factor = output_fps // render_fps
            logger.info(f"Interpolating {interpolation_factor}x frames: {render_fps}fps -> {output_fps}fps")
            
            success = False
            if self.interpolation_settings["method"] == "opencv":
                success = self._opencv_interpolation(input_path, output_path, interpolation_factor)
            else:
                success = self._ai_interpolation(input_path, output_path, interpolation_factor)
            
            if success:
                self._save_to_cache(output_path, cache_key)
            
            return success
                
        except Exception as e:
            logger.error(f"Frame interpolation failed: {e}")
            return False
    
    def _copy_video(self, input_path: str, output_path: str) -> bool:
        """Copy video without interpolation."""
        try:
            if VideoFileClip is None:
                logger.warning("MoviePy not available, using file copy")
                shutil.copy2(input_path, output_path)
                return True
            shutil.copy2(input_path, output_path)
            return True
        except Exception as e:
            logger.error(f"Video copy failed: {e}")
            return False
    
    def _opencv_interpolation(self, input_path: str, output_path: str, factor: int) -> bool:
        """Basic OpenCV-based frame interpolation."""
        try:
            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps * factor, (width, height))
            
            prev_frame = None
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                out.write(frame)
                
                if prev_frame is not None:
                    for i in range(1, factor):
                        alpha = i / factor
                        interpolated = cv2.addWeighted(prev_frame, 1-alpha, frame, alpha, 0)
                        out.write(interpolated)
                
                prev_frame = frame.copy()
            
            cap.release()
            out.release()
            return True
        except Exception as e:
            logger.error(f"OpenCV interpolation failed: {e}")
            return False
    
    def _ai_interpolation(self, input_path: str, output_path: str, factor: int) -> bool:
        """AI-based frame interpolation using Deforum or similar models."""
        try:
            logger.info("AI interpolation not yet implemented, falling back to OpenCV")
            return self._opencv_interpolation(input_path, output_path, factor)
        except Exception as e:
            logger.error(f"AI interpolation failed: {e}")
            return False

def interpolate_video_frames(input_path: str, output_path: str, render_fps: int, output_fps: int, vram_tier: str = "medium") -> bool:
    """
    Convenience function for frame interpolation.
    
    Args:
        input_path: Path to input video
        output_path: Path to output interpolated video
        render_fps: Original render FPS
        output_fps: Target output FPS
        vram_tier: VRAM tier for optimization
        
    Returns:
        bool: Success status
    """
    interpolator = FrameInterpolator(vram_tier)
    return interpolator.interpolate_video(input_path, output_path, render_fps, output_fps)
