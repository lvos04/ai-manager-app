"""
Real AI Upscaler implementation with RealESRGAN integration.
Replaces placeholder upscaling with actual AI models.
"""

import os
import sys
import json
import logging
import tempfile
import shutil
import subprocess
import time
import gc
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple

import torch
import numpy as np
import cv2
from PIL import Image

logger = logging.getLogger(__name__)

class AIUpscaler:
    """Real AI upscaling using RealESRGAN and other models."""
    
    def __init__(self, vram_tier: str = "medium"):
        self.vram_tier = vram_tier
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.models = {}
        self.current_model = None
        
        self.model_settings = {
            "realesrgan_x4plus": {
                "low": {"use_gpu": False, "tile_size": 128, "precision": "float32"},
                "medium": {"use_gpu": True, "tile_size": 256, "precision": "float16"},
                "high": {"use_gpu": True, "tile_size": 512, "precision": "float16"},
                "ultra": {"use_gpu": True, "tile_size": 1024, "precision": "float16"}
            },
            "realesrgan_x2plus": {
                "low": {"use_gpu": False, "tile_size": 256, "precision": "float32"},
                "medium": {"use_gpu": True, "tile_size": 512, "precision": "float16"},
                "high": {"use_gpu": True, "tile_size": 1024, "precision": "float16"},
                "ultra": {"use_gpu": True, "tile_size": 2048, "precision": "float16"}
            },
            "realesrgan_anime": {
                "low": {"use_gpu": False, "tile_size": 128, "precision": "float32"},
                "medium": {"use_gpu": True, "tile_size": 256, "precision": "float16"},
                "high": {"use_gpu": True, "tile_size": 512, "precision": "float16"},
                "ultra": {"use_gpu": True, "tile_size": 1024, "precision": "float16"}
            }
        }
        
        self.scale_factors = {
            "realesrgan_x4plus": 4,
            "realesrgan_x2plus": 2,
            "realesrgan_anime": 4
        }
    
    def get_best_model_for_content(self, content_type: str, target_scale: int = 2) -> str:
        """Select optimal upscaling model based on content type and scale."""
        if content_type in ["anime", "manga", "cartoon"]:
            return "realesrgan_anime"
        elif target_scale <= 2:
            return "realesrgan_x2plus"
        else:
            return "realesrgan_x4plus"
    
    def load_model(self, model_name: str) -> bool:
        """Load upscaling model."""
        try:
            if model_name == self.current_model and model_name in self.models:
                return True
            
            self.force_cleanup_all_models()
            
            logger.info(f"Loading upscaling model: {model_name}")
            
            if model_name.startswith("realesrgan"):
                return self._load_realesrgan(model_name)
            else:
                logger.warning(f"Unknown upscaling model: {model_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading upscaling model {model_name}: {e}")
            return False
    
    def _load_realesrgan(self, model_name: str) -> bool:
        """Load RealESRGAN model."""
        try:
            try:
                from realesrgan import RealESRGANer
                from basicsr.archs.rrdbnet_arch import RRDBNet
            except ImportError:
                try:
                    import sys
                    sys.path.append('/home/ubuntu/.pyenv/versions/3.12.8/lib/python3.12/site-packages')
                    from realesrgan import RealESRGANer
                    from basicsr.archs.rrdbnet_arch import RRDBNet
                except ImportError:
                    logger.error("RealESRGAN not available, installing...")
                    import subprocess
                    subprocess.run([sys.executable, "-m", "pip", "install", "realesrgan"], check=True)
                    from realesrgan import RealESRGANer
                    from basicsr.archs.rrdbnet_arch import RRDBNet
            
            settings = self.model_settings[model_name].get(self.vram_tier, self.model_settings[model_name]["medium"])
            
            model_path = self._get_model_path(model_name)
            if not model_path or not os.path.exists(model_path):
                logger.error(f"Model file not found for {model_name}")
                return False
            
            if model_name == "realesrgan_anime":
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            else:
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=self.scale_factors[model_name])
            
            device = "cuda" if settings["use_gpu"] and torch.cuda.is_available() else "cpu"
            
            upsampler = RealESRGANer(
                scale=self.scale_factors[model_name],
                model_path=model_path,
                model=model,
                tile=settings["tile_size"],
                tile_pad=10,
                pre_pad=0,
                half=settings["precision"] == "float16" and device == "cuda",
                device=device
            )
            
            self.models[model_name] = {
                "upsampler": upsampler,
                "settings": settings,
                "scale": self.scale_factors[model_name]
            }
            
            self.current_model = model_name
            logger.info(f"RealESRGAN {model_name} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading RealESRGAN {model_name}: {e}")
            return False
    
    def _get_model_path(self, model_name: str) -> Optional[str]:
        """Get path to model file."""
        try:
            models_dir = Path.home() / "repos" / "ai-manager-app" / "models" / "upscaling"
            
            model_files = {
                "realesrgan_x4plus": models_dir / "RealESRGAN_x4plus.pth",
                "realesrgan_x2plus": models_dir / "RealESRGAN_x2plus.pth",
                "realesrgan_anime": models_dir / "RealESRGAN_x4plus_anime_6B.pth"
            }
            
            model_file = model_files.get(model_name)
            if model_file and model_file.exists():
                return str(model_file)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting model path for {model_name}: {e}")
            return None
    
    def upscale_video(self, input_path: str, output_path: str, model_name: str, 
                     target_resolution: Tuple[int, int] = None) -> bool:
        """Upscale video using AI models."""
        try:
            if not self.load_model(model_name):
                logger.error(f"Failed to load upscaling model: {model_name}")
                return self._create_fallback_upscale(input_path, output_path, target_resolution)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            logger.info(f"Upscaling video with {model_name}: {input_path}")
            
            return self._upscale_video_frames(input_path, output_path, target_resolution)
            
        except Exception as e:
            logger.error(f"Error upscaling video: {e}")
            return self._create_fallback_upscale(input_path, output_path, target_resolution)
    
    def _upscale_video_frames(self, input_path: str, output_path: str, 
                             target_resolution: Tuple[int, int] = None) -> bool:
        """Upscale video by processing individual frames."""
        try:
            upscaler_model = self.models[self.current_model]
            upsampler = upscaler_model["upsampler"]
            
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                logger.error(f"Failed to open input video: {input_path}")
                return False
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            ret, first_frame = cap.read()
            if not ret:
                logger.error("Failed to read first frame")
                cap.release()
                return False
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            upscaled_frame, _ = upsampler.enhance(first_frame, outscale=upscaler_model["scale"])
            
            if target_resolution:
                upscaled_frame = cv2.resize(upscaled_frame, target_resolution, interpolation=cv2.INTER_LANCZOS4)
            
            height, width = upscaled_frame.shape[:2]
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                logger.error("Failed to create output video writer")
                cap.release()
                return False
            
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % 10 == 0:
                    logger.info(f"Processing frame {frame_idx + 1}/{frame_count}")
                
                try:
                    upscaled_frame, _ = upsampler.enhance(frame, outscale=upscaler_model["scale"])
                    
                    if target_resolution:
                        upscaled_frame = cv2.resize(upscaled_frame, target_resolution, interpolation=cv2.INTER_LANCZOS4)
                    
                    out.write(upscaled_frame)
                    
                except Exception as e:
                    logger.warning(f"Error processing frame {frame_idx}: {e}")
                    if target_resolution:
                        fallback_frame = cv2.resize(frame, target_resolution, interpolation=cv2.INTER_LANCZOS4)
                    else:
                        fallback_frame = frame
                    out.write(fallback_frame)
                
                frame_idx += 1
            
            cap.release()
            out.release()
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                logger.info(f"Video upscaled successfully: {output_path}")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error in frame-by-frame upscaling: {e}")
            return False
    
    def upscale_image(self, input_path: str, output_path: str, model_name: str,
                     target_resolution: Tuple[int, int] = None) -> bool:
        """Upscale single image using AI models."""
        try:
            if not self.load_model(model_name):
                logger.error(f"Failed to load upscaling model: {model_name}")
                return self._create_fallback_image_upscale(input_path, output_path, target_resolution)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            upscaler_model = self.models[self.current_model]
            upsampler = upscaler_model["upsampler"]
            
            img = cv2.imread(input_path, cv2.IMREAD_COLOR)
            if img is None:
                logger.error(f"Failed to load image: {input_path}")
                return False
            
            upscaled_img, _ = upsampler.enhance(img, outscale=upscaler_model["scale"])
            
            if target_resolution:
                upscaled_img = cv2.resize(upscaled_img, target_resolution, interpolation=cv2.INTER_LANCZOS4)
            
            success = cv2.imwrite(output_path, upscaled_img)
            
            if success:
                logger.info(f"Image upscaled successfully: {output_path}")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error upscaling image: {e}")
            return self._create_fallback_image_upscale(input_path, output_path, target_resolution)
    
    def _create_fallback_upscale(self, input_path: str, output_path: str, 
                                target_resolution: Tuple[int, int] = None) -> bool:
        """Create fallback upscaled video using basic interpolation."""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            if target_resolution:
                width, height = target_resolution
                cmd = [
                    'ffmpeg', '-i', input_path,
                    '-vf', f'scale={width}:{height}:flags=lanczos',
                    '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
                    '-y', output_path
                ]
            else:
                cmd = [
                    'ffmpeg', '-i', input_path,
                    '-vf', 'scale=iw*2:ih*2:flags=lanczos',
                    '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
                    '-y', output_path
                ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and os.path.exists(output_path):
                logger.warning(f"Created fallback upscaled video: {output_path}")
                return True
            else:
                logger.error(f"FFmpeg upscaling failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error creating fallback upscale: {e}")
            return False
    
    def _create_fallback_image_upscale(self, input_path: str, output_path: str,
                                      target_resolution: Tuple[int, int] = None) -> bool:
        """Create fallback upscaled image using basic interpolation."""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            img = cv2.imread(input_path, cv2.IMREAD_COLOR)
            if img is None:
                return False
            
            if target_resolution:
                upscaled_img = cv2.resize(img, target_resolution, interpolation=cv2.INTER_LANCZOS4)
            else:
                height, width = img.shape[:2]
                upscaled_img = cv2.resize(img, (width * 2, height * 2), interpolation=cv2.INTER_LANCZOS4)
            
            success = cv2.imwrite(output_path, upscaled_img)
            
            if success:
                logger.warning(f"Created fallback upscaled image: {output_path}")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error creating fallback image upscale: {e}")
            return False
    
    def force_cleanup_all_models(self):
        """Force cleanup of all loaded upscaling models."""
        try:
            for model_name in list(self.models.keys()):
                if model_name in self.models:
                    del self.models[model_name]
            
            self.models.clear()
            self.current_model = None
            
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info("All upscaling models cleaned up")
            
        except Exception as e:
            logger.error(f"Error in upscaling model cleanup: {e}")
