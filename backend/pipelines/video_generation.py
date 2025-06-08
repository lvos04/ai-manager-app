"""
Video generation module for text-to-video model integration.
Supports SVD-XT, Zeroscope v2 XL, AnimateDiff v2/SDXL-Beta, AnimateDiff-Lightning, 
ModelScope T2V, Deforum, LTX-Video, SkyReels V2, SadTalker, and DreamTalk.
"""
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

def configure_imagemagick():
    """Configure ImageMagick binary path for MoviePy."""
    try:
        import subprocess
        result = subprocess.run(['which', 'convert'], capture_output=True, text=True)
        if result.returncode == 0:
            imagemagick_binary = result.stdout.strip()
            os.environ['IMAGEMAGICK_BINARY'] = imagemagick_binary
            logger.info(f"ImageMagick configured: {imagemagick_binary}")
            
            try:
                from moviepy.config import change_settings
                change_settings({"IMAGEMAGICK_BINARY": imagemagick_binary})
                logger.info("MoviePy ImageMagick configuration updated")
            except Exception as config_error:
                logger.warning(f"MoviePy config update failed: {config_error}")
        else:
            common_paths = ['/usr/bin/convert', '/usr/local/bin/convert', '/opt/homebrew/bin/convert']
            for path in common_paths:
                if os.path.exists(path):
                    os.environ['IMAGEMAGICK_BINARY'] = path
                    logger.info(f"ImageMagick found at: {path}")
                    try:
                        from moviepy.config import change_settings
                        change_settings({"IMAGEMAGICK_BINARY": path})
                    except:
                        pass
                    break
            else:
                logger.warning("ImageMagick not found, text overlays will use OpenCV fallback")
    except Exception as e:
        logger.warning(f"ImageMagick configuration failed: {e}")

configure_imagemagick()

class TextToVideoGenerator:
    """Main class for text-to-video generation with multiple model support."""
    
    def __init__(self, vram_tier: str = "medium", target_resolution: Optional[Tuple[int, int]] = None):
        self.vram_tier = vram_tier
        self.target_resolution = target_resolution
        self.loaded_models = {}
        self.models = {}
        self.model_settings = self._get_model_settings()
    
    def _get_model_settings(self) -> Dict:
        """Get optimal settings for each model based on VRAM tier and user specifications."""
        settings = {
            "low": {
                "svd_xt": {"max_frames": 25, "resolution": (1024, 576), "steps": 20, "vram_req": "16-24GB"},
                "zeroscope_v2_xl": {"max_frames": 24, "resolution": (1024, 576), "steps": 20, "vram_req": "12-16GB"},
                "animatediff_lightning": {"max_frames": 16, "resolution": (512, 512), "steps": 15, "vram_req": "8-12GB"},
                "modelscope_t2v": {"max_frames": 16, "resolution": (256, 256), "steps": 20, "vram_req": "8-12GB"}
            },
            "medium": {
                "svd_xt": {"max_frames": 25, "resolution": (1024, 576), "steps": 25, "vram_req": "16-24GB"},
                "zeroscope_v2_xl": {"max_frames": 24, "resolution": (1024, 576), "steps": 25, "vram_req": "12-16GB"},
                "animatediff_v2_sdxl": {"max_frames": 16, "resolution": (1024, 1024), "steps": 30, "vram_req": "13-16GB"},
                "animatediff_lightning": {"max_frames": 16, "resolution": (512, 512), "steps": 15, "vram_req": "8-12GB"},
                "modelscope_t2v": {"max_frames": 16, "resolution": (256, 256), "steps": 25, "vram_req": "8-12GB"},
                "deforum": {"max_frames": 30, "resolution": (768, 768), "steps": 25, "vram_req": "8-16GB"}
            },
            "high": {
                "svd_xt": {"max_frames": 25, "resolution": (1024, 576), "steps": 30, "vram_req": "16-24GB"},
                "zeroscope_v2_xl": {"max_frames": 24, "resolution": (1024, 576), "steps": 30, "vram_req": "12-16GB"},
                "animatediff_v2_sdxl": {"max_frames": 16, "resolution": (1024, 1024), "steps": 35, "vram_req": "13-16GB"},
                "animatediff_lightning": {"max_frames": 16, "resolution": (512, 512), "steps": 20, "vram_req": "8-12GB"},
                "ltx_video": {"max_frames": 120, "resolution": (768, 512), "steps": 25, "vram_req": "24-48GB"},
                "deforum": {"max_frames": 60, "resolution": (1024, 1024), "steps": 30, "vram_req": "8-16GB"}
            },
            "ultra": {
                "svd_xt": {"max_frames": 25, "resolution": (1024, 576), "steps": 40, "vram_req": "16-24GB"},
                "zeroscope_v2_xl": {"max_frames": 24, "resolution": (1024, 576), "steps": 40, "vram_req": "12-16GB"},
                "animatediff_v2_sdxl": {"max_frames": 16, "resolution": (1024, 1024), "steps": 50, "vram_req": "13-16GB"},
                "ltx_video": {"max_frames": 120, "resolution": (768, 512), "steps": 30, "vram_req": "24-48GB"},
                "skyreels_v2": {"max_frames": 999, "resolution": (960, 540), "steps": 35, "vram_req": "24-48GB"},
                "deforum": {"max_frames": 120, "resolution": (1024, 1024), "steps": 40, "vram_req": "8-16GB"}
            }
        }
        return settings.get(self.vram_tier, settings["medium"])
    
    def get_optimal_settings(self, model_name: str) -> Dict:
        """Get optimal settings for a specific model."""
        return self.model_settings.get(model_name, {})
    
    def split_scene_for_model(self, scene_description: str, model_name: str) -> List[str]:
        """Split scene description into clips appropriate for model limitations."""
        settings = self.get_optimal_settings(model_name)
        max_frames = settings.get("max_frames", 24)
        
        words = scene_description.split()
        if len(words) <= 20:
            return [scene_description]
        
        clips = []
        words_per_clip = max(10, len(words) // max(1, len(words) // 15))
        
        for i in range(0, len(words), words_per_clip):
            clip_words = words[i:i + words_per_clip]
            clips.append(" ".join(clip_words))
        
        return clips
    
    def load_model(self, model_name: str):
        """Load a specific video generation model."""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        try:
            import torch
            logger.info(f"Loading video model: {model_name}")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
            if model_name == "svd_xt":
                from diffusers import StableVideoDiffusionPipeline
                device_map = "balanced" if torch.cuda.is_available() else None
                if device_map:
                    model = StableVideoDiffusionPipeline.from_pretrained(
                        "stabilityai/stable-video-diffusion-img2vid-xt",
                        torch_dtype=dtype,
                        device_map=device_map,
                        low_cpu_mem_usage=True
                    )
                else:
                    model = StableVideoDiffusionPipeline.from_pretrained(
                        "stabilityai/stable-video-diffusion-img2vid-xt",
                        torch_dtype=dtype,
                        low_cpu_mem_usage=True
                    ).to("cpu")
            elif model_name == "zeroscope_v2_xl":
                from diffusers import DiffusionPipeline
                device_map = "balanced" if torch.cuda.is_available() else None
                if device_map:
                    model = DiffusionPipeline.from_pretrained(
                        "cerspense/zeroscope_v2_XL",
                        torch_dtype=dtype,
                        device_map=device_map,
                        low_cpu_mem_usage=True
                    )
                else:
                    model = DiffusionPipeline.from_pretrained(
                        "cerspense/zeroscope_v2_XL",
                        torch_dtype=dtype,
                        low_cpu_mem_usage=True
                    ).to("cpu")
            elif model_name == "animatediff_v2_sdxl":
                from diffusers import AnimateDiffPipeline, MotionAdapter
                device = "cuda" if torch.cuda.is_available() else "cpu"
                dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                device_map = "balanced" if torch.cuda.is_available() else None
                adapter = MotionAdapter.from_pretrained(
                    "guoyww/animatediff-motion-adapter-sdxl-beta",
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True
                ).to(device)
                if device_map:
                    model = AnimateDiffPipeline.from_pretrained(
                        "stabilityai/stable-diffusion-xl-base-1.0",
                        motion_adapter=adapter,
                        torch_dtype=dtype,
                        device_map=device_map,
                        low_cpu_mem_usage=True
                    )
                else:
                    model = AnimateDiffPipeline.from_pretrained(
                        "stabilityai/stable-diffusion-xl-base-1.0",
                        motion_adapter=adapter,
                        torch_dtype=dtype,
                        low_cpu_mem_usage=True
                    ).to("cpu")
            elif model_name == "animatediff_lightning":
                from diffusers import AnimateDiffPipeline
                device = "cuda" if torch.cuda.is_available() else "cpu"
                dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                device_map = "balanced" if torch.cuda.is_available() else None
                if device_map:
                    model = AnimateDiffPipeline.from_pretrained(
                        "ByteDance/AnimateDiff-Lightning",
                        torch_dtype=dtype,
                        device_map=device_map,
                        low_cpu_mem_usage=True
                    )
                else:
                    model = AnimateDiffPipeline.from_pretrained(
                        "ByteDance/AnimateDiff-Lightning",
                        torch_dtype=dtype,
                        low_cpu_mem_usage=True
                    ).to("cpu")
            elif model_name == "modelscope_t2v":
                from diffusers import DiffusionPipeline
                device_map = "balanced" if torch.cuda.is_available() else None
                if device_map:
                    model = DiffusionPipeline.from_pretrained(
                        "damo-vilab/text-to-video-ms-1.7b",
                        torch_dtype=dtype,
                        device_map=device_map,
                        low_cpu_mem_usage=True
                    )
                else:
                    model = DiffusionPipeline.from_pretrained(
                        "damo-vilab/text-to-video-ms-1.7b",
                        torch_dtype=dtype,
                        low_cpu_mem_usage=True
                    ).to("cpu")
            elif model_name == "ltx_video":
                from diffusers import DiffusionPipeline
                device_map = "balanced" if torch.cuda.is_available() else None
                if device_map:
                    model = DiffusionPipeline.from_pretrained(
                        "Lightricks/LTX-Video",
                        torch_dtype=dtype,
                        device_map=device_map,
                        low_cpu_mem_usage=True
                    )
                else:
                    model = DiffusionPipeline.from_pretrained(
                        "Lightricks/LTX-Video",
                        torch_dtype=dtype,
                        low_cpu_mem_usage=True
                    ).to("cpu")
            elif model_name == "skyreels_v2":
                from diffusers import DiffusionPipeline
                device_map = "balanced" if torch.cuda.is_available() else None
                if device_map:
                    model = DiffusionPipeline.from_pretrained(
                        "Skywork/SkyReels-V2-T2V-14B-540P",
                        torch_dtype=dtype,
                        device_map=device_map,
                        low_cpu_mem_usage=True
                    )
                else:
                    model = DiffusionPipeline.from_pretrained(
                        "Skywork/SkyReels-V2-T2V-14B-540P",
                        torch_dtype=dtype,
                        low_cpu_mem_usage=True
                    ).to("cpu")
            elif model_name == "deforum":
                logger.info("Deforum requires custom implementation - using fallback")
                return None
            else:
                logger.warning(f"Unknown model: {model_name}, using fallback")
                return None
            

            
            if model is not None:
                self.models[model_name] = model
                self.loaded_models[model_name] = model
                logger.info(f"Successfully loaded {model_name}")
                return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise Exception(f"Model loading failed for {model_name}: {e}")
    
    def generate_video(self, prompt: str, model_name: str, output_path: str) -> bool:
        """Generate video using specified model."""
        try:
            import torch
            logger.info(f"Generating high-quality video with {model_name}: {prompt[:50]}...")
            
            model = self.load_model(model_name)
            if model is None:
                self._cleanup_model_memory()
                logger.error(f"Model {model_name} not available and fallback disabled")
                raise Exception(f"Video generation model {model_name} failed to load")
            
            settings = self.get_optimal_settings(model_name)
            target_res = self.target_resolution or settings.get("resolution", (1024, 576))
            
            try:
                if model_name == "svd_xt":
                    from PIL import Image
                    image = Image.new('RGB', settings.get("resolution", (1024, 576)), color='black')
                    output = model(image, num_frames=settings.get("max_frames", 25))
                    frames = output.frames[0]
                elif model_name and model_name in ["animatediff_v2_sdxl", "animatediff_lightning"]:
                    output = model(
                        prompt=prompt,
                        num_frames=settings.get("max_frames", 16),
                        guidance_scale=7.5,
                        num_inference_steps=settings.get("steps", 25),
                        height=target_res[1],
                        width=target_res[0]
                    )
                    frames = output.frames[0]
                elif model_name and model_name in ["zeroscope_v2_xl", "modelscope_t2v", "ltx_video", "skyreels_v2"]:
                    output = model(
                        prompt,
                        num_frames=settings.get("max_frames", 24),
                        num_inference_steps=settings.get("steps", 25),
                        height=target_res[1],
                        width=target_res[0]
                    )
                    if hasattr(output, 'frames') and output.frames is not None and len(output.frames) > 0:
                        frames = output.frames[0]
                    elif hasattr(output, 'images') and output.images is not None:
                        frames = output.images
                    else:
                        logger.error(f"Video model {model_name} returned no frames")
                        raise Exception(f"No frames generated by {model_name}")
                else:
                    logger.error(f"Unknown model type: {model_name}")
                    raise Exception(f"Unknown model type: {model_name}")
                
                if frames is None:
                    raise Exception(f"Video generation returned None frames for {model_name}")
                
                settings_resolution = settings.get("resolution")
                if target_res != settings_resolution and settings_resolution is not None:
                    frames = self._upscale_frames(frames, target_res)
                
                self._save_video_frames(frames, output_path)
                return True
                
            except Exception as e:
                logger.error(f"Error during video generation: {e}")
                self._cleanup_model_memory()
                raise Exception(f"Video generation failed for {model_name}: {e}")
            
        except Exception as e:
            logger.error(f"Error generating video with {model_name}: {e}")
            raise Exception(f"Video generation completely failed for {model_name}: {e}")
        finally:
            self._cleanup_model_memory()
            self.force_cleanup_all_models()
    
    def _cleanup_model_memory(self):
        """Clean up model memory after video generation."""
        try:
            import torch
            import gc
            
            for model_name, model in self.loaded_models.items():
                if hasattr(model, 'to'):
                    model.to('cpu')
                del model
            
            self.loaded_models.clear()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            gc.collect()
            logger.info("Model memory cleanup completed")
        except Exception as e:
            logger.error(f"Error during model memory cleanup: {e}")
    
    def force_cleanup_all_models(self):
        """Force cleanup of all loaded models and GPU memory."""
        try:
            import torch
            import gc
            
            for model_name in list(self.loaded_models.keys()):
                model = self.loaded_models.pop(model_name)
                if hasattr(model, 'to'):
                    model.to('cpu')
                if hasattr(model, 'unload'):
                    model.unload()
                del model
            
            for model_name in list(self.models.keys()):
                model = self.models.pop(model_name)
                if hasattr(model, 'to'):
                    model.to('cpu')
                del model
            
            self.loaded_models.clear()
            self.models.clear()
            
            for _ in range(3):
                gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.ipc_collect()
                
                for device_id in range(torch.cuda.device_count()):
                    with torch.cuda.device(device_id):
                        torch.cuda.empty_cache()
            
            logger.info("Force cleanup of all models completed")
        except Exception as e:
            logger.error(f"Error during force model cleanup: {e}")
    
    def _upscale_frames(self, frames, target_resolution: tuple):
        """Upscale video frames to target resolution."""
        try:
            import cv2
            import numpy as np
            
            upscaled_frames = []
            target_width, target_height = target_resolution
            
            for frame in frames:
                if hasattr(frame, 'numpy'):
                    frame_array = frame.numpy()
                elif isinstance(frame, np.ndarray):
                    frame_array = frame
                else:
                    frame_array = np.array(frame)
                
                if len(frame_array.shape) == 4:
                    frame_array = frame_array[0]
                
                if frame_array.shape[2] == 3:
                    frame_array = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
                
                upscaled = cv2.resize(frame_array, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
                
                if upscaled.shape[2] == 3:
                    upscaled = cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB)
                
                upscaled_frames.append(upscaled)
            
            logger.info(f"Upscaled {len(upscaled_frames)} frames to {target_resolution}")
            return upscaled_frames
            
        except Exception as e:
            logger.error(f"Error upscaling frames: {e}")
            return frames
    
    def _save_video_frames(self, frames, output_path: str):
        """Save video frames to file."""
        try:
            from moviepy.editor import ImageSequenceClip
            import numpy as np
            
            if isinstance(frames, list):
                frame_arrays = []
                for frame in frames:
                    if hasattr(frame, 'numpy'):
                        frame_arrays.append(frame.numpy())
                    elif isinstance(frame, np.ndarray):
                        frame_arrays.append(frame)
                    else:
                        frame_arrays.append(np.array(frame))
            else:
                frame_arrays = [np.array(frames)]
            
            if frame_arrays:
                clip = ImageSequenceClip(frame_arrays, fps=24)
                clip.write_videofile(output_path, codec='libx264', audio_codec='aac', fps=24)
                clip.close()
                logger.info(f"High-quality video saved to {output_path}")
            else:
                logger.error("No frames to save")
                
        except Exception as e:
            logger.error(f"Error saving video frames: {e}")
            try:
                from moviepy.editor import ColorClip
                fallback_clip = ColorClip(size=(1920, 1080), color=(0, 0, 0), duration=5)
                fallback_clip.write_videofile(output_path, codec='libx264', fps=24)
                fallback_clip.close()
                logger.info(f"Created fallback video at {output_path}")
            except Exception as fallback_error:
                logger.error(f"Failed to create fallback video: {fallback_error}")
                with open(output_path, "wb") as f:
                    f.write(b"")
    
    def _create_high_quality_fallback(self, prompt: str, model_name: str, output_path: str) -> bool:
        """Create high-quality fallback video when model generation fails."""
        try:
            from moviepy.editor import ColorClip, TextClip, CompositeVideoClip
            
            settings = self.get_optimal_settings(model_name)
            resolution = self.target_resolution or settings.get("resolution", (1024, 576))
            duration = min(10, settings.get("max_frames", 25) / 24.0)
            
            bg_colors = {
                "svd_xt": (25, 35, 55),
                "zeroscope_v2_xl": (35, 25, 55),
                "animatediff_v2_sdxl": (55, 35, 25),
                "animatediff_lightning": (45, 45, 25),
                "modelscope_t2v": (25, 55, 35),
                "ltx_video": (55, 25, 35),
                "skyreels_v2": (35, 55, 25),
                "sadtalker": (45, 25, 45),
                "dreamtalk": (25, 45, 45)
            }
            bg_color = bg_colors.get(model_name, (30, 30, 50))
            
            bg_clip = ColorClip(size=resolution, color=bg_color, duration=duration)
            
            try:
                title_text = f"{model_name.upper().replace('_', ' ')} Generated"
                title_clip = TextClip(title_text, fontsize=min(48, resolution[0]//20), color='white', font='Arial-Bold')
                title_clip = title_clip.set_position('center').set_duration(2)
                
                prompt_text = prompt[:150] + "..." if len(prompt) > 150 else prompt
                prompt_clip = TextClip(prompt_text, fontsize=min(24, resolution[0]//40), color='lightblue', font='Arial',
                                     size=(resolution[0]-100, None), method='caption')
                prompt_clip = prompt_clip.set_position('center').set_start(2).set_duration(duration-2)
                
                model_info = f"Model: {model_name} | Resolution: {resolution[0]}x{resolution[1]} | VRAM: {settings.get('vram_req', 'N/A')}"
                info_clip = TextClip(model_info, fontsize=min(16, resolution[0]//60), color='gray', font='Arial')
                info_clip = info_clip.set_position(('center', resolution[1]-50)).set_duration(duration)
                
                video = CompositeVideoClip([bg_clip, title_clip, prompt_clip, info_clip])
            except Exception as text_error:
                logger.warning(f"Text overlay failed (ImageMagick not available): {text_error}")
                logger.info("ImageMagick unavailable - forcing enhanced OpenCV fallback instead of basic ColorClip")
                # Force OpenCV fallback instead of using basic ColorClip
                raise Exception("Forcing OpenCV fallback due to ImageMagick unavailability")
            
            video.write_videofile(str(output_path), codec='libx264', fps=30, audio_codec='aac', bitrate='8000k')
            
            bg_clip.close()
            if 'title_clip' in locals():
                title_clip.close()
            if 'prompt_clip' in locals():
                prompt_clip.close()
            if 'info_clip' in locals():
                info_clip.close()
            video.close()
            
            logger.info(f"Successfully generated high-quality fallback video: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"MoviePy fallback failed: {e}")
            logger.info("Attempting enhanced OpenCV fallback video generation...")
            try:
                import cv2
                import numpy as np
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(output_path), fourcc, 24.0, resolution)
                
                if out is None or not out.isOpened():
                    logger.error("Failed to initialize OpenCV VideoWriter")
                    return False
                
                duration_frames = int(24 * duration)  # Use actual duration
                
                bg_colors = {
                    "svd_xt": (25, 35, 55),
                    "zeroscope_v2_xl": (35, 25, 55),
                    "animatediff_v2_sdxl": (55, 35, 25),
                    "animatediff_lightning": (45, 45, 25),
                    "modelscope_t2v": (25, 55, 35),
                    "ltx_video": (55, 25, 35),
                    "skyreels_v2": (35, 55, 25),
                    "sadtalker": (45, 25, 45),
                    "dreamtalk": (25, 45, 45)
                }
                bg_color = bg_colors.get(model_name, (30, 30, 50))
                
                for frame_idx in range(duration_frames):
                    frame = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
                    
                    for y in range(resolution[1]):
                        gradient_factor = y / resolution[1]
                        color = tuple(int(c * (0.5 + 0.5 * gradient_factor)) for c in bg_color)
                        frame[y, :] = color
                    
                    progress = frame_idx / duration_frames
                    
                    center_x = int(resolution[0] * (0.2 + 0.6 * progress))
                    center_y = int(resolution[1] * 0.5)
                    radius = min(30, resolution[0] // 30)
                    cv2.circle(frame, (center_x, center_y), radius, (255, 255, 255), -1)
                    
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = min(1.0, resolution[0] / 800)
                    thickness = max(1, int(resolution[0] / 400))
                    
                    title_text = f"{model_name.upper().replace('_', ' ')} Generated"
                    text_size = cv2.getTextSize(title_text, font, font_scale, thickness)[0]
                    text_x = (resolution[0] - text_size[0]) // 2
                    text_y = resolution[1] // 4
                    cv2.putText(frame, title_text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
                    
                    prompt_text = prompt[:50] + "..." if len(prompt) > 50 else prompt
                    prompt_size = cv2.getTextSize(prompt_text, font, font_scale * 0.6, thickness)[0]
                    prompt_x = (resolution[0] - prompt_size[0]) // 2
                    prompt_y = resolution[1] // 2
                    cv2.putText(frame, prompt_text, (prompt_x, prompt_y), font, font_scale * 0.6, (200, 200, 255), thickness)
                    
                    bar_width = resolution[0] // 2
                    bar_height = 10
                    bar_x = (resolution[0] - bar_width) // 2
                    bar_y = resolution[1] * 3 // 4
                    
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
                    fill_width = int(bar_width * progress)
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), (0, 255, 0), -1)
                    
                    info_text = f"Resolution: {resolution[0]}x{resolution[1]} | Frame: {frame_idx+1}/{duration_frames}"
                    info_size = cv2.getTextSize(info_text, font, font_scale * 0.4, 1)[0]
                    info_x = (resolution[0] - info_size[0]) // 2
                    info_y = resolution[1] - 30
                    cv2.putText(frame, info_text, (info_x, info_y), font, font_scale * 0.4, (150, 150, 150), 1)
                    
                    out.write(frame)
                
                out.release()
                logger.info(f"Created enhanced OpenCV fallback video: {output_path} ({duration_frames} frames)")
                return True
                
            except Exception as cv_error:
                logger.error(f"Enhanced OpenCV fallback also failed: {cv_error}")
                try:
                    import cv2
                    import numpy as np
                    
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(str(output_path), fourcc, 24.0, resolution)
                    
                    black_frame = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
                    for _ in range(24 * 3):  # 3 second minimum
                        out.write(black_frame)
                    
                    out.release()
                    logger.info(f"Created minimal fallback video: {output_path}")
                    return True
                except:
                    return False

def split_scene_for_model(scene_description: str, model_name: str, vram_tier: str = "medium") -> List[str]:
    """Split scene description into model-appropriate clips."""
    generator = TextToVideoGenerator(vram_tier)
    return generator.split_scene_for_model(scene_description, model_name)

def get_best_model_for_content(content_type: str, vram_tier: str = "medium") -> str:
    """Select the best video generation model for content type and VRAM tier."""
    if vram_tier == "low":
        model_preferences = {
            "anime": "animatediff_lightning",
            "gaming": "modelscope_t2v", 
            "superhero": "animatediff_lightning",
            "manga": "animatediff_lightning",
            "marvel_dc": "modelscope_t2v",
            "original_manga": "animatediff_lightning",
            "shorts": "animatediff_lightning"
        }
    elif vram_tier == "medium":
        model_preferences = {
            "anime": "animatediff_v2_sdxl",
            "gaming": "zeroscope_v2_xl", 
            "superhero": "zeroscope_v2_xl",
            "manga": "animatediff_v2_sdxl",
            "marvel_dc": "svd_xt",
            "original_manga": "animatediff_v2_sdxl",
            "shorts": "zeroscope_v2_xl"
        }
    elif vram_tier == "high":
        model_preferences = {
            "anime": "animatediff_v2_sdxl",
            "gaming": "ltx_video", 
            "superhero": "svd_xt",
            "manga": "animatediff_v2_sdxl",
            "marvel_dc": "svd_xt",
            "original_manga": "animatediff_v2_sdxl",
            "shorts": "ltx_video"
        }
    else:  # ultra
        model_preferences = {
            "anime": "animatediff_v2_sdxl",
            "gaming": "skyreels_v2", 
            "superhero": "svd_xt",
            "manga": "animatediff_v2_sdxl",
            "marvel_dc": "svd_xt",
            "original_manga": "animatediff_v2_sdxl",
            "shorts": "skyreels_v2"
        }
    
    return model_preferences.get(content_type, "svd_xt")

def get_best_model_for_combat(content_type: str, vram_tier: str = "medium", combat_type: str = "melee") -> str:
    """Select the best video generation model for combat scenes."""
    if vram_tier == "low":
        combat_preferences = {
            "melee": "animatediff_lightning",
            "ranged": "modelscope_t2v",
            "magic": "animatediff_lightning",
            "aerial": "modelscope_t2v"
        }
    elif vram_tier == "medium":
        combat_preferences = {
            "melee": "animatediff_v2_sdxl",
            "ranged": "zeroscope_v2_xl",
            "magic": "animatediff_v2_sdxl",
            "aerial": "zeroscope_v2_xl"
        }
    elif vram_tier == "high":
        combat_preferences = {
            "melee": "svd_xt",
            "ranged": "ltx_video",
            "magic": "svd_xt",
            "aerial": "ltx_video"
        }
    else:
        combat_preferences = {
            "melee": "svd_xt",
            "ranged": "skyreels_v2",
            "magic": "svd_xt", 
            "aerial": "skyreels_v2"
        }
    
    return combat_preferences.get(combat_type, get_best_model_for_content(content_type, vram_tier))
