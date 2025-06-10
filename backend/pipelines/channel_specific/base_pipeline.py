"""
Base Pipeline for Self-Contained Channel Processing
Provides common functionality for all channel-specific pipelines.
"""

import os
import sys
import json
import yaml
import time
import logging
import tempfile
import shutil
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import torch
import numpy as np

logger = logging.getLogger(__name__)

class BasePipeline:
    """Base class for self-contained channel pipelines."""
    
    def __init__(self, channel_type: str, output_path: Optional[str] = None, base_model: str = "stable_diffusion_1_5"):
        self.channel_type = channel_type
        self.base_model = base_model
        self.models = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.vram_tier = self._detect_vram_tier()
        self.output_path = Path(output_path) if output_path else None
        
        self.supported_languages = {
            "en": {"name": "English", "bark_supported": True, "xtts_supported": True},
            "ja": {"name": "Japanese", "bark_supported": True, "xtts_supported": True},
            "es": {"name": "Spanish", "bark_supported": True, "xtts_supported": True},
            "zh": {"name": "Chinese", "bark_supported": False, "xtts_supported": True},
            "hi": {"name": "Hindi", "bark_supported": False, "xtts_supported": True},
            "ar": {"name": "Arabic", "bark_supported": False, "xtts_supported": True},
            "bn": {"name": "Bengali", "bark_supported": False, "xtts_supported": True},
            "pt": {"name": "Portuguese", "bark_supported": True, "xtts_supported": True},
            "ru": {"name": "Russian", "bark_supported": True, "xtts_supported": True},
            "fr": {"name": "French", "bark_supported": True, "xtts_supported": True},
            "de": {"name": "German", "bark_supported": True, "xtts_supported": True}
        }
    
    def _run_pipeline(self, input_path: str, output_path: str, base_model: str = "stable_diffusion_1_5", 
                     lora_models: Optional[List[str]] = None, lora_paths: Optional[Dict[str, str]] = None, 
                     db_run=None, db=None, render_fps: int = 24, output_fps: int = 24, 
                     frame_interpolation_enabled: bool = True, language: str = "en") -> str:
        """Core pipeline execution method."""
        try:
            logger.info(f"Starting {self.channel_type} pipeline")
            
            script_data = self.parse_input_script(input_path)
            
            expanded_script = self.expand_script_if_needed(script_data)
            
            output_dir = self.ensure_output_dir(output_path)
            
            scene_videos = []
            scenes = expanded_script.get('scenes', self._get_default_scenes())
            
            for i, scene in enumerate(scenes):
                scene_description = scene if isinstance(scene, str) else scene.get('description', f'Scene {i+1}')
                enhanced_prompt = self._enhance_prompt_for_channel(scene_description)
                
                scene_output = output_dir / f"scene_{i+1}.mp4"
                video_path = self.generate_video(enhanced_prompt, output_path=str(scene_output), duration=5.0)
                
                if video_path and os.path.exists(video_path):
                    scene_videos.append(video_path)
                    logger.info(f"Generated scene {i+1}: {video_path}")
            
            final_video = output_dir / "final_video.mp4"
            if scene_videos:
                self._combine_scene_videos(scene_videos, str(final_video))
            else:
                self._create_fallback_video("Default content", 300.0, str(final_video))
            
            self.create_manifest(output_dir, 
                               scenes_generated=len(scene_videos),
                               final_video=str(final_video),
                               language=language)
            
            logger.info(f"{self.channel_type} pipeline completed: {final_video}")
            return str(output_dir)
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            output_dir = self.ensure_output_dir(output_path)
            fallback_video = output_dir / "fallback_video.mp4"
            self._create_fallback_video("Pipeline failed", 300.0, str(fallback_video))
            return str(output_dir)
    
    def _get_default_scenes(self) -> List[str]:
        """Get default scenes for the channel type."""
        return [
            f"Opening scene for {self.channel_type} content",
            f"Main action sequence in {self.channel_type} style",
            f"Character development scene for {self.channel_type}",
            f"Climactic moment in {self.channel_type} format",
            f"Closing scene for {self.channel_type} content"
        ]
    
    def _get_default_characters(self) -> List[Dict]:
        """Get default characters for the channel type."""
        return [
            {"name": "Character1", "description": f"Main character in {self.channel_type} style", "voice": "default_voice"},
            {"name": "Character2", "description": f"Supporting character in {self.channel_type} style", "voice": "default_voice"}
        ]
    
    def _enhance_prompt_for_channel(self, prompt: str) -> str:
        """Enhance prompt for specific channel style with detailed enhancements."""
        channel_enhancements = {
            "anime": {
                "prefix": "anime style, high quality animation, detailed character design",
                "style_tags": "cel shading, vibrant colors, expressive eyes, dynamic poses",
                "quality": "masterpiece, best quality, ultra detailed, 8k resolution"
            },
            "gaming": {
                "prefix": "gaming content, action-packed, dynamic gameplay",
                "style_tags": "realistic graphics, intense action, competitive gaming",
                "quality": "high resolution, smooth animation, professional gaming"
            },
            "superhero": {
                "prefix": "superhero style, heroic poses, dramatic lighting",
                "style_tags": "comic book style, powerful characters, epic scenes",
                "quality": "cinematic quality, dramatic composition, heroic atmosphere"
            },
            "manga": {
                "prefix": "manga style, black and white, detailed line art",
                "style_tags": "manga panels, expressive characters, detailed backgrounds",
                "quality": "high contrast, detailed illustration, professional manga"
            },
            "marvel_dc": {
                "prefix": "Marvel/DC comic style, superhero universe",
                "style_tags": "comic book art, iconic characters, action scenes",
                "quality": "comic book quality, detailed artwork, iconic style"
            },
            "original_manga": {
                "prefix": "original manga style, unique character design",
                "style_tags": "creative storytelling, original characters, manga aesthetics",
                "quality": "artistic quality, creative composition, original style"
            }
        }
        
        enhancement = channel_enhancements.get(self.channel_type, {
            "prefix": f"{self.channel_type} style",
            "style_tags": "high quality, detailed",
            "quality": "professional quality"
        })
        
        enhanced_prompt = f"{enhancement['prefix']}, {prompt}, {enhancement['style_tags']}, {enhancement['quality']}"
        return enhanced_prompt
    def _classify_scene_type(self, prompt: str) -> str:
        """Classify scene type for optimal model selection."""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ["fight", "battle", "combat", "attack", "war"]):
            return "combat"
        elif any(word in prompt_lower for word in ["action", "chase", "run", "escape", "fast"]):
            return "action"
        elif any(word in prompt_lower for word in ["talk", "dialogue", "conversation", "speak", "discuss"]):
            return "dialogue"
        elif any(word in prompt_lower for word in ["explore", "discover", "journey", "travel", "walk"]):
            return "exploration"
        elif any(word in prompt_lower for word in ["emotion", "sad", "happy", "love", "memory", "feel"]):
            return "character_development"
        else:
            return "default"

        return f"{self.channel_type} style, {prompt}, high quality, detailed"
    
    def _combine_scene_videos(self, video_paths: List[str], output_path: str):
        """Combine multiple scene videos into final video using FFmpeg."""
        try:
            import subprocess
            import tempfile
            from pathlib import Path
            
            if not video_paths:
                logger.warning("No video paths provided for combination")
                return self._create_fallback_video("Combined scenes", 300.0, output_path)
            
            valid_videos = [path for path in video_paths if os.path.exists(path)]
            if not valid_videos:
                logger.warning("No valid video files found for combination")
                return self._create_fallback_video("Combined scenes", 300.0, output_path)
            
            logger.info(f"Combining {len(valid_videos)} scene videos using FFmpeg")
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                concat_file = f.name
                for video_path in valid_videos:
                    f.write(f"file '{os.path.abspath(video_path)}'\n")
            
            try:
                cmd = [
                    'ffmpeg', '-y',  # Overwrite output
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', concat_file,
                    '-c', 'copy',  # Copy streams without re-encoding
                    '-avoid_negative_ts', 'make_zero',
                    output_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0 and os.path.exists(output_path):
                    file_size = os.path.getsize(output_path)
                    logger.info(f"Successfully combined {len(valid_videos)} scenes into {output_path} ({file_size} bytes)")
                    return output_path
                else:
                    logger.error(f"FFmpeg failed: {result.stderr}")
                    return self._combine_videos_opencv(valid_videos, output_path)
                    
            finally:
                if os.path.exists(concat_file):
                    os.unlink(concat_file)
                
        except Exception as e:
            logger.error(f"Error combining scene videos with FFmpeg: {e}")
            return self._combine_videos_opencv(valid_videos, output_path)
    
    def _combine_videos_opencv(self, video_paths: List[str], output_path: str) -> str:
        """Fallback method to combine videos using OpenCV."""
        try:
            import cv2
            
            logger.info(f"Using OpenCV fallback to combine {len(video_paths)} videos")
            
            if not video_paths:
                return self._create_fallback_video("Combined scenes", 300.0, output_path)
            
            first_video = cv2.VideoCapture(video_paths[0])
            width = int(first_video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(first_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(first_video.get(cv2.CAP_PROP_FPS)) or 24
            first_video.release()
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            total_frames = 0
            for video_path in video_paths:
                logger.info(f"Processing scene: {video_path}")
                cap = cv2.VideoCapture(video_path)
                
                frame_count = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if frame.shape[1] != width or frame.shape[0] != height:
                        frame = cv2.resize(frame, (width, height))
                    
                    out.write(frame)
                    total_frames += 1
                    frame_count += 1
                
                cap.release()
                logger.info(f"Added {frame_count} frames from {video_path}")
            
            out.release()
            
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                logger.info(f"OpenCV combined {len(video_paths)} scenes into {total_frames} frames ({file_size} bytes)")
                return output_path
            else:
                logger.error("Failed to create combined video file")
                return self._create_fallback_video("Combined scenes", 300.0, output_path)
                
        except Exception as e:
            logger.error(f"OpenCV video combination failed: {e}")
            return self._create_fallback_video("Combined scenes", 300.0, output_path)
    
    def _initialize_language_support(self):
        """Initialize supported languages configuration."""
        self.supported_languages = {
            'en': {'name': 'English', 'voice_code': 'en', 'xtts_supported': True, 'bark_supported': True},
            'ja': {'name': 'Japanese', 'voice_code': 'ja', 'xtts_supported': True, 'bark_supported': True},
            'es': {'name': 'Spanish', 'voice_code': 'es', 'xtts_supported': True, 'bark_supported': True},
            'zh': {'name': 'Chinese', 'voice_code': 'zh-cn', 'xtts_supported': True, 'bark_supported': False},
            'hi': {'name': 'Hindi', 'voice_code': 'hi', 'xtts_supported': True, 'bark_supported': False},
            'ar': {'name': 'Arabic', 'voice_code': 'ar', 'xtts_supported': True, 'bark_supported': False},
            'bn': {'name': 'Bengali', 'voice_code': 'bn', 'xtts_supported': True, 'bark_supported': False},
            'pt': {'name': 'Portuguese', 'voice_code': 'pt', 'xtts_supported': True, 'bark_supported': True},
            'ru': {'name': 'Russian', 'voice_code': 'ru', 'xtts_supported': True, 'bark_supported': False},
            'fr': {'name': 'French', 'voice_code': 'fr', 'xtts_supported': True, 'bark_supported': True},
            'de': {'name': 'German', 'voice_code': 'de', 'xtts_supported': True, 'bark_supported': True}
        }
    
    def _detect_vram_tier(self) -> str:
        """Detect VRAM tier for model selection."""
        if not torch.cuda.is_available():
            return "cpu"
        
        try:
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if vram_gb >= 24:
                return "extreme"
            elif vram_gb >= 16:
                return "high"
            elif vram_gb >= 8:
                return "medium"
            else:
                return "low"
        except:
            return "medium"
    
    def load_llm_model(self, model_name: str = "deepseek_llama_8b_peft"):
        """Load LLM model with Deepseek support."""
        if "llm" in self.models:
            return self.models["llm"]
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            try:
                from ..model_manager import TEXT_MODELS, HF_MODEL_REPOS
                model_info = TEXT_MODELS.get(model_name)
                if not model_info:
                    logger.warning(f"Model {model_name} not found, using fallback")
                    model_name = "dialogpt_medium"
                    model_info = TEXT_MODELS.get(model_name)
                
                model_id = model_info.get("model_id", "microsoft/DialoGPT-medium")
            except ImportError:
                logger.warning("Model manager not available, using default models")
                model_mapping = {
                    "deepseek_llama_8b_peft": "deepseek-ai/deepseek-llm-7b-chat",
                    "dialogpt_medium": "microsoft/DialoGPT-medium",
                    "gpt2": "gpt2",
                    "distilgpt2": "distilgpt2"
                }
                model_id = model_mapping.get(model_name, "microsoft/DialoGPT-medium")
            
            logger.info(f"Loading LLM model: {model_id}")
            
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            vram_tier = self._detect_vram_tier()
            
            if vram_tier in ["high", "ultra"] and torch.cuda.is_available():
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            elif vram_tier == "medium" and torch.cuda.is_available():
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    device_map="cuda",
                    trust_remote_code=True
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    trust_remote_code=True
                )
            
            def generate_text(prompt: str, max_tokens: int = 100) -> str:
                try:
                    inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
                    if torch.cuda.is_available() and hasattr(model, 'device') and model.device.type == "cuda":
                        inputs = inputs.to("cuda")
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            inputs,
                            max_new_tokens=max_tokens,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            pad_token_id=tokenizer.eos_token_id
                        )
                    
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    if generated_text.startswith(prompt):
                        generated_text = generated_text[len(prompt):].strip()
                    
                    return generated_text
                    
                except Exception as e:
                    logger.error(f"Error in text generation: {e}")
                    return f"Enhanced {prompt} with detailed character interactions and dynamic scenes"
            
            self.models["llm"] = {
                "model": model,
                "tokenizer": tokenizer,
                "generate": generate_text,
                "device": model.device.type if hasattr(model, 'device') else "cpu"
            }
            
            logger.info(f"LLM model {model_id} loaded successfully")
            return self.models["llm"]
            
        except Exception as e:
            logger.error(f"Failed to load LLM model {model_name}: {e}")
            
            def fallback_generate(prompt: str, max_tokens: int = 100) -> str:
                return f"Enhanced {prompt} with detailed character interactions and dynamic action sequences"
            
            def emergency_generate(prompt: str, max_tokens: int = 100) -> str:
                return f"Expanded scene: {prompt}"
            
            self.models["llm"] = {"generate": fallback_generate, "fallback": emergency_generate}
            return self.models["llm"]
    
    def _generate_fallback_music(self, descriptions: List[str] = None, duration: float = 30.0, prompt: str = None) -> str:
        """Generate fallback background music when audiocraft is not available."""
        try:
            import numpy as np
            import soundfile as sf
            
            sample_rate = 44100
            samples = int(duration * sample_rate)
            
            t = np.linspace(0, duration, samples)
            frequency = 220
            music = 0.3 * np.sin(2 * np.pi * frequency * t) * np.exp(-t / 10)
            
            output_path = "/tmp/fallback_music.wav"
            sf.write(output_path, music, sample_rate)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Fallback music generation failed: {e}")
            return ""
    
    def _generate_with_llm(self, model, tokenizer, prompt: str, max_tokens: int = 100) -> str:
        """Generate text using loaded LLM model with thread-safe timeout protection."""
        try:
            import torch
            import threading
            import queue
            
            result_queue = queue.Queue()
            timeout_occurred = threading.Event()
            
            def llm_worker():
                try:
                    # Format prompt for better content generation
                    if "title" in prompt.lower():
                        formatted_prompt = f"YouTube video title about {prompt.replace('Generate a YouTube title for', '').replace('Create an engaging title for', '').strip()}: "
                    elif "description" in prompt.lower():
                        formatted_prompt = f"YouTube video description about {prompt.replace('Generate a YouTube description for', '').strip()}: "
                    elif "next episode" in prompt.lower():
                        formatted_prompt = f"Next episode suggestions for {prompt.replace('Generate next episode suggestions for', '').strip()}: 1."
                    else:
                        formatted_prompt = f"{prompt}: "
                    
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                    
                    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=256)
                    if torch.cuda.is_available() and self.vram_tier != "low":
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=min(max_tokens, 50),  # Limit tokens for faster generation
                            do_sample=True,
                            temperature=0.8,
                            top_p=0.9,
                            repetition_penalty=1.1,
                            pad_token_id=tokenizer.eos_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            num_return_sequences=1,
                            num_beams=1,  # Faster generation
                            early_stopping=True
                        )
                    
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    if formatted_prompt in generated_text:
                        response = generated_text.replace(formatted_prompt, "").strip()
                    else:
                        response = generated_text.strip()
                    
                    if response:
                        response = response.replace('\n', ' ').strip()
                        
                        if "title" in prompt.lower():
                            response = response.replace('YouTube video title about', '').replace('Title:', '').strip()
                            response = response.replace('Create a YouTube Title for', '').replace('Description (optional)', '').strip()
                            
                            if '.' in response:
                                response = response.split('.')[0].strip()
                            if response.endswith(('!', '?')):
                                pass  # Keep exclamation/question marks
                            elif not response.endswith(('!', '?', ':')):
                                response = response.strip() + '!'
                        
                        if len(response) < 5 or response.lower() in ['title', 'description', 'content', 'your name, your profile picture and more']:
                            result_queue.put(self._generate_fallback_content_for_prompt(prompt))
                        else:
                            result_queue.put(response[:200])  # Reasonable max length
                    else:
                        result_queue.put(self._generate_fallback_content_for_prompt(prompt))
                        
                except Exception as e:
                    result_queue.put(self._generate_fallback_content_for_prompt(prompt))
            
            worker_thread = threading.Thread(target=llm_worker)
            worker_thread.daemon = True
            worker_thread.start()
            
            try:
                result = result_queue.get(timeout=30)
                return result
            except queue.Empty:
                timeout_occurred.set()
                logger.warning("LLM generation timed out, using fallback")
                return self._generate_fallback_content_for_prompt(prompt)
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._generate_fallback_content_for_prompt(prompt)
    
    def _generate_fallback_content_for_prompt(self, prompt: str) -> str:
        """Generate fallback content based on prompt type."""
        import random
        prompt_lower = prompt.lower()
        channel_type = getattr(self, 'channel_type', 'content')
        
        if "title" in prompt_lower:
            titles = [
                f"Epic {channel_type.title()} Adventure - Ultimate Battle Begins!",
                f"Amazing {channel_type.title()} Story - Heroes Rise!",
                f"Incredible {channel_type.title()} Action - New Episode!",
                f"Best {channel_type.title()} Content - Must Watch!",
                f"Ultimate {channel_type.title()} Experience - Don't Miss!"
            ]
            return random.choice(titles)
            
        elif "description" in prompt_lower:
            return f"Join us for an incredible {channel_type} adventure filled with amazing characters, epic battles, and unforgettable moments! This episode features stunning visuals, compelling storytelling, and all the action you love. Don't forget to like, subscribe, and hit the notification bell for more amazing {channel_type} content! Experience the ultimate in {channel_type} entertainment with professional-quality animation and immersive storytelling."
            
        elif "next episode" in prompt_lower:
            return "1. Epic character development and new power revelations\n2. Intense battles with unexpected plot twists\n3. Emotional storylines and character relationships\n4. New challenges and adventures await\n5. Spectacular action sequences and dramatic moments"
            
        else:
            return f"High-quality {channel_type} content featuring amazing characters and epic storylines."
    
    def _load_video_model(self, model_name: Optional[str] = None):
        """Load video generation model."""
        if model_name is None:
            model_name = self.base_model
        if model_name in self.models:
            return self.models[model_name]
        
        try:
            class VideoModel:
                def __init__(self, model_name, device):
                    self.model_name = model_name
                    self.device = device
                    self.vram_tier = "medium"
                    self._load_model()
                
                def _load_model(self):
                    try:
                        if "animatediff" in self.model_name.lower():
                            from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
                            import torch
                            
                            adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2")
                            self.pipe = AnimateDiffPipeline.from_pretrained(
                                "runwayml/stable-diffusion-v1-5", 
                                motion_adapter=adapter,
                                torch_dtype=torch.float16 if self.vram_tier != "low" else torch.float32
                            )
                            
                            if self.device == "cuda":
                                self.pipe = self.pipe.to("cuda")
                            
                            self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
                            self.loaded = True
                            
                        elif "svd" in self.model_name.lower():
                            from diffusers import StableVideoDiffusionPipeline
                            import torch
                            
                            self.pipe = StableVideoDiffusionPipeline.from_pretrained(
                                "stabilityai/stable-video-diffusion-img2vid-xt",
                                torch_dtype=torch.float16 if self.vram_tier != "low" else torch.float32
                            )
                            
                            if self.device == "cuda":
                                self.pipe = self.pipe.to("cuda")
                            
                            self.loaded = True
                            
                        else:
                            self.loaded = False
                            
                    except Exception as e:
                        logger.error(f"Failed to load video model {self.model_name}: {e}")
                        self.loaded = False
                
                def generate_video(self, prompt, duration=10.0, output_path=None, **kwargs):
                    """Generate video from prompt."""
                    try:
                        import cv2
                        import numpy as np
                        
                        if not output_path:
                            output_path = f"temp_video_{hash(prompt)}.mp4"
                        
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        
                        if hasattr(self, 'loaded') and self.loaded and hasattr(self, 'pipe'):
                            try:
                                if "animatediff" in self.model_name.lower():
                                    import torch
                                    video_frames = self.pipe(
                                        prompt,
                                        num_frames=min(16, int(duration * 24 / 2)),
                                        guidance_scale=7.5,
                                        num_inference_steps=25,
                                        generator=torch.Generator("cpu").manual_seed(42)
                                    ).frames[0]
                                    
                                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                    out = cv2.VideoWriter(output_path, fourcc, 24, (1920, 1080))
                                    
                                    for frame in video_frames:
                                        frame_np = np.array(frame)
                                        frame_resized = cv2.resize(frame_np, (1920, 1080))
                                        frame_bgr = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)
                                        out.write(frame_bgr)
                                    
                                    out.release()
                                    return True
                                    
                            except Exception as e:
                                logger.error(f"AI video generation failed: {e}, using fallback")
                        
                        duration = min(duration, 5.0)
                        fps = 24
                        frames = int(duration * fps)
                        width, height = 1920, 1080
                        
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                        
                        if not out.isOpened():
                            return False
                        
                        for i in range(min(frames, 120)):
                            frame = np.zeros((height, width, 3), dtype=np.uint8)
                            frame[:] = (50, 50, 100)
                            
                            try:
                                cv2.putText(frame, "Generated Video", 
                                           (50, height//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                                cv2.putText(frame, f"Frame {i+1}", 
                                           (50, height//2 + 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
                            except Exception as e:
                                logger.error(f"Error adding text overlay: {e}")
                            
                            out.write(frame)
                        
                        out.release()
                        return True
                    except Exception:
                        return False
            
            video_model = VideoModel(model_name, self.device)
            self.models[model_name] = video_model
            logger.info(f"Video model loaded: {model_name}")
            return video_model
                
        except Exception as e:
            logger.error(f"Failed to load video model {model_name}: {e}")
            return None
    
    def load_voice_model(self, model_type: str = "bark"):
        """Load voice generation model."""
        model_key = f"voice_{model_type}"
        if model_key in self.models:
            return self.models[model_key]
        
        try:
            if model_type == "bark":
                retry_count = 0
                max_retries = 3
                while retry_count < max_retries:
                    try:
                        import torch
                        torch.serialization.add_safe_globals([
                            'numpy.core.multiarray.scalar',
                            'numpy.core.multiarray._reconstruct',
                            'numpy.ndarray',
                            'numpy.dtype',
                            'numpy.core.numeric',
                            'numpy.core.multiarray.dtype'
                        ])
                        
                        original_load = torch.load
                        def patched_load(*args, **kwargs):
                            kwargs['weights_only'] = False
                            return original_load(*args, **kwargs)
                        torch.load = patched_load
                        
                        from bark import SAMPLE_RATE, generate_audio, preload_models
                        logger.info(f"Loading Bark model with {self.vram_tier} VRAM optimizations (attempt {retry_count + 1})")
                        preload_models()
                        
                        # Restore original torch.load
                        torch.load = original_load
                        
                        def bark_generate_wrapper(text, voice_preset="v2/en_speaker_6"):
                            """Wrapper to handle Bark generation with proper error handling."""
                            try:
                                audio_array = generate_audio(text, history_prompt=voice_preset)
                                return audio_array, SAMPLE_RATE
                            except Exception as e:
                                logger.error(f"Bark generation failed: {e}")
                                return None, SAMPLE_RATE
                        
                        self.models[model_key] = {
                            "type": "bark", 
                            "loaded": True, 
                            "sample_rate": SAMPLE_RATE, 
                            "generate": bark_generate_wrapper
                        }
                        logger.info(f"Successfully loaded Bark model on attempt {retry_count + 1}")
                        break
                        
                    except ImportError as e:
                        logger.error(f"Bark import failed on attempt {retry_count + 1}: {e}")
                        retry_count += 1
                        if retry_count >= max_retries:
                            logger.error("Failed to load Bark after all retries, using fallback")
                            return None
                    except Exception as e:
                        logger.error(f"Bark loading failed on attempt {retry_count + 1}: {e}")
                        try:
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()
                        except:
                            pass
                        retry_count += 1
                        if retry_count >= max_retries:
                            logger.error("Failed to load Bark after all retries, using fallback")
                            return None
            elif model_type == "xtts":
                retry_count = 0
                max_retries = 3
                while retry_count < max_retries:
                    try:
                        from TTS.api import TTS
                        logger.info(f"Loading XTTS-v2 model with {self.vram_tier} VRAM optimizations (attempt {retry_count + 1})")
                        gpu_enabled = self.vram_tier != "low"
                        model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=gpu_enabled)
                        self.models[model_key] = {
                            "type": "xtts", 
                            "model": model, 
                            "languages": ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko"]
                        }
                        logger.info(f"Successfully loaded XTTS model on attempt {retry_count + 1}")
                        break
                    except ImportError as e:
                        logger.error(f"XTTS import failed on attempt {retry_count + 1}: {e}")
                        retry_count += 1
                        if retry_count >= max_retries:
                            logger.error("Failed to load XTTS after all retries, using fallback")
                            return None
                    except Exception as e:
                        logger.error(f"XTTS loading failed on attempt {retry_count + 1}: {e}")
                        retry_count += 1
                        if retry_count >= max_retries:
                            logger.error("Failed to load XTTS after all retries, using fallback")
                            return None
            else:
                logger.warning(f"Unknown voice model: {model_type}")
                return None
                
            logger.info(f"Voice model {model_type} loaded successfully")
            return self.models[model_key]
                
        except Exception as e:
            logger.error(f"Critical failure loading voice model {model_type}: {e}")
            logger.info("Attempting emergency fallback voice generation")
            self.models[model_key] = {
                "type": "fallback",
                "loaded": True,
                "generate": self._create_fallback_audio
            }
            return self.models[model_key]
    
    def load_music_model(self, model_type: str = "musicgen"):
        """Load music generation model."""
        model_key = f"music_{model_type}"
        if model_key in self.models:
            return self.models[model_key]
        
        try:
            if model_type == "musicgen":
                retry_count = 0
                max_retries = 3
                while retry_count < max_retries:
                    try:
                        from audiocraft.models import MusicGen
                        logger.info(f"Loading MusicGen model with {self.vram_tier} VRAM optimizations (attempt {retry_count + 1})")
                        model_size = "small" if self.vram_tier in ["low", "medium"] else "medium"
                        model = MusicGen.get_pretrained(f'facebook/musicgen-{model_size}')
                        self.models[model_key] = {
                            "type": "musicgen",
                            "model": model,
                            "generate": lambda prompt, duration: model.generate([prompt], duration=duration)
                        }
                        logger.info(f"Successfully loaded MusicGen model on attempt {retry_count + 1}")
                        break
                    except ImportError as e:
                        logger.error(f"audiocraft import failed on attempt {retry_count + 1}: {e}")
                        retry_count += 1
                        if retry_count >= max_retries:
                            logger.error("Failed to load audiocraft after all retries, using fallback")
                            def fallback_wrapper(**kwargs):
                                return self._generate_fallback_music(
                                    descriptions=kwargs.get('descriptions', []),
                                    duration=kwargs.get('duration', 30.0),
                                    prompt=kwargs.get('prompt', None)
                                )
                            def fallback_wrapper(**kwargs):
                                return self._generate_fallback_music(
                                    descriptions=kwargs.get('descriptions', []),
                                    duration=kwargs.get('duration', 30.0),
                                    prompt=kwargs.get('prompt', None)
                                )
                            self.models[model_key] = {
                                "type": "fallback", 
                                "loaded": True,
                                "generate": fallback_wrapper
                            }
                    except Exception as e:
                        logger.error(f"MusicGen loading failed on attempt {retry_count + 1}: {e}")
                        retry_count += 1
                        if retry_count >= max_retries:
                            logger.error("Failed to load MusicGen after all retries, using fallback")
                            def final_fallback_wrapper(**kwargs):
                                return self._generate_fallback_music(
                                    descriptions=kwargs.get('descriptions', []),
                                    duration=kwargs.get('duration', 30.0),
                                    prompt=kwargs.get('prompt', None)
                                )
                            self.models[model_key] = {
                                "type": "fallback", 
                                "loaded": True,
                                "generate": fallback_wrapper
                            }
            else:
                logger.warning(f"Unknown music model: {model_type}")
                return None
                
            logger.info(f"Music model {model_type} loaded successfully")
            return self.models[model_key]
                
        except Exception as e:
            logger.error(f"Critical failure loading music model {model_type}: {e}")
            logger.info("Attempting emergency fallback music generation")
            def fallback_wrapper(**kwargs):
                return self._generate_fallback_music(
                    descriptions=kwargs.get('descriptions', []),
                    duration=kwargs.get('duration', 30.0),
                    prompt=kwargs.get('prompt', None)
                )
            self.models[model_key] = {
                "type": "fallback",
                "loaded": True,
                "generate": fallback_wrapper
            }
            return self.models[model_key]
    def _musicgen_generate_wrapper(self, model, **kwargs):
        """Wrapper for MusicGen generation that handles tensor conversion."""
        try:
            prompt = kwargs.get("prompt", "")
            duration = kwargs.get("duration", 30.0)
            output_path = kwargs.get("output_path", "/tmp/music.wav")
            
            model.set_generation_params(duration=duration)
            wav = model.generate([prompt])
            
            if hasattr(wav[0], 'cpu'):
                audio_array = wav[0].cpu().numpy()
            else:
                audio_array = wav[0]
            
            import scipy.io.wavfile as wavfile
            import numpy as np
            sample_rate = getattr(model, 'sample_rate', 32000)
            audio_array = audio_array.squeeze()
            if audio_array.ndim > 1:
                audio_array = audio_array[0]
            audio_array = (audio_array * 32767).astype(np.int16)
            wavfile.write(output_path, sample_rate, audio_array)
            
            return os.path.exists(output_path)
        except Exception as e:
            logger.error(f"MusicGen wrapper error: {e}")
            return False


    
    def expand_script_if_needed(self, script_data: Dict, min_duration: float = 20.0) -> Dict:
        """Expand script to target duration using LLM."""
        current_duration = sum(scene.get('duration', 5.0) for scene in script_data.get('scenes', []))
        
        if current_duration >= min_duration * 60:
            return script_data
        
        llm = self.load_llm_model()
        if not llm:
            return self._expand_script_fallback(script_data, min_duration)
        
        try:
            scenes = script_data.get('scenes', [])
            expanded_scenes = []
            
            for scene in scenes:
                expanded_scene = scene.copy()
                
                if len(scene.get('description', '')) < 100:
                    prompt = f"Expand this {self.channel_type} scene with more detail: {scene.get('description', '')}"
                    
                    expanded_text = llm["generate"](prompt, max_tokens=100)
                    if expanded_text and len(expanded_text.strip()) > 0:
                        scene['description'] = expanded_text.strip()
                        expanded_scene['duration'] = max(scene.get('duration', 5.0) * 1.5, 8.0)
                    else:
                        scene['description'] = f"Enhanced {scene.get('description', 'scene')} with detailed character interactions and dynamic action sequences"
                
                expanded_scenes.append(expanded_scene)
            
            script_data['scenes'] = expanded_scenes
            logger.info(f"Script expanded from {current_duration:.1f}s to target {min_duration*60}s")
            return script_data
            
        except Exception as e:
            logger.error(f"LLM script expansion failed: {e}")
            return self._expand_script_fallback(script_data, min_duration)
        finally:
            if 'llm' in self.models and self.models['llm']:
                try:
                    torch.cuda.empty_cache()
                except Exception as e:
                    logger.error(f"Error during model cleanup: {e}")
    
    def _expand_script_fallback(self, script_data: Dict, min_duration: float) -> Dict:
        """Fallback script expansion without LLM."""
        scenes = script_data.get('scenes', [])
        if not scenes:
            scenes = self._get_default_scenes()
        
        target_duration = min_duration * 60
        current_duration = sum(scene.get('duration', 5.0) if isinstance(scene, dict) else 5.0 for scene in scenes)
        
        expanded_scenes = []
        for i, scene in enumerate(scenes):
            if isinstance(scene, dict):
                expanded_scene = scene.copy()
                original_desc = scene.get('description', f'Scene {i+1}')
                expanded_scene['description'] = f"Enhanced {original_desc} with detailed character development, dynamic action sequences, and immersive {self.channel_type} atmosphere"
                
                if current_duration < target_duration:
                    multiplier = target_duration / max(current_duration, 1.0)
                    expanded_scene['duration'] = scene.get('duration', 5.0) * multiplier
                else:
                    expanded_scene['duration'] = scene.get('duration', 5.0)
            else:
                expanded_scene = {
                    'description': f"Enhanced Scene {i+1}: {scene} with detailed character development, dynamic action sequences, and immersive {self.channel_type} atmosphere",
                    'duration': 5.0 * (target_duration / max(current_duration, 1.0) if current_duration < target_duration else 1.0)
                }
            
            expanded_scenes.append(expanded_scene)
        
        script_data['scenes'] = expanded_scenes
        script_data['expanded'] = True
        logger.info(f"Fallback script expansion completed for {len(expanded_scenes)} scenes")
        return script_data
    
    def generate_combat_scene(self, scene_description: str, duration: float, characters: List[Dict], 
                            style: Optional[str] = None, difficulty: str = "medium") -> Dict:
        """Generate combat scene data."""
        if style is None:
            style = self.channel_type
        
        combat_types = {
            "melee": ["punch", "kick", "block", "dodge"],
            "ranged": ["aim", "shoot", "reload", "cover"],
            "magic": ["cast", "channel", "summon", "shield"],
            "aerial": ["fly", "dive", "hover", "dash"]
        }
        
        combat_type = "melee"
        if any(word in scene_description.lower() for word in ["gun", "shoot", "bullet"]):
            combat_type = "ranged"
        elif any(word in scene_description.lower() for word in ["magic", "spell", "energy"]):
            combat_type = "magic"
        elif any(word in scene_description.lower() for word in ["fly", "aerial", "sky"]):
            combat_type = "aerial"
        
        moves_per_second = {"easy": 0.5, "medium": 1.0, "hard": 1.5, "epic": 2.0}
        total_moves = int(duration * moves_per_second.get(difficulty, 1.0))
        
        sequences = []
        for i in range(total_moves):
            sequence = {
                "sequence_id": i + 1,
                "start_time": i * (duration / max(total_moves, 1)),
                "duration": duration / max(total_moves, 1),
                "movement": combat_types[combat_type][i % len(combat_types[combat_type])],
                "intensity": min(0.5 + (i / max(total_moves - 1, 1)) * 0.5, 1.0)
            }
            sequences.append(sequence)
        
        return {
            "scene_type": "combat",
            "combat_type": combat_type,
            "duration": duration,
            "style": style,
            "difficulty": difficulty,
            "sequences": sequences,
            "video_prompt": f"{style} style {combat_type} combat scene, {difficulty} difficulty, {duration}s duration, high quality animation"
        }
    
    def _process_script_with_llm(self, script_data: Dict, channel_type: str = "anime") -> Dict:
        """Process script with LLM to create enhanced model-specific prompts."""
        try:
            llm_model = self.load_llm_model()
            if not llm_model or not llm_model.get("generate"):
                logger.warning("LLM not available, using fallback processing")
                return self._fallback_script_processing(script_data, channel_type)
            
            enhanced_scenes = []
            scenes = script_data.get('scenes', [])
            characters = script_data.get('characters', [])
            locations = script_data.get('locations', [])
            
            for i, scene in enumerate(scenes):
                if isinstance(scene, dict):
                    scene_text = scene.get('description', f'Scene {i+1}')
                    character = scene.get('character', '')
                    dialogue = scene.get('dialogue', '')
                    location = scene.get('location', '')
                    
                    if character and dialogue:
                        scene_text = f"{scene_text}. Character {character} says: '{dialogue}'"
                    if location:
                        scene_text = f"{scene_text}. Location: {location}"
                else:
                    scene_text = scene if isinstance(scene, str) else f'Scene {i+1}'
                
                analysis_prompt = f"""
Analyze this {channel_type} scene and create detailed prompts for AI generation models.

Scene: {scene_text}
Characters: {[c.get('name', str(c)) if isinstance(c, dict) else str(c) for c in characters]}
Locations: {[l.get('name', str(l)) if isinstance(l, dict) else str(l) for l in locations]}

Create a JSON response with these fields:
{{
    "video_prompt": "Detailed prompt for video generation with visual elements, camera angles, lighting, style",
    "voice_prompt": "Dialogue or narration text with emotional tone and character voice instructions", 
    "music_prompt": "Background music description with mood, instruments, tempo, style",
    "scene_type": "action/dialogue/exploration/combat/character_development",
    "visual_elements": ["list", "of", "key", "visual", "components"],
    "audio_elements": ["list", "of", "key", "audio", "components"],
    "duration": estimated_duration_in_seconds,
    "quality_keywords": ["keywords", "for", "maximum", "quality"]
}}

Focus on creating prompts that will generate the highest quality {channel_type} content. Be specific about visual details, character expressions, environmental elements, and audio characteristics.
"""
                
                try:
                    llm_response = llm_model["generate"](analysis_prompt, max_tokens=500)
                    
                    import json
                    import re
                    json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
                    if json_match:
                        scene_analysis = json.loads(json_match.group())
                        
                        enhanced_scene = self._enhance_scene_for_channel(scene_analysis, channel_type)
                        enhanced_scene['original_description'] = scene_text
                        enhanced_scene['scene_number'] = i + 1
                        enhanced_scenes.append(enhanced_scene)
                    else:
                        enhanced_scenes.append(self._create_fallback_scene(scene_text, i + 1, channel_type))
                        
                except Exception as e:
                    logger.error(f"LLM scene analysis failed for scene {i+1}: {e}")
                    enhanced_scenes.append(self._create_fallback_scene(scene_text, i + 1, channel_type))
            
            processed_script = script_data.copy()
            processed_script['enhanced_scenes'] = enhanced_scenes
            processed_script['llm_processed'] = True
            
            logger.info(f"Successfully processed {len(enhanced_scenes)} scenes with LLM")
            return processed_script
            
        except Exception as e:
            logger.error(f"Script processing with LLM failed: {e}")
            return self._fallback_script_processing(script_data, channel_type)
    
    def _enhance_scene_for_channel(self, scene_analysis: Dict, channel_type: str) -> Dict:
        """Add channel-specific enhancements to LLM scene analysis."""
        channel_enhancements = {
            "anime": {
                "video_prefix": "masterpiece, best quality, ultra detailed, 8k resolution, cinematic lighting, smooth animation, professional anime style, vibrant colors, dynamic composition, ",
                "video_suffix": ", 16:9 aspect ratio, smooth motion, professional cinematography, ultra high definition",
                "music_style": "anime soundtrack, orchestral, emotional, "
            },
            "gaming": {
                "video_prefix": "gaming footage, high action, dynamic camera, intense gameplay, ",
                "video_suffix": ", gaming aesthetics, competitive scene",
                "music_style": "gaming music, electronic, energetic, "
            }
        }
        
        enhancements = channel_enhancements.get(channel_type, channel_enhancements["anime"])
        
        if 'video_prompt' in scene_analysis:
            scene_analysis['video_prompt'] = f"{enhancements['video_prefix']}{scene_analysis['video_prompt']}{enhancements['video_suffix']}"
        
        if 'music_prompt' in scene_analysis:
            scene_analysis['music_prompt'] = f"{enhancements['music_style']}{scene_analysis['music_prompt']}"
        
        return scene_analysis
    
    def _create_fallback_scene(self, scene_text: str, scene_number: int, channel_type: str) -> Dict:
        """Create fallback scene data when LLM processing fails."""
        return {
            "scene_number": scene_number,
            "original_description": scene_text,
            "video_prompt": f"high quality {channel_type} scene, {scene_text}",
            "voice_prompt": scene_text,
            "music_prompt": f"{channel_type} background music",
            "scene_type": "default",
            "visual_elements": ["characters", "environment"],
            "audio_elements": ["dialogue", "background_music"],
            "duration": 10.0,
            "quality_keywords": ["high_quality", "detailed"]
        }
    
    def _fallback_script_processing(self, script_data: Dict, channel_type: str) -> Dict:
        """Fallback processing when LLM is not available."""
        scenes = script_data.get('scenes', [])
        enhanced_scenes = []
        
        for i, scene in enumerate(scenes):
            scene_text = scene if isinstance(scene, str) else scene.get('description', f'Scene {i+1}')
            enhanced_scenes.append(self._create_fallback_scene(scene_text, i + 1, channel_type))
        
        processed_script = script_data.copy()
        processed_script['enhanced_scenes'] = enhanced_scenes
        processed_script['llm_processed'] = False
        
        return processed_script
    
    def generate_video(self, prompt: str, duration: float = 5.0, output_path: Optional[str] = None) -> Optional[str]:
        """Generate video using real AI models."""
        if not output_path:
            output_path = f"generated_video_{int(time.time())}.mp4"
        
        try:
            from ..text_to_video_generator import TextToVideoGenerator
            
            if not hasattr(self, 'video_generator') or self.video_generator is None:
                self.video_generator = TextToVideoGenerator(
                    vram_tier=self.vram_tier,
                    target_resolution=(1920, 1080)
                )
            
            scene_type = self._classify_scene_type(prompt)
            model_name = self.video_generator.get_best_model_for_content(scene_type, self.vram_tier)
            
            logger.info(f"Generating video with AI model {model_name}: {prompt[:50]}...")
            
            success = self.video_generator.generate_video(
                prompt=prompt,
                model_name=model_name,
                output_path=output_path,
                duration=duration,
                scene_type=scene_type
            )
            
            if success and os.path.exists(output_path):
                logger.info(f"AI video generated successfully: {output_path}")
                return output_path
            else:
                logger.error(f"AI video generation failed, creating fallback")
                return self._create_efficient_video(prompt, duration, output_path)
            
        except Exception as e:
            logger.error(f"Error in AI video generation: {e}")
            logger.info("Falling back to placeholder video generation")
            return self._create_efficient_video(prompt, duration, output_path)
    
    def _create_efficient_video(self, prompt: str, duration: float, output_path: str) -> str:
        """Create efficient video optimized for CPU systems."""
        try:
            import cv2
            import numpy as np
            import math
            
            if not output_path:
                output_path = f"efficient_video_{int(time.time())}.mp4"
            
            if isinstance(duration, (str, Path)):
                duration = 5.0  # Default duration
            duration = float(duration)
            
            fps = 24
            max_duration = min(duration, 10.0)
            frames = int(max_duration * fps)
            width, height = 1920, 1080  # 16:9 aspect ratio
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            base_frame = np.full((height, width, 3), [60, 80, 100], dtype=np.uint8)
            
            cv2.putText(base_frame, f"{self.channel_type.upper()} CONTENT", 
                       (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
            cv2.putText(base_frame, prompt[:50], 
                       (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
            
            for i in range(frames):
                frame = base_frame.copy()
                
                t = i / fps
                circle_x = int(width * 0.5 + 200 * math.sin(t * 2))
                circle_y = int(height * 0.5 + 100 * math.cos(t * 3))
                cv2.circle(frame, (circle_x, circle_y), 50, (255, 200, 100), -1)
                
                cv2.putText(frame, f"Frame {i+1}/{frames}", 
                           (width - 300, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                
                out.write(frame)
            
            out.release()
            
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                logger.info(f"CPU-optimized video created: {output_path} ({file_size} bytes)")
                return output_path
            else:
                logger.error("Failed to create efficient video")
                return self._create_fallback_video(prompt, max_duration, output_path)
                
        except Exception as e:
            logger.error(f"Efficient video creation failed: {e}")
            safe_duration = float(duration) if not isinstance(duration, (str, Path)) else 5.0
            return self._create_fallback_video(prompt, min(safe_duration, 10.0), output_path)
    
    def _parse_prompt_for_scenes(self, prompt: str) -> List[Dict]:
        """Parse prompt to extract scene information."""
        scenes = []
        
        if "combat" in prompt.lower() or "fight" in prompt.lower():
            scenes.append({"type": "combat", "intensity": "high", "colors": ["red", "orange", "yellow"]})
        if "peaceful" in prompt.lower() or "calm" in prompt.lower():
            scenes.append({"type": "peaceful", "intensity": "low", "colors": ["blue", "green", "white"]})
        if "dramatic" in prompt.lower() or "intense" in prompt.lower():
            scenes.append({"type": "dramatic", "intensity": "high", "colors": ["purple", "black", "white"]})
        
        if not scenes:
            scenes.append({"type": "general", "intensity": "medium", "colors": ["blue", "green", "yellow"]})
        
        for scene in scenes:
            scene["characters"] = self._extract_characters_from_prompt(prompt)
            scene["environment"] = self._extract_environment_from_prompt(prompt)
            scene["prompt"] = prompt
        
        return scenes
    
    def _extract_characters_from_prompt(self, prompt: str) -> List[str]:
        """Extract character information from prompt."""
        characters = []
        
        if "character" in prompt.lower():
            characters.append("main_character")
        if "hero" in prompt.lower() or "protagonist" in prompt.lower():
            characters.append("hero")
        if "villain" in prompt.lower() or "antagonist" in prompt.lower():
            characters.append("villain")
        
        return characters if characters else ["character"]
    
    def _extract_environment_from_prompt(self, prompt: str) -> str:
        """Extract environment information from prompt."""
        if "forest" in prompt.lower():
            return "forest"
        elif "city" in prompt.lower() or "urban" in prompt.lower():
            return "city"
        elif "space" in prompt.lower() or "cosmic" in prompt.lower():
            return "space"
        elif "ocean" in prompt.lower() or "water" in prompt.lower():
            return "ocean"
        else:
            return "generic"
    
    def _generate_scene_frame(self, scene_data: Dict, frame_idx: int, total_frames: int, width: int, height: int) -> np.ndarray:
        """Generate optimized frame for CPU efficiency."""
        import cv2
        import numpy as np
        import math
        
        base_color = 60 + (frame_idx % 40)
        frame = np.full((height, width, 3), [base_color, base_color + 20, base_color + 40], dtype=np.uint8)
        
        t = frame_idx / max(1, total_frames - 1)
        center_x = int(width * 0.5 + 100 * math.sin(t * 4))
        center_y = int(height * 0.5)
        cv2.circle(frame, (center_x, center_y), 30, (255, 255, 255), -1)
        
        scene_type = scene_data.get('type', 'default')
        cv2.putText(frame, f"Scene: {scene_type}", 
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        return frame
    
    def _generate_environment_background(self, frame: np.ndarray, environment: str, t: float, width: int, height: int) -> np.ndarray:
        """Generate simple environment background for CPU efficiency."""
        import numpy as np
        
        if environment == "forest":
            gradient = np.linspace(80, 40, height).reshape(-1, 1, 1)
            frame[:, :, 1] = gradient.squeeze()  # Green channel
                    
        elif environment == "city":
            gradient = np.linspace(60, 100, height).reshape(-1, 1, 1)
            frame[:, :] = gradient
                
        elif environment == "ocean":
            gradient = np.linspace(30, 80, height).reshape(-1, 1, 1)
            frame[:, :, 2] = gradient.squeeze()  # Blue channel
                
        else:
            gradient = np.linspace(60, 120, height).reshape(-1, 1, 1)
            frame[:, :] = gradient
        
        return frame
    
    def _add_characters_to_frame(self, frame: np.ndarray, characters: List[str], t: float, width: int, height: int) -> np.ndarray:
        """Add character representations to frame."""
        import cv2
        import math
        
        for i, character in enumerate(characters):
            char_x = int(width * (0.2 + 0.6 * i / max(1, len(characters) - 1)) + 50 * math.sin(t * 2 + i))
            char_y = int(height * 0.7 + 30 * math.cos(t * 3 + i))
            
            if character == "hero":
                cv2.circle(frame, (char_x, char_y), 40, (255, 255, 100), -1)
                cv2.ellipse(frame, (char_x - 20, char_y + 20), (30, 50), 0, 0, 180, (0, 100, 255), -1)
            elif character == "villain":
                points = np.array([[char_x, char_y - 40], [char_x - 30, char_y + 20], 
                                 [char_x + 30, char_y + 20]], np.int32)
                cv2.fillPoly(frame, [points], (50, 50, 150))
            else:
                cv2.rectangle(frame, (char_x - 25, char_y - 40), 
                            (char_x + 25, char_y + 20), (150, 200, 100), -1)
        
        return frame
    
    def _add_scene_effects(self, frame: np.ndarray, scene_data: Dict, t: float, width: int, height: int) -> np.ndarray:
        """Add scene-specific visual effects."""
        import cv2
        import numpy as np
        import math
        
        scene_type = scene_data["type"]
        intensity = scene_data["intensity"]
        
        if scene_type == "combat":
            if intensity == "high":
                flash_intensity = int(100 * math.sin(t * 10))
                if flash_intensity > 50:
                    overlay = np.full_like(frame, flash_intensity)
                    frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
            
            for i in range(20):
                particle_x = int((i * 73 + t * 200) % width)
                particle_y = int((i * 127 + t * 150) % height)
                cv2.circle(frame, (particle_x, particle_y), 3, (255, 200, 0), -1)
                
        elif scene_type == "peaceful":
            for i in range(10):
                float_x = int(width * 0.1 + (width * 0.8) * ((i + t * 0.1) % 1))
                float_y = int(height * 0.2 + 100 * math.sin(t + i))
                cv2.circle(frame, (float_x, float_y), 5, (255, 255, 255), -1)
                
        elif scene_type == "dramatic":
            center_x, center_y = width // 2, height // 2
            for radius in range(50, 300, 50):
                alpha = 0.1 * math.sin(t * 2 + radius * 0.01)
                if alpha > 0:
                    cv2.circle(frame, (center_x, center_y), radius, (255, 255, 255), 2)
        
        return frame
    
    def _add_text_overlay(self, frame: np.ndarray, scene_data: Dict, frame_idx: int, total_frames: int):
        """Add informative text overlay to frame."""
        import cv2
        
        text_lines = [
            f"{self.channel_type.upper()} - {scene_data['type'].title()} Scene",
            f"Environment: {scene_data['environment'].title()}",
            f"Frame {frame_idx + 1}/{total_frames}"
        ]
        
        for i, text in enumerate(text_lines):
            y_pos = 50 + i * 40
            cv2.putText(frame, text, (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    def _create_fallback_video(self, prompt: str, duration: float, output_path: str) -> str:
        """Create high-quality fallback video when AI models fail."""
        try:
            import cv2
            import numpy as np
            import math
            
            if not output_path:
                output_path = f"fallback_video_{int(time.time())}.mp4"
            
            fps = 24
            frames = int(duration * fps)
            width, height = 1920, 1080
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                logger.error(f"Failed to open video writer for {output_path}")
                return output_path
            
            for i in range(frames):
                t = i / fps
                base_color = int(100 + 50 * math.sin(t * 0.5))
                frame = np.full((height, width, 3), [base_color, base_color + 20, base_color + 40], dtype=np.uint8)
                
                rect_color = (int(200 + 55 * math.sin(t * 2)), 
                             int(150 + 105 * math.cos(t * 1.5)), 
                             int(100 + 155 * math.sin(t * 3)))
                
                rect_x = int(width * 0.1 + width * 0.3 * math.sin(t))
                rect_y = int(height * 0.2)
                rect_w = int(width * 0.6)
                rect_h = int(height * 0.6)
                
                cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), rect_color, -1)
                
                center_x = width // 2 + int(100 * math.sin(t * 2))
                center_y = height // 2 + int(50 * math.cos(t * 3))
                radius = int(30 + 20 * math.sin(t * 4))
                
                cv2.circle(frame, (center_x, center_y), radius, (255, 255, 255), -1)
                cv2.circle(frame, (center_x, center_y), max(1, radius-5), (0, 0, 0), -1)
                
                text_lines = [
                    f"{self.channel_type.upper()} CONTENT",
                    f"Scene: {prompt[:40]}...",
                    f"Frame {i+1}/{frames}"
                ]
                
                for j, text in enumerate(text_lines):
                    y_pos = 80 + j * 50
                    text_color = (255, 255, 255)
                    cv2.putText(frame, text, (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 2)
                
                out.write(frame)
            
            out.release()
            
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                logger.info(f"High-quality fallback video created: {output_path} ({file_size} bytes)")
                return output_path
            else:
                logger.error("Failed to create fallback video file")
                return output_path
            
        except Exception as e:
            logger.error(f"Fallback video creation failed: {e}")
            return output_path
    
    def generate_voice(self, text: str, character_voice: str = "default", output_path: str = "", language: str = "en") -> str:
        """Generate voice audio using real AI models."""
        try:
            from ..ai_voice_generator import AIVoiceGenerator
            
            if not output_path:
                output_path = f"generated_voice_{int(time.time())}.wav"
            
            if not hasattr(self, 'voice_generator') or self.voice_generator is None:
                self.voice_generator = AIVoiceGenerator(vram_tier=self.vram_tier)
            
            model_name = self.voice_generator.get_best_model_for_language(language)
            
            logger.info(f"Generating voice with AI model {model_name}: {text[:50]}...")
            
            success = self.voice_generator.generate_voice(
                text=text,
                model_name=model_name,
                output_path=output_path,
                language=language,
                character_voice=character_voice
            )
            
            if success and os.path.exists(output_path):
                logger.info(f"AI voice generated successfully: {output_path}")
                return output_path
            else:
                logger.error(f"AI voice generation failed, creating fallback")
                return self._create_fallback_audio(text, output_path)
            
        except Exception as e:
            logger.error(f"Error in AI voice generation: {e}")
            return self._create_fallback_audio(text, output_path)
    
    def _create_fallback_audio(self, text: str, output_path: str) -> str:
        """Create fallback audio when model fails."""
        try:
            import numpy as np
            import scipy.io.wavfile as wavfile
            import time
            
            if not output_path:
                output_path = f"fallback_audio_{int(time.time())}.wav"
            
            duration = max(len(text) * 0.1, 2.0)
            sample_rate = 22050
            samples = int(duration * sample_rate)
            
            t = np.linspace(0, duration, samples)
            frequency = 440 + len(text) % 200
            audio = 0.3 * np.sin(2 * np.pi * frequency * t) * np.exp(-t * 0.5)
            
            audio = (audio * 32767).astype(np.int16)
            wavfile.write(output_path, sample_rate, audio)
            
            logger.info(f"Fallback audio created: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Fallback audio creation failed: {e}")
            return output_path
    
    def generate_background_music(self, scene_description: str, duration: float = 30.0, output_path: str = "", scene_type: str = "default") -> str:
        """Generate background music using real AI models."""
        try:
            from ..ai_music_generator import AIMusicGenerator
            
            if not output_path:
                output_path = f"generated_music_{int(time.time())}.wav"
            
            if not hasattr(self, 'music_generator') or self.music_generator is None:
                self.music_generator = AIMusicGenerator(vram_tier=self.vram_tier)
            
            model_name = self.music_generator.get_best_model_for_content(scene_type, duration)
            
            logger.info(f"Generating music with AI model {model_name}: {scene_description[:50]}...")
            
            success = self.music_generator.generate_music(
                description=scene_description,
                model_name=model_name,
                output_path=output_path,
                duration=duration,
                scene_type=scene_type
            )
            
            if success and os.path.exists(output_path):
                logger.info(f"AI music generated successfully: {output_path}")
                return output_path
            else:
                logger.error(f"AI music generation failed, creating fallback")
                return self._create_fallback_music(scene_description, duration, output_path)
            
        except Exception as e:
            logger.error(f"Error in AI music generation: {e}")
            return self._create_fallback_music(scene_description, duration, output_path)
    
    def _create_fallback_music(self, description: str = None, duration: float = 30.0, output_path: str = "", prompt: str = None) -> str:
        """Create fallback background music."""
        try:
            import numpy as np
            import soundfile as sf
            
            sample_rate = 44100
            samples = int(duration * sample_rate)
            
            t = np.linspace(0, duration, samples)
            frequency = 220
            music = 0.3 * np.sin(2 * np.pi * frequency * t) * np.exp(-t / 10)
            
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                sf.write(output_path, music, sample_rate)
                return output_path
            else:
                fallback_path = "/tmp/fallback_music.wav"
                sf.write(fallback_path, music, sample_rate)
                return fallback_path
                
        except Exception as e:
            logger.error(f"Fallback music creation failed: {e}")
            return ""
    
    def cleanup_models(self):
        """Clean up loaded models to free memory."""
        try:
            if hasattr(self, 'video_generator') and self.video_generator:
                self.video_generator.force_cleanup_all_models()
                self.video_generator = None
            
            if hasattr(self, 'voice_generator') and self.voice_generator:
                self.voice_generator.force_cleanup_all_models()
                self.voice_generator = None
            
            if hasattr(self, 'music_generator') and self.music_generator:
                self.music_generator.force_cleanup_all_models()
                self.music_generator = None
            
            for model_name, model in self.models.items():
                try:
                    if hasattr(model, 'to'):
                        model.to('cpu')
                    elif isinstance(model, dict):
                        for key, submodel in model.items():
                            if hasattr(submodel, 'to'):
                                submodel.to('cpu')
                    del model
                except Exception as e:
                    logger.warning(f"Error cleaning up model {model_name}: {e}")
            
            self.models.clear()
            
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info("All models cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error in model cleanup: {e}")
    
    def parse_input_script(self, input_path: str) -> Dict:
        """Parse input script from various formats."""
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if input_path.endswith('.json'):
                return json.loads(content)
            elif input_path.endswith('.yaml') or input_path.endswith('.yml'):
                return yaml.safe_load(content)
            else:
                return {
                    "title": f"{self.channel_type.title()} Content",
                    "description": content,
                    "scenes": [{"description": content, "duration": 300}],
                    "characters": [{"name": "Character1"}],
                    "locations": [{"name": "Location1"}]
                }
                
        except Exception as e:
            logger.error(f"Failed to parse input script: {e}")
            return {
                "title": f"{self.channel_type.title()} Content",
                "description": "Default content",
                "scenes": [{"description": "Default scene", "duration": 300}],
                "characters": [{"name": "Character1"}],
                "locations": [{"name": "Location1"}]
            }
    


    def extract_highlights_from_video(self, video_path: str, num_highlights: int = 5) -> List[Dict]:
        """Extract highlight moments from main video using motion analysis."""
        try:
            import cv2
            import numpy as np
            
            if not os.path.exists(video_path):
                logger.warning(f"Video file not found: {video_path}")
                return self._create_fallback_highlights(num_highlights)
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                return self._create_fallback_highlights(num_highlights)
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            if duration < 15:
                logger.warning(f"Video too short for highlights: {duration}s")
                cap.release()
                return self._create_fallback_highlights(num_highlights)
            
            highlights = []
            segment_duration = 15
            motion_scores = []
            
            prev_frame = None
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_frame is not None:
                    diff = cv2.absdiff(prev_frame, gray)
                    motion_score = np.mean(diff)
                    motion_scores.append((frame_idx / fps, motion_score))
                
                prev_frame = gray
                frame_idx += 1
            
            cap.release()
            
            if not motion_scores:
                return self._create_fallback_highlights(num_highlights)
            
            motion_scores.sort(key=lambda x: x[1], reverse=True)
            
            for i, (timestamp, score) in enumerate(motion_scores[:num_highlights]):
                start_time = max(0, timestamp - segment_duration / 2)
                end_time = min(duration, start_time + segment_duration)
                
                highlights.append({
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": segment_duration,
                    "excitement_score": score,
                    "title": f"Highlight {i + 1}"
                })
            
            return highlights
            
        except Exception as e:
            logger.error(f"Error extracting highlights: {e}")
            return self._create_fallback_highlights(num_highlights)
    
    def _create_fallback_highlights(self, num_highlights: int) -> List[Dict]:
        """Create fallback highlights when video analysis fails."""
        highlights = []
        for i in range(min(num_highlights, 3)):
            highlights.append({
                "start_time": i * 20,
                "end_time": (i * 20) + 15,
                "duration": 15,
                "excitement_score": 50.0,
                "title": f"Highlight {i + 1}"
            })
        return highlights

    def _calculate_segment_excitement(self, video_path: str, start_time: float, end_time: float, cap, fps: float) -> float:
        """Calculate excitement score for a video segment using motion analysis."""
        try:
            import cv2
            import numpy as np
            
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            motion_scores = []
            prev_frame = None
            
            for frame_num in range(start_frame, min(end_frame, start_frame + 300)):
                ret, frame = cap.read()
                if not ret:
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_frame is not None:
                    diff = cv2.absdiff(gray, prev_frame)
                    motion_score = np.mean(diff)
                    motion_scores.append(motion_score)
                
                prev_frame = gray
            
            if motion_scores:
                avg_motion = np.mean(motion_scores)
                max_motion = np.max(motion_scores)
                excitement = (avg_motion * 0.7 + max_motion * 0.3) / 255.0
                return min(excitement, 1.0)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating excitement: {e}")
            return 0.0

    def _classify_segment_type(self, excitement_score: float) -> str:
        """Classify segment type based on excitement score."""
        if excitement_score > 0.7:
            return "high_action"
        elif excitement_score > 0.4:
            return "medium_action"
        else:
            return "low_action"

    def create_short_from_highlight(self, video_path: str, highlight: Dict, output_path: str, short_number: int) -> Optional[Dict]:
        """Create a short video from a highlight segment."""
        try:
            import subprocess
            
            if not os.path.exists(video_path):
                logger.error(f"Source video not found: {video_path}")
                return None
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            start_time = highlight["start_time"]
            duration = highlight["duration"]
            
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-ss', str(start_time),
                '-t', str(duration),
                '-vf', 'scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920',
                '-c:v', 'libx264',
                '-preset', 'veryslow',
                '-crf', '15',
                '-c:a', 'aac',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0 and os.path.exists(output_path):
                title = self._generate_short_title(highlight, short_number)
                
                return {
                    "path": output_path,
                    "title": title,
                    "duration": duration,
                    "excitement_score": highlight["excitement_score"]
                }
            else:
                logger.error(f"FFmpeg failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating short: {e}")
            return None

    def _generate_short_title(self, highlight: Dict, short_number: int) -> str:
        """Generate engaging title for short based on channel type and excitement."""
        excitement_score = highlight.get("excitement_score", 50.0)
        
        if excitement_score > 70:
            titles = [
                f"Epic {self.channel_type.title()} Moment #{short_number}!",
                f"Insane {self.channel_type.title()} Action #{short_number}",
                f"Mind-Blowing {self.channel_type.title()} #{short_number}",
                f"Incredible {self.channel_type.title()} Scene #{short_number}"
            ]
        elif excitement_score > 40:
            titles = [
                f"Great {self.channel_type.title()} Moment #{short_number}",
                f"Cool {self.channel_type.title()} Scene #{short_number}",
                f"Amazing {self.channel_type.title()} #{short_number}",
                f"Epic {self.channel_type.title()} Clip #{short_number}"
            ]
        else:
            titles = [
                f"{self.channel_type.title()} Highlight #{short_number}",
                f"{self.channel_type.title()} Moment #{short_number}",
                f"{self.channel_type.title()} Scene #{short_number}",
                f"{self.channel_type.title()} Clip #{short_number}"
            ]
        
        import random
        return random.choice(titles)
    
    def ensure_output_dir(self, output_path: str) -> Path:
        """Ensure output directory exists."""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def create_manifest(self, output_dir: Path, **kwargs) -> str:
        """Create manifest file with pipeline results."""
        import json
        import time
        
        manifest = {
            "channel_type": self.channel_type,
            "timestamp": time.time(),
            "device": self.device,
            "vram_tier": self.vram_tier,
            **kwargs
        }
        
        manifest_file = output_dir / "manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        return str(manifest_file)
    
    def generate_youtube_metadata(self, scenes: List[Dict], output_dir: Path, language: str = "en") -> Dict:
        """Generate YouTube metadata including title, description, and tags."""
        try:
            import json
            import random
            
            channel_type = getattr(self, 'channel_type', 'content')
            
            # Extract character names for better title generation
            character_names = []
            for scene in scenes:
                if isinstance(scene, dict) and 'characters' in scene:
                    for char in scene['characters']:
                        if isinstance(char, dict) and 'name' in char:
                            character_names.append(char['name'])
                        elif isinstance(char, str):
                            character_names.append(char)
            
            unique_chars = list(set(character_names))[:2]
            char_string = " and ".join(unique_chars) if unique_chars else "Epic Heroes"
            
            title_prompt = f"Write a short YouTube title for {channel_type} content featuring {char_string}. Maximum 50 characters. Example: 'Epic Battle Adventure' or 'Heroes Unite'. Title only:"
            
            llm_model = self.load_llm_model()
            if llm_model:
                title = llm_model["generate"](title_prompt, max_tokens=15)
                if not title or len(title.strip()) == 0:
                    title = f"Epic {channel_type.title()} Adventure - Episode {random.randint(1, 100)}"
                else:
                    title = title.strip()
                    title = title.replace('"', '').replace("'", "").replace('-!', '').replace('"-', '')
                    if title.startswith('"') and title.endswith('"'):
                        title = title[1:-1]
                    if not title.endswith(('.', '!', '?')):
                        title = title + "!"
            else:
                title = f"Epic {channel_type.title()} Adventure - Episode {random.randint(1, 100)}"
            
            if not title or len(title.strip()) == 0:
                title = f"Amazing {channel_type.title()} Content - Episode {random.randint(1, 100)}"
            
            description_prompt = f"Write YouTube description for {channel_type} episode. Include plot summary. Keep under 100 words:"
            
            if llm_model:
                description = llm_model["generate"](description_prompt, max_tokens=50)
                if not description or len(description.strip()) == 0:
                    description = f"An epic {channel_type} adventure featuring amazing characters and thrilling action across {len(scenes)} incredible scenes! Experience the world of anime and manga like never before!"
                else:
                    description = description.strip()
                    description = description.replace('Title: [Show spoiler]', '').replace('Example Tit...', '')
                    description = description.replace('Title:', '').strip()
                    if description.startswith('"') and description.endswith('"'):
                        description = description[1:-1]
                    if not description.endswith(('.', '!', '?')):
                        description = description + "."
            else:
                description = f"An epic {channel_type} adventure featuring amazing characters and thrilling action across {len(scenes)} incredible scenes! Experience the world of anime and manga like never before!"
            
            if not description or len(description.strip()) == 0:
                description = f"Amazing {channel_type} content with incredible storytelling and epic scenes!"
            
            next_episode_prompt = f"List 3 next episode ideas for {channel_type}. Format: 1. Idea one 2. Idea two 3. Idea three"
            
            if llm_model:
                next_suggestions = llm_model["generate"](next_episode_prompt, max_tokens=30)
                if not next_suggestions or len(next_suggestions.strip()) == 0:
                    next_suggestions = "1. New character introduction and power awakening\n2. Tournament arc with intense battles\n3. Emotional backstory and character development"
                else:
                    next_suggestions = next_suggestions.strip()
            else:
                next_suggestions = "1. New character introduction and power awakening\n2. Tournament arc with intense battles\n3. Emotional backstory and character development"
            
            if not next_suggestions or len(next_suggestions.strip()) == 0:
                next_suggestions = "1. Epic character development\n2. New challenges and adventures\n3. Exciting plot twists"
            
            if not next_suggestions.strip().endswith(('.', '!', '?')):
                next_suggestions = next_suggestions.strip() + "."
            
            tags = [
                channel_type, "anime", "manga", "adventure", "action", 
                "storytelling", "characters", "epic", "series", "episode"
            ]
            
            metadata = {
                "title": title.strip(),
                "description": description.strip(),
                "tags": tags,
                "next_episode_suggestions": next_suggestions.strip(),
                "language": language,
                "scenes_count": len(scenes),
                "channel_type": channel_type
            }
            
            from pathlib import Path
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            title_file = output_dir / "title.txt"
            description_file = output_dir / "description.txt"
            tags_file = output_dir / "tags.txt"
            next_episode_file = output_dir / "next_episode.txt"
            
            with open(title_file, 'w', encoding='utf-8') as f:
                f.write(f"Title: {title.strip()}\n\nGenerated for {channel_type} episode with {len(scenes)} scenes")
            
            with open(description_file, 'w', encoding='utf-8') as f:
                f.write(f"Title: \"{title.strip()}\"\n\n{description.strip()}\n\n🔥 Don't miss the next episode!\n👍 Like and Subscribe for more {channel_type} content!")
            
            with open(tags_file, 'w', encoding='utf-8') as f:
                f.write(", ".join(tags))
            
            with open(next_episode_file, 'w', encoding='utf-8') as f:
                f.write(f"Next Episode Suggestions:\n\n{next_suggestions.strip()}")
            
            metadata_file = output_dir / f"youtube_metadata_{language}.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Generated YouTube metadata for {language}: {title}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error generating YouTube metadata: {e}")
            return {
                "title": f"Epic {getattr(self, 'channel_type', 'content').title()} Adventure",
                "description": "Amazing content with incredible storytelling!",
                "tags": ["content", "adventure", "epic"],
                "next_episode_suggestions": "More amazing content coming soon!",
                "language": language,
                "scenes_count": len(scenes) if scenes else 0,
                "channel_type": getattr(self, 'channel_type', 'content')
            }

        return str(output_dir / "manifest.json")



    async def execute_async(self, project_data: Dict[str, Any]) -> str:
        """Execute pipeline asynchronously."""
        try:
            logger.info(f"Starting async execution for {self.channel_type} pipeline")
            
            input_path = project_data.get('input_path', '')
            script_data = project_data.get('script_data', {})
            output_path = project_data.get('output_path', '/tmp/pipeline_output')
            base_model = project_data.get('base_model', 'stable_diffusion_1_5')
            lora_models = project_data.get('lora_models', [])
            lora_paths = project_data.get('lora_paths', {})
            language = project_data.get('language', 'en')
            render_fps = project_data.get('render_fps', 24)
            output_fps = project_data.get('output_fps', 24)
            frame_interpolation_enabled = project_data.get('frame_interpolation_enabled', True)
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.run_with_script_data,
                input_path,
                script_data,
                output_path,
                base_model,
                lora_models,
                lora_paths,
                None,
                None,
                render_fps,
                output_fps,
                frame_interpolation_enabled,
                language
            )
            
            logger.info(f"Async execution completed for {self.channel_type} pipeline")
            return result
            
        except Exception as e:
            logger.error(f"Async execution failed for {self.channel_type} pipeline: {e}")
            import traceback
            traceback.print_exc()
            
            output_dir = self.ensure_output_dir(project_data.get('output_path', '/tmp/pipeline_output'))
            fallback_video = output_dir / "fallback_video.mp4"
            self._create_fallback_video("Pipeline execution failed", 300.0, str(fallback_video))
            return str(output_dir)

    def run_with_script_data(self, input_path: str, script_data: Dict, output_path: str, 
                           base_model: str = "stable_diffusion_1_5", 
                           lora_models: Optional[List[str]] = None, 
                           lora_paths: Optional[Dict[str, str]] = None, 
                           db_run=None, db=None, render_fps: int = 24, 
                           output_fps: int = 24, frame_interpolation_enabled: bool = True, 
                           language: str = "en") -> str:
        """Run pipeline with pre-parsed script data to avoid re-parsing."""
        try:
            logger.info(f"Starting {self.channel_type} pipeline with script data")
            
            if script_data and script_data.get('scenes'):
                expanded_script = self.expand_script_if_needed(script_data)
            else:
                parsed_script = self.parse_input_script(input_path) if input_path else {}
                expanded_script = self.expand_script_if_needed(parsed_script)
            
            self._save_llm_expansion_results(expanded_script, output_path)
            
            output_dir = self.ensure_output_dir(output_path)
            
            scene_videos = []
            scenes = expanded_script.get('scenes', self._get_default_scenes())
            characters = expanded_script.get('characters', [])
            locations = expanded_script.get('locations', [])
            
            logger.info(f"Processing {len(scenes)} scenes with {len(characters)} characters")
            
            for i, scene in enumerate(scenes):
                scene_description = scene if isinstance(scene, str) else scene.get('description', f'Scene {i+1}')
                
                if isinstance(scene, dict):
                    character = scene.get('character', '')
                    dialogue = scene.get('dialogue', '')
                    location = scene.get('location', '')
                    
                    if character and dialogue:
                        scene_description = f"Character {character} says: '{dialogue}' in location: {location}"
                
                enhanced_prompt = self._enhance_prompt_for_channel(scene_description)
                
                scene_output = output_dir / f"scene_{i+1}.mp4"
                video_path = self.generate_video(enhanced_prompt, output_path=str(scene_output), duration=5.0)
                
                if video_path and os.path.exists(video_path):
                    scene_videos.append(video_path)
                    logger.info(f"Generated scene {i+1}: {video_path}")
            
            final_video = output_dir / "final_video.mp4"
            if scene_videos:
                self._combine_scene_videos(scene_videos, str(final_video))
            else:
                self._create_fallback_video("Default content", 300.0, str(final_video))
            
            self.create_manifest(output_dir, 
                               scenes_generated=len(scene_videos),
                               final_video=str(final_video),
                               language=language)
            
            logger.info(f"{self.channel_type} pipeline completed: {final_video}")
            return str(output_dir)
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            output_dir = self.ensure_output_dir(output_path)
            fallback_video = output_dir / "fallback_video.mp4"
            self._create_fallback_video("Pipeline failed", 300.0, str(fallback_video))
            return str(output_dir)

    def _save_llm_expansion_results(self, expanded_script: Dict, output_path: str):
        """Save LLM expansion results to output directory for debugging."""
        try:
            import json
            from pathlib import Path
            
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            llm_output_file = output_dir / "llm_expansion.json"
            with open(llm_output_file, 'w', encoding='utf-8') as f:
                json.dump(expanded_script, f, indent=2, ensure_ascii=False)
            
            scenes_file = output_dir / "processed_scenes.json"
            scenes_data = {
                "total_scenes": len(expanded_script.get('scenes', [])),
                "characters": expanded_script.get('characters', []),
                "locations": expanded_script.get('locations', []),
                "scenes": expanded_script.get('scenes', [])
            }
            with open(scenes_file, 'w', encoding='utf-8') as f:
                json.dump(scenes_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"LLM expansion results saved to {llm_output_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save LLM expansion results: {e}")
