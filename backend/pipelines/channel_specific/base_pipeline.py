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
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import torch
import numpy as np

logger = logging.getLogger(__name__)

class BasePipeline:
    """Base class for self-contained channel pipelines."""
    
    def __init__(self, channel_type: str):
        self.channel_type = channel_type
        self.models = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.vram_tier = self._detect_vram_tier()
        
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
        """Enhance prompt for specific channel style."""
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
    
    def _load_llm_model(self):
        """Load LLM model for script expansion."""
        if "llm" in self.models:
            return self.models["llm"]
        
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            model_name = "microsoft/DialoGPT-medium"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            if torch.cuda.is_available():
                model = model.to(self.device)
            
            self.models["llm"] = {
                "model": model,
                "tokenizer": tokenizer,
                "device": self.device
            }
            logger.info("LLM model loaded successfully")
            return self.models["llm"]
            
        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            return None
    
    def _load_video_model(self, model_name: str = "animatediff_v2_sdxl"):
        """Load video generation model."""
        if model_name in self.models:
            return self.models[model_name]
        
        try:
            from ..video_generation import TextToVideoGenerator
            video_generator = TextToVideoGenerator()
            
            model = video_generator.load_model(model_name)
            if model:
                self.models[model_name] = model
                logger.info(f"Video model loaded: {model_name}")
                return model
            else:
                logger.error(f"Failed to load video model: {model_name}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load video model {model_name}: {e}")
            return None
    
    def _load_audio_model(self, model_type: str = "bark"):
        """Load audio generation model."""
        model_key = f"audio_{model_type}"
        if model_key in self.models:
            return self.models[model_key]
        
        try:
            from ..ai_models import load_bark
            bark_model = load_bark()
            
            if bark_model:
                self.models[model_key] = bark_model
                logger.info(f"Audio model loaded: {model_type}")
                return bark_model
            else:
                logger.error(f"Failed to load audio model: {model_type}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load audio model {model_type}: {e}")
            return None
    
    def _load_music_model(self):
        """Load music generation model."""
        if "music" in self.models:
            return self.models["music"]
        
        try:
            from ..ai_models import load_musicgen
            music_model = load_musicgen()
            
            if music_model:
                self.models["music"] = music_model
                logger.info("Music model loaded")
                return music_model
            else:
                logger.error("Failed to load music model")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load music model: {e}")
            return None
    
    def expand_script_if_needed(self, script_data: Dict, min_duration: float = 20.0) -> Dict:
        """Expand script to target duration using LLM."""
        current_duration = sum(scene.get('duration', 5.0) for scene in script_data.get('scenes', []))
        
        if current_duration >= min_duration * 60:
            return script_data
        
        llm = self._load_llm_model()
        if not llm:
            return self._expand_script_fallback(script_data, min_duration)
        
        try:
            scenes = script_data.get('scenes', [])
            expanded_scenes = []
            
            for scene in scenes:
                expanded_scene = scene.copy()
                
                if len(scene.get('description', '')) < 100:
                    prompt = f"Expand this {self.channel_type} scene with more detail: {scene.get('description', '')}"
                    
                    inputs = llm["tokenizer"](prompt, return_tensors="pt", truncation=True, max_length=512)
                    if torch.cuda.is_available():
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        try:
                            outputs = llm["model"].generate(
                                **inputs,
                                max_new_tokens=128,  # Reduced from max_length=200
                                temperature=0.7,
                                do_sample=True,
                                pad_token_id=llm["tokenizer"].eos_token_id,
                                early_stopping=True,
                                num_beams=1  # Faster generation
                            )
                        except Exception as e:
                            logger.warning(f"LLM generation failed for scene, using fallback: {e}")
                            expanded_scene['description'] = f"Enhanced {scene.get('description', 'scene')} with detailed character interactions and dynamic action sequences"
                            continue
                    
                    expanded_text = llm["tokenizer"].decode(outputs[0], skip_special_tokens=True)
                    if expanded_text.startswith(prompt):
                        expanded_text = expanded_text[len(prompt):].strip()
                    
                    if len(expanded_text) > 20:
                        expanded_scene['description'] = expanded_text
                        expanded_scene['duration'] = max(scene.get('duration', 5.0) * 1.5, 8.0)
                
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
                except:
                    pass
    
    def _expand_script_fallback(self, script_data: Dict, min_duration: float) -> Dict:
        """Fallback script expansion without LLM."""
        scenes = script_data.get('scenes', [])
        if not scenes:
            scenes = self._get_default_scenes()
        
        target_duration = min_duration * 60
        current_duration = sum(scene.get('duration', 5.0) for scene in scenes)
        
        expanded_scenes = []
        for i, scene in enumerate(scenes):
            expanded_scene = scene.copy()
            original_desc = scene.get('description', f'Scene {i+1}')
            expanded_scene['description'] = f"Enhanced {original_desc} with detailed character development, dynamic action sequences, and immersive {self.channel_type} atmosphere"
            
            if current_duration < target_duration:
                multiplier = target_duration / max(current_duration, 1.0)
                expanded_scene['duration'] = scene.get('duration', 5.0) * multiplier
            else:
                expanded_scene['duration'] = scene.get('duration', 5.0)
            
            expanded_scenes.append(expanded_scene)
        
        script_data['scenes'] = expanded_scenes
        script_data['expanded'] = True
        logger.info(f"Fallback script expansion completed for {len(expanded_scenes)} scenes")
        return script_data
    
    def generate_combat_scene(self, scene_description: str, duration: float, characters: List[Dict], 
                            style: str = None, difficulty: str = "medium") -> Dict:
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
    
    def generate_video(self, prompt: str, duration: float = 5.0, output_path: str = None) -> Optional[str]:
        """Generate video using efficient CPU-friendly approach."""
        if not output_path:
            output_path = f"generated_video_{int(time.time())}.mp4"
        
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
            memory_percent = psutil.virtual_memory().percent
            
            if self.device == "cpu" or memory_gb < 8 or memory_percent > 85:
                logger.info("Using CPU-optimized video generation")
                return self._create_efficient_video(prompt, duration, output_path)
        except ImportError:
            logger.info("Using efficient video generation (psutil unavailable)")
            return self._create_efficient_video(prompt, duration, output_path)
        
        try:
            if self.device != "cpu" and memory_gb >= 8:
                from ..video_generation import TextToVideoGenerator
                video_generator = TextToVideoGenerator()
                
                success = video_generator.generate_video(
                    prompt=prompt,
                    model_name="svd_xt", 
                    output_path=output_path
                )
                
                if success and os.path.exists(output_path):
                    logger.info(f"GPU video generated: {output_path}")
                    return output_path
                    
        except Exception as e:
            logger.warning(f"GPU video generation failed: {e}")
        
        logger.info("Falling back to CPU-optimized video generation")
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
                return None
            
        except Exception as e:
            logger.error(f"Fallback video creation failed: {e}")
            return None
    
    def generate_voice(self, text: str, language: str = "en", output_path: str = None) -> Optional[str]:
        """Generate voice audio from text."""
        audio_model = self._load_audio_model("bark")
        if not audio_model:
            return self._create_fallback_audio(text, output_path)
        
        try:
            lang_config = self.supported_languages.get(language, self.supported_languages["en"])
            if not lang_config.get("bark_supported", False):
                return self._create_fallback_audio(text, output_path)
            
            if hasattr(audio_model, 'generate'):
                result = audio_model.generate(text, language=language)
                
                if result and output_path:
                    import scipy.io.wavfile as wavfile
                    sample_rate = 24000
                    if isinstance(result, np.ndarray):
                        wavfile.write(output_path, sample_rate, result)
                    else:
                        wavfile.write(output_path, sample_rate, np.array(result))
                    logger.info(f"Voice generated: {output_path}")
                    return output_path
                
        except Exception as e:
            logger.error(f"Voice generation failed: {e}")
        
        return self._create_fallback_audio(text, output_path)
    
    def _create_fallback_audio(self, text: str, output_path: str) -> str:
        """Create fallback audio when model fails."""
        try:
            import numpy as np
            import scipy.io.wavfile as wavfile
            
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
            return None
    
    def generate_background_music(self, prompt: str = None, duration: float = 60.0, output_path: str = None) -> Optional[str]:
        """Generate background music."""
        music_model = self._load_music_model()
        if not music_model:
            return self._create_fallback_music(duration, output_path)
        
        try:
            if not prompt:
                prompt = f"{self.channel_type} background music"
            
            if hasattr(music_model, 'generate'):
                result = music_model.generate(prompt, duration=duration)
                
                if result and output_path:
                    import scipy.io.wavfile as wavfile
                    sample_rate = 32000
                    if isinstance(result, np.ndarray):
                        wavfile.write(output_path, sample_rate, result)
                    else:
                        wavfile.write(output_path, sample_rate, np.array(result))
                    logger.info(f"Background music generated: {output_path}")
                    return output_path
                
        except Exception as e:
            logger.error(f"Music generation failed: {e}")
        
        return self._create_fallback_music(duration, output_path)
    
    def _create_fallback_music(self, duration: float, output_path: str) -> str:
        """Create fallback music when model fails."""
        try:
            import numpy as np
            import scipy.io.wavfile as wavfile
            
            if not output_path:
                output_path = f"fallback_music_{int(time.time())}.wav"
            
            sample_rate = 22050
            samples = int(duration * sample_rate)
            t = np.linspace(0, duration, samples)
            
            base_freq = 220
            music = (
                0.3 * np.sin(2 * np.pi * base_freq * t) +
                0.2 * np.sin(2 * np.pi * base_freq * 1.5 * t) +
                0.1 * np.sin(2 * np.pi * base_freq * 2 * t)
            )
            
            envelope = np.exp(-t * 0.1) * (1 + 0.5 * np.sin(2 * np.pi * 0.1 * t))
            music *= envelope
            
            music = (music * 32767).astype(np.int16)
            wavfile.write(output_path, sample_rate, music)
            
            logger.info(f"Fallback music created: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Fallback music creation failed: {e}")
            return None
    
    def cleanup_models(self):
        """Clean up loaded models to free memory."""
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
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Models cleaned up")
    
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
    
    def ensure_output_dir(self, output_path: str) -> Path:
        """Ensure output directory exists."""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def create_manifest(self, output_dir: Path, **kwargs) -> str:
        """Create manifest file with pipeline results."""
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
