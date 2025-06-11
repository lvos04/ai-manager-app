"""
Real AI Music Generator implementation with MusicGen integration.
Replaces placeholder music generation with actual AI models.
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
import scipy.io.wavfile as wavfile

logger = logging.getLogger(__name__)

class AIMusicGenerator:
    """Real AI music generation using MusicGen and other models."""
    
    def __init__(self, vram_tier: str = "medium"):
        self.vram_tier = vram_tier
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.models = {}
        self.current_model = None
        
        self.model_settings = {
            "musicgen_small": {
                "low": {"use_gpu": False, "max_duration": 15, "precision": "float32"},
                "medium": {"use_gpu": True, "max_duration": 30, "precision": "float16"},
                "high": {"use_gpu": True, "max_duration": 60, "precision": "float16"},
                "ultra": {"use_gpu": True, "max_duration": 120, "precision": "float16"}
            },
            "musicgen_medium": {
                "low": {"use_gpu": False, "max_duration": 10, "precision": "float32"},
                "medium": {"use_gpu": True, "max_duration": 20, "precision": "float16"},
                "high": {"use_gpu": True, "max_duration": 45, "precision": "float16"},
                "ultra": {"use_gpu": True, "max_duration": 90, "precision": "float16"}
            },
            "musicgen_large": {
                "low": {"use_gpu": False, "max_duration": 5, "precision": "float32"},
                "medium": {"use_gpu": True, "max_duration": 15, "precision": "float16"},
                "high": {"use_gpu": True, "max_duration": 30, "precision": "float16"},
                "ultra": {"use_gpu": True, "max_duration": 60, "precision": "float16"}
            }
        }
        
        self.genre_prompts = {
            "action": "intense orchestral music with dramatic percussion and brass",
            "combat": "epic battle music with heavy drums and aggressive orchestration",
            "dialogue": "soft ambient background music with gentle piano and strings",
            "exploration": "atmospheric ambient music with mysterious undertones",
            "character_development": "emotional orchestral music with piano and strings",
            "flashback": "nostalgic piano melody with soft orchestral accompaniment",
            "world_building": "cinematic orchestral music with grand themes",
            "transition": "subtle ambient transition music",
            "anime": "anime-style orchestral music with emotional melodies",
            "gaming": "electronic gaming music with synthesizers and beats",
            "superhero": "heroic orchestral music with powerful brass and percussion",
            "manga": "Japanese-style orchestral music with traditional elements",
            "marvel_dc": "superhero theme music with epic orchestral arrangements"
        }
    
    def get_best_model_for_content(self, content_type: str, duration: float) -> str:
        """Select optimal music model based on content type and duration."""
        vram_gb = self._get_available_vram()
        
        if duration <= 15 and vram_gb >= 8:
            return "musicgen_large"
        elif duration <= 30 and vram_gb >= 12:
            return "musicgen_medium"
        elif duration <= 60:
            return "musicgen_small"
        else:
            return "musicgen_small"
    
    def _get_available_vram(self) -> float:
        """Get available VRAM in GB."""
        try:
            if torch.cuda.is_available():
                return torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return 0
        except:
            return 8
    
    def load_model(self, model_name: str) -> bool:
        """Load music generation model."""
        try:
            if model_name == self.current_model and model_name in self.models:
                return True
            
            self.force_cleanup_all_models()
            
            logger.info(f"Loading music model: {model_name}")
            
            if model_name.startswith("musicgen"):
                return self._load_musicgen(model_name)
            else:
                logger.warning(f"Unknown music model: {model_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading music model {model_name}: {e}")
            return False
    
    def _load_musicgen(self, model_name: str) -> bool:
        """Load MusicGen model with fallback for missing audiocraft package."""
        try:
            try:
                from audiocraft.models import MusicGen
            except ImportError:
                logger.warning("Audiocraft package not available (Python 3.12 compatibility issue)")
                return False
            
            settings = self.model_settings[model_name].get(self.vram_tier, self.model_settings[model_name]["medium"])
            
            model_size = model_name.split("_")[1]
            
            model = MusicGen.get_pretrained(f"facebook/musicgen-{model_size}")
            
            if settings["use_gpu"] and torch.cuda.is_available():
                model = model.to("cuda")
                if settings["precision"] == "float16":
                    model = model.half()
            else:
                model = model.to("cpu")
            
            model.set_generation_params(duration=settings["max_duration"])
            
            self.models[model_name] = {
                "model": model,
                "settings": settings
            }
            
            self.current_model = model_name
            logger.info(f"MusicGen {model_size} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading MusicGen {model_name}: {e}")
            return False
    
    def generate_music(self, description: str, model_name: str, output_path: str, 
                      duration: float = 30.0, scene_type: str = "default") -> bool:
        """Generate music using specified model."""
        try:
            if not self.load_model(model_name):
                logger.error(f"Failed to load music model: {model_name}")
                return self._handle_music_generation_failure(description, duration, output_path)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            optimized_prompt = self.optimize_prompt_for_music(description, scene_type)
            settings = self.models[model_name]["settings"]
            
            actual_duration = min(duration, settings["max_duration"])
            
            logger.info(f"Generating music with {model_name}: {optimized_prompt[:100]}...")
            
            if model_name.startswith("musicgen"):
                return self._generate_musicgen_audio(optimized_prompt, output_path, actual_duration)
            else:
                logger.error(f"Unknown music generation method for {model_name}")
                return self._handle_music_generation_failure(description, duration, output_path)
                
        except Exception as e:
            logger.error(f"Error generating music: {e}")
            return self._handle_music_generation_failure(description, duration, output_path)
    
    def _generate_musicgen_audio(self, prompt: str, output_path: str, duration: float) -> bool:
        """Generate music using MusicGen."""
        try:
            musicgen_model = self.models[self.current_model]
            model = musicgen_model["model"]
            
            model.set_generation_params(duration=duration)
            
            descriptions = [prompt]
            wav = model.generate(descriptions)
            
            audio_array = wav[0].cpu().numpy()
            
            sample_rate = model.sample_rate
            
            audio_array = audio_array.squeeze()
            
            if audio_array.ndim > 1:
                audio_array = audio_array[0]
            
            audio_array = (audio_array * 32767).astype(np.int16)
            
            wavfile.write(output_path, sample_rate, audio_array)
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                logger.info(f"MusicGen audio generated successfully: {output_path}")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error in MusicGen generation: {e}")
            return False
    
    def optimize_prompt_for_music(self, description: str, scene_type: str = "default") -> str:
        """Optimize prompt for music generation."""
        optimized = description.strip()
        
        genre_prompt = self.genre_prompts.get(scene_type, "cinematic background music")
        
        if not any(genre in optimized.lower() for genre in ["music", "orchestral", "piano", "guitar", "drums"]):
            optimized = f"{genre_prompt}, {optimized}"
        
        quality_terms = ["high quality", "professional", "cinematic", "studio quality"]
        if not any(term in optimized.lower() for term in quality_terms):
            optimized = f"{optimized}, high quality cinematic"
        
        if len(optimized) > 200:
            optimized = optimized[:197] + "..."
        
        return optimized
    
    def generate_adaptive_music(self, scene_descriptions: List[str], output_dir: str, 
                               total_duration: float) -> List[str]:
        """Generate adaptive music that changes based on scene content."""
        try:
            music_files = []
            scene_duration = total_duration / len(scene_descriptions) if scene_descriptions else 30.0
            
            for i, scene_desc in enumerate(scene_descriptions):
                scene_type = self._classify_scene_for_music(scene_desc)
                model_name = self.get_best_model_for_content(scene_type, scene_duration)
                
                output_file = os.path.join(output_dir, f"music_scene_{i+1}.wav")
                
                success = self.generate_music(
                    description=scene_desc,
                    model_name=model_name,
                    output_path=output_file,
                    duration=scene_duration,
                    scene_type=scene_type
                )
                
                if success:
                    music_files.append(output_file)
                else:
                    fallback_file = os.path.join(output_dir, f"fallback_music_{i+1}.wav")
                    self._handle_music_generation_failure(scene_desc, scene_duration, fallback_file)
                    music_files.append(fallback_file)
            
            return music_files
            
        except Exception as e:
            logger.error(f"Error generating adaptive music: {e}")
            return []
    
    def _classify_scene_for_music(self, scene_description: str) -> str:
        """Classify scene type for appropriate music generation."""
        scene_desc_lower = scene_description.lower()
        
        if any(word in scene_desc_lower for word in ["fight", "battle", "combat", "attack"]):
            return "combat"
        elif any(word in scene_desc_lower for word in ["action", "chase", "run", "escape"]):
            return "action"
        elif any(word in scene_desc_lower for word in ["talk", "dialogue", "conversation", "speak"]):
            return "dialogue"
        elif any(word in scene_desc_lower for word in ["explore", "discover", "journey", "travel"]):
            return "exploration"
        elif any(word in scene_desc_lower for word in ["emotion", "sad", "happy", "love", "memory"]):
            return "character_development"
        elif any(word in scene_desc_lower for word in ["flashback", "remember", "past", "memory"]):
            return "flashback"
        elif any(word in scene_desc_lower for word in ["world", "environment", "landscape", "city"]):
            return "world_building"
        else:
            return "default"
    
    def combine_music_tracks(self, music_files: List[str], output_path: str, 
                           crossfade_duration: float = 2.0) -> bool:
        """Combine multiple music tracks with crossfading."""
        try:
            if not music_files:
                return False
            
            from pydub import AudioSegment
            
            combined = AudioSegment.empty()
            
            for i, music_file in enumerate(music_files):
                if not os.path.exists(music_file):
                    continue
                
                audio = AudioSegment.from_wav(music_file)
                
                if i == 0:
                    combined = audio
                else:
                    crossfade_ms = int(crossfade_duration * 1000)
                    combined = combined.append(audio, crossfade=crossfade_ms)
            
            combined.export(output_path, format="wav")
            
            logger.info(f"Music tracks combined successfully: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error combining music tracks: {e}")
            return self._combine_music_ffmpeg(music_files, output_path)
    
    def _combine_music_ffmpeg(self, music_files: List[str], output_path: str) -> bool:
        """Fallback music combination using FFmpeg."""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                concat_file = f.name
                for music_file in music_files:
                    if os.path.exists(music_file):
                        f.write(f"file '{music_file}'\n")
            
            cmd = [
                'ffmpeg', '-f', 'concat', '-safe', '0', '-i', concat_file,
                '-c', 'copy', '-y', output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if os.path.exists(concat_file):
                os.unlink(concat_file)
            
            if result.returncode == 0:
                logger.info("Music files combined successfully with FFmpeg")
                return True
            else:
                logger.error(f"FFmpeg music combination failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error in FFmpeg music combination: {e}")
            return False
    
    def _handle_music_generation_failure(self, description: str, duration: float, output_path: str) -> bool:
        """Handle music generation failure by trying alternative models."""
        try:
            alternative_models = ["musicgen_small", "musicgen_medium"]
            for model_name in alternative_models:
                if model_name != self.current_model:
                    if self.load_model(model_name):
                        return self.generate_music(description, model_name, output_path, duration)
            
            logger.error("All music models failed, cannot generate music")
            return False
            
        except Exception as e:
            logger.error(f"Error in music generation failure handling: {e}")
            return False
    
    def force_cleanup_all_models(self):
        """Force cleanup of all loaded music models."""
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
            
            logger.info("All music models cleaned up")
            
        except Exception as e:
            logger.error(f"Error in music model cleanup: {e}")
