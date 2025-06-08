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
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            model_name = "microsoft/DialoGPT-medium"
            if self.vram_tier in ["high", "extreme"]:
                model_name = "microsoft/DialoGPT-large"
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=self.dtype,
                device_map="balanced" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            )
            
            if not torch.cuda.is_available():
                model = model.to("cpu")
            
            self.models["llm"] = {"model": model, "tokenizer": tokenizer}
            logger.info(f"LLM model loaded: {model_name}")
            return self.models["llm"]
            
        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            return None
    
    def _load_video_model(self, model_name: str = "animatediff_v2_sdxl"):
        """Load video generation model."""
        if model_name in self.models:
            return self.models[model_name]
        
        try:
            if model_name == "animatediff_v2_sdxl":
                from diffusers import AnimateDiffPipeline, MotionAdapter
                
                adapter = MotionAdapter.from_pretrained(
                    "guoyww/animatediff-motion-adapter-sdxl-beta",
                    torch_dtype=self.dtype,
                    device_map="balanced" if torch.cuda.is_available() else None,
                    low_cpu_mem_usage=True
                )
                if not torch.cuda.is_available():
                    adapter = adapter.to("cpu")
                
                device_map = "balanced" if torch.cuda.is_available() else None
                if device_map:
                    model = AnimateDiffPipeline.from_pretrained(
                        "stabilityai/stable-diffusion-xl-base-1.0",
                        motion_adapter=adapter,
                        torch_dtype=self.dtype,
                        device_map=device_map,
                        low_cpu_mem_usage=True
                    )
                else:
                    model = AnimateDiffPipeline.from_pretrained(
                        "stabilityai/stable-diffusion-xl-base-1.0",
                        motion_adapter=adapter,
                        torch_dtype=self.dtype,
                        low_cpu_mem_usage=True
                    ).to("cpu")
                
                self.models[model_name] = model
                logger.info(f"Video model loaded: {model_name}")
                return model
                
        except Exception as e:
            logger.error(f"Failed to load video model {model_name}: {e}")
            return None
    
    def _load_audio_model(self, model_type: str = "bark"):
        """Load audio generation model."""
        model_key = f"audio_{model_type}"
        if model_key in self.models:
            return self.models[model_key]
        
        try:
            if model_type == "bark":
                from transformers import BarkModel, BarkProcessor
                
                model = BarkModel.from_pretrained(
                    "suno/bark-small",
                    torch_dtype=self.dtype,
                    device_map="balanced" if torch.cuda.is_available() else None,
                    low_cpu_mem_usage=True
                )
                processor = BarkProcessor.from_pretrained("suno/bark-small")
                
                if not torch.cuda.is_available():
                    model = model.to("cpu")
                
                self.models[model_key] = {"model": model, "processor": processor}
                logger.info(f"Audio model loaded: {model_type}")
                return self.models[model_key]
                
        except Exception as e:
            logger.error(f"Failed to load audio model {model_type}: {e}")
            return None
    
    def _load_music_model(self):
        """Load music generation model."""
        if "music" in self.models:
            return self.models["music"]
        
        try:
            from transformers import MusicgenForConditionalGeneration, AutoProcessor
            
            model = MusicgenForConditionalGeneration.from_pretrained(
                "facebook/musicgen-small",
                torch_dtype=self.dtype,
                device_map="balanced" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            )
            processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
            
            if not torch.cuda.is_available():
                model = model.to("cpu")
            
            self.models["music"] = {"model": model, "processor": processor}
            logger.info("Music model loaded")
            return self.models["music"]
            
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
                        outputs = llm["model"].generate(
                            **inputs,
                            max_length=200,
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=llm["tokenizer"].eos_token_id
                        )
                    
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
    
    def _expand_script_fallback(self, script_data: Dict, min_duration: float) -> Dict:
        """Fallback script expansion without LLM."""
        scenes = script_data.get('scenes', [])
        target_duration = min_duration * 60
        current_duration = sum(scene.get('duration', 5.0) for scene in scenes)
        
        if current_duration < target_duration:
            multiplier = target_duration / max(current_duration, 1.0)
            for scene in scenes:
                scene['duration'] = scene.get('duration', 5.0) * multiplier
        
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
        """Generate video from prompt."""
        video_model = self._load_video_model()
        if not video_model:
            return self._create_fallback_video(prompt, duration, output_path)
        
        try:
            if hasattr(video_model, '__call__'):
                result = video_model(
                    prompt=prompt,
                    num_frames=min(int(duration * 8), 24),
                    guidance_scale=7.5,
                    num_inference_steps=20
                )
                
                if hasattr(result, 'frames') and result.frames:
                    frames = result.frames[0]
                    
                    if output_path:
                        import imageio
                        imageio.mimsave(output_path, frames, fps=8)
                        logger.info(f"Video generated: {output_path}")
                        return output_path
            
        except Exception as e:
            logger.error(f"Video generation failed: {e}")
        
        return self._create_fallback_video(prompt, duration, output_path)
    
    def _create_fallback_video(self, prompt: str, duration: float, output_path: str) -> str:
        """Create fallback video when model fails."""
        try:
            import cv2
            import numpy as np
            
            if not output_path:
                output_path = f"fallback_video_{int(time.time())}.mp4"
            
            fps = 24
            frames = int(duration * fps)
            width, height = 1920, 1080
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            for i in range(frames):
                frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
                
                color = (50 + i % 200, 100 + (i * 2) % 150, 150 + (i * 3) % 100)
                cv2.rectangle(frame, (100, 100), (width-100, height-100), color, -1)
                
                text = f"{self.channel_type.upper()}: {prompt[:30]}..."
                cv2.putText(frame, text, (150, height//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                
                out.write(frame)
            
            out.release()
            logger.info(f"Fallback video created: {output_path}")
            return output_path
            
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
            
            inputs = audio_model["processor"](text, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                audio_array = audio_model["model"].generate(**inputs)
            
            if output_path:
                import scipy.io.wavfile as wavfile
                sample_rate = 24000
                wavfile.write(output_path, sample_rate, audio_array.cpu().numpy().squeeze())
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
            
            inputs = music_model["processor"](
                text=[prompt],
                padding=True,
                return_tensors="pt"
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                audio_values = music_model["model"].generate(
                    **inputs,
                    max_new_tokens=int(duration * 50)
                )
            
            if output_path:
                import scipy.io.wavfile as wavfile
                sample_rate = 32000
                audio_data = audio_values[0, 0].cpu().numpy()
                wavfile.write(output_path, sample_rate, audio_data)
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
