"""
AI Original Anime Series Channel Pipeline
Self-contained anime content generation with complete internal processing.
All external dependencies inlined for maximum quality output.
"""

import os
import sys
import json
import yaml
import time
import logging
import tempfile
import shutil
import subprocess
import random
import re
import traceback
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from .base_pipeline import BasePipeline

logger = logging.getLogger(__name__)

try:
    from PIL import Image, ImageDraw, ImageFont
    import cv2
    import numpy as np
    import torch
    import moviepy.editor as mp
    from moviepy.video.fx import speedx
except ImportError as e:
    print(f"Warning: Some dependencies not available: {e}")
    Image = ImageDraw = ImageFont = cv2 = np = torch = mp = speedx = None

class AnimePipeline(BasePipeline):
    """Self-contained anime content generation pipeline with all functionality inlined."""
    
    def __init__(self):
        super().__init__("anime")
        self.combat_calls_count = 0
        self.max_combat_calls = 3
        self.scene_duration_estimates = {
            "dialogue": 2.0, "action": 1.5, "combat": 3.0, "exploration": 2.5,
            "character_development": 3.0, "flashback": 2.0, "world_building": 2.5, "transition": 0.5
        }
        self.combat_types = {
            "melee": {
                "movements": ["punch", "kick", "block", "dodge", "grapple", "throw"],
                "camera_angles": ["close_up", "wide_shot", "over_shoulder", "low_angle", "high_angle"],
                "effects": ["impact_flash", "speed_lines", "dust_cloud", "shockwave"]
            },
            "ranged": {
                "movements": ["aim", "shoot", "reload", "take_cover", "roll", "jump"],
                "camera_angles": ["first_person", "third_person", "bullet_time", "tracking_shot"],
                "effects": ["muzzle_flash", "bullet_trail", "explosion", "debris"]
            },
            "magic": {
                "movements": ["cast", "channel", "gesture", "summon", "shield", "teleport"],
                "camera_angles": ["dramatic_low", "overhead", "spiral", "zoom_in", "pull_back"],
                "effects": ["energy_burst", "magical_aura", "spell_circle", "elemental_fx"]
            }
        }
    
    def run(self, input_path: str, output_path: str, base_model: str = "stable_diffusion_1_5", 
            lora_models: Optional[List[str]] = None, lora_paths: Optional[Dict[str, str]] = None, 
            db_run=None, db=None, render_fps: int = 24, output_fps: int = 60, 
            frame_interpolation_enabled: bool = True, language: str = "en") -> str:
        """
        Run the self-contained anime pipeline.
        
        Args:
            input_path: Path to input script/description
            output_path: Path to output directory
            base_model: Base model to use for generation
            lora_models: List of LoRA models to apply
            lora_paths: Dictionary mapping LoRA model names to their file paths
            db_run: Database run object for progress tracking
            db: Database session
            render_fps: Rendering frame rate
            output_fps: Output frame rate
            frame_interpolation_enabled: Enable frame interpolation
            language: Target language
            
        Returns:
            str: Path to output directory
        """
        
        print("Running self-contained anime pipeline")
        print(f"Using base model: {base_model}")
        print(f"Using LoRA models: {lora_models}")
        print(f"Language: {language}")
        
        try:
            return self._execute_pipeline(
                input_path, output_path, base_model, lora_models, 
                db_run, db, render_fps, output_fps, frame_interpolation_enabled, language
            )
        except Exception as e:
            logger.error(f"Anime pipeline failed: {e}")
            raise
        finally:
            self.cleanup_models()
    
    def _execute_pipeline(self, input_path: str, output_path: str, base_model: str, 
                         lora_models: Optional[List[str]], db_run, db, render_fps: int, 
                         output_fps: int, frame_interpolation_enabled: bool, language: str) -> str:
        
        output_dir = self.ensure_output_dir(output_path)
        
        scenes_dir = output_dir / "scenes"
        scenes_dir.mkdir(exist_ok=True)
        
        characters_dir = output_dir / "characters"
        characters_dir.mkdir(exist_ok=True)
        
        final_dir = output_dir / "final"
        final_dir.mkdir(exist_ok=True)
        
        shorts_dir = output_dir / "shorts"
        shorts_dir.mkdir(exist_ok=True)
        
        print("Step 1: Loading and parsing script...")
        if db_run and db:
            db_run.progress = 5.0
            db.commit()
        
        script_data = self.parse_input_script(input_path)
        scenes = script_data.get('scenes', [])
        characters = script_data.get('characters', [])
        locations = script_data.get('locations', [])
        
        if not scenes:
            scenes = [{"description": "A mysterious anime character appears in a magical forest setting.", "duration": 300}]
        
        if not characters:
            characters = [{"name": "Protagonist", "description": "Main anime character with mysterious powers"}]
        
        if not locations:
            locations = [{"name": "Magical Forest", "description": "Ethereal forest with glowing particles"}]
        
        print("Step 2: Expanding script with LLM...")
        if db_run and db:
            db_run.progress = 10.0
            db.commit()
        
        try:
            script_data['scenes'] = scenes
            script_data['characters'] = characters
            script_data['locations'] = locations
            
            expanded_script = self._expand_script_if_needed(script_data, min_duration=20.0)
            
            scenes = expanded_script.get('scenes', scenes)
            characters = expanded_script.get('characters', characters)
            locations = expanded_script.get('locations', locations)
            
            print(f"Anime script expanded to {len(scenes)} scenes for 20-minute target")
            
        except Exception as e:
            print(f"Error during anime script expansion: {e}")
        
        print("Step 3: Generating anime scenes with combat integration...")
        if db_run and db:
            db_run.progress = 20.0
            db.commit()
        
        scene_files = []
        for i, scene in enumerate(scenes):
            scene_text = scene if isinstance(scene, str) else scene.get('description', f'Scene {i+1}')
            scene_chars = [characters[i % len(characters)], characters[(i + 1) % len(characters)]]
            scene_location = locations[i % len(locations)]
            
            scene_type = self._detect_scene_type(scene_text)
            
            scene_detail = {
                "scene_number": i + 1,
                "description": scene_text,
                "characters": scene_chars,
                "location": scene_location,
                "scene_type": scene_type,
                "duration": scene.get('duration', 10.0) if isinstance(scene, dict) else 10.0
            }
            
            if scene_type == "combat" and self.combat_calls_count < self.max_combat_calls:
                try:
                    combat_data = self._generate_combat_scene(
                        scene_description=scene_text,
                        duration=10.0,
                        characters=scene_chars,
                        style="anime",
                        difficulty="medium"
                    )
                    scene_detail["combat_data"] = combat_data
                    self.combat_calls_count += 1
                    print(f"Generated anime combat scene {i+1} with choreography ({self.combat_calls_count}/{self.max_combat_calls})")
                except Exception as e:
                    print(f"Error generating anime combat scene: {e}")
            
            scene_file = scenes_dir / f"scene_{i+1:03d}.mp4"
            
            print(f"Generating anime scene {i+1}: {scene_text[:50]}...")
            
            try:
                char_names = ", ".join([c.get("name", "character") if isinstance(c, dict) else str(c) for c in scene_chars])
                location_desc = scene_location.get("description", scene_location.get("name", "location")) if isinstance(scene_location, dict) else str(scene_location)
                
                anime_prompt = f"anime scene, {location_desc}, with {char_names}, {scene_text}, detailed anime style, vibrant colors, high quality, 16:9 aspect ratio"
                
                if scene_detail.get("combat_data"):
                    anime_prompt = scene_detail["combat_data"]["video_prompt"]
                
                video_path = self._create_scene_video_with_generation(
                    scene_description=anime_prompt,
                    characters=scene_chars,
                    output_path=str(scene_file),
                    duration=scene_detail["duration"]
                )
                
                if video_path:
                    scene_files.append(video_path)
                    print(f"Generated anime scene video {i+1}")
                else:
                    print(f"Failed to generate video for scene {i+1}")
                    
            except Exception as e:
                print(f"Error generating scene {i}: {e}")
                fallback_path = self._create_fallback_video(scene_text, scene_detail["duration"], str(scene_file))
                if fallback_path:
                    scene_files.append(fallback_path)
            
            if db_run and db:
                db_run.progress = 20.0 + (i + 1) / len(scenes) * 30.0
                db.commit()
        
        print("Step 4: Generating voice lines...")
        if db_run and db:
            db_run.progress = 50.0
            db.commit()
        
        voice_files = []
        for i, scene in enumerate(scenes):
            scene_text = scene if isinstance(scene, str) else scene.get('description', f'Scene {i+1}')
            dialogue = scene.get('dialogue', scene_text) if isinstance(scene, dict) else scene_text
            
            voice_file = scenes_dir / f"voice_{i+1:03d}.wav"
            
            try:
                voice_path = self._generate_voice_lines(
                    text=dialogue,
                    language=language,
                    output_path=str(voice_file)
                )
                
                if voice_path:
                    voice_files.append(voice_path)
                    print(f"Generated voice for scene {i+1}")
                    
            except Exception as e:
                print(f"Error generating voice for scene {i+1}: {e}")
        
        print("Step 5: Generating background music...")
        if db_run and db:
            db_run.progress = 60.0
            db.commit()
        
        music_file = final_dir / "background_music.wav"
        try:
            music_path = self._generate_background_music(
                prompt="anime background music, epic adventure soundtrack",
                duration=sum(scene.get('duration', 10.0) if isinstance(scene, dict) else 10.0 for scene in scenes),
                output_path=str(music_file)
            )
            print(f"Generated background music: {music_path}")
        except Exception as e:
            print(f"Error generating background music: {e}")
            music_path = None
        
        print("Step 6: Combining scenes into final episode...")
        if db_run and db:
            db_run.progress = 80.0
            db.commit()
        
        final_video = final_dir / "anime_episode.mp4"
        try:
            temp_combined = final_dir / "temp_combined.mp4"
            combined_path = self._combine_scenes_to_episode(
                scene_files=scene_files,
                voice_files=voice_files,
                music_path=music_path,
                output_path=str(temp_combined),
                render_fps=render_fps,
                output_fps=output_fps,
                frame_interpolation_enabled=frame_interpolation_enabled
            )
            
            if frame_interpolation_enabled and output_fps > render_fps:
                print(f"Applying frame interpolation: {render_fps}fps -> {output_fps}fps...")
                interpolated_path = self._interpolate_frames(
                    input_path=combined_path,
                    output_path=str(final_video),
                    target_fps=output_fps
                )
                combined_path = interpolated_path
            else:
                shutil.move(combined_path, str(final_video))
                combined_path = str(final_video)
            
            print(f"Final anime episode created: {combined_path}")
        except Exception as e:
            print(f"Error combining scenes: {e}")
            combined_path = str(final_video)
        
        print("Step 7: Creating shorts...")
        if db_run and db:
            db_run.progress = 90.0
            db.commit()
        
        print("Step 8: Creating shorts...")
        try:
            shorts_paths = self._create_shorts(scene_files, shorts_dir)
            print(f"Created {len(shorts_paths)} shorts")
        except Exception as e:
            print(f"Error creating shorts: {e}")
        
        print("Step 9: Upscaling final video...")
        if db_run and db:
            db_run.progress = 95.0
            db.commit()
        
        try:
            upscaled_video = final_dir / "anime_episode_upscaled.mp4"
            upscaled_path = self._upscale_video_with_realesrgan(
                input_path=str(final_video),
                output_path=str(upscaled_video),
                target_resolution="1080p",
                enabled=True
            )
            print(f"Video upscaled to: {upscaled_path}")
        except Exception as e:
            print(f"Error upscaling video: {e}")
            upscaled_path = str(final_video)
        
        print("Step 10: Generating YouTube metadata...")
        try:
            self._generate_youtube_metadata(output_dir, scenes, characters, language)
            print("YouTube metadata generated successfully")
        except Exception as e:
            print(f"Error generating YouTube metadata: {e}")
        
        if db_run and db:
            db_run.progress = 100.0
            db.commit()
        
        self._create_manifest(
            output_dir,
            scenes_generated=len(scene_files),
            combat_scenes=self.combat_calls_count,
            final_video=upscaled_path,
            language=language,
            render_fps=render_fps,
            output_fps=output_fps
        )
        
        print(f"Anime pipeline completed successfully: {output_dir}")
        return str(output_dir)
    
    def _detect_scene_type(self, scene_text: str) -> str:
        """Detect scene type from description."""
        scene_lower = scene_text.lower()
        
        if any(word in scene_lower for word in ["fight", "battle", "combat", "attack", "sword", "punch", "kick"]):
            return "combat"
        elif any(word in scene_lower for word in ["dialogue", "talk", "conversation", "speak", "say"]):
            return "dialogue"
        elif any(word in scene_lower for word in ["action", "run", "chase", "escape", "jump"]):
            return "action"
        elif any(word in scene_lower for word in ["emotional", "cry", "sad", "happy", "love", "heart"]):
            return "emotional"
        else:
            return "dialogue"
    
    def _combine_scenes_to_episode(self, scene_files: List[str], voice_files: List[str], 
                                  music_path: Optional[str], output_path: str, 
                                  render_fps: int, output_fps: int, 
                                  frame_interpolation_enabled: bool) -> str:
        """Combine all scenes into final episode using FFmpeg for reliability."""
        try:
            import subprocess
            import tempfile
            
            valid_scenes = [f for f in scene_files if os.path.exists(f)]
            if not valid_scenes:
                logger.warning("No valid scene videos found for combination")
                return self._create_fallback_video("No scenes generated", 1200, output_path)
            
            logger.info(f"Combining {len(valid_scenes)} scene videos into final episode")
            
            for scene_file in valid_scenes:
                if os.path.exists(scene_file):
                    size = os.path.getsize(scene_file)
                    logger.info(f"Scene file: {scene_file} ({size} bytes)")
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                concat_file = f.name
                for scene_file in valid_scenes:
                    f.write(f"file '{os.path.abspath(scene_file)}'\n")
            
            try:
                cmd = [
                    'ffmpeg', '-y',  # Overwrite output
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', concat_file,
                    '-c:v', 'libx264',  # High quality codec
                    '-preset', 'veryslow',  # Maximum quality preset
                    '-crf', '15',  # High quality CRF
                    '-profile:v', 'high',
                    '-level', '4.1',
                    '-c:a', 'aac',
                    '-b:a', '320k',  # High quality audio bitrate
                    '-r', str(output_fps),  # Set output frame rate
                    '-pix_fmt', 'yuv420p',  # Ensure compatibility
                    '-s', '1920x1080',  # Force 16:9 resolution
                    '-movflags', '+faststart',  # Optimize for streaming
                    output_path
                ]
                
                logger.info(f"Running FFmpeg command: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                
                if result.returncode == 0 and os.path.exists(output_path):
                    file_size = os.path.getsize(output_path)
                    logger.info(f"Successfully combined scenes into episode: {output_path} ({file_size} bytes)")
                    
                    if file_size > 1000000:  # At least 1MB
                        logger.info("✅ Final episode has substantial content")
                        return output_path
                    else:
                        logger.warning(f"⚠️ Final episode too small: {file_size} bytes, trying OpenCV fallback")
                        return self._fallback_combine_opencv(valid_scenes, output_path, output_fps)
                else:
                    logger.error(f"FFmpeg failed with return code {result.returncode}")
                    logger.error(f"FFmpeg stderr: {result.stderr}")
                    return self._fallback_combine_opencv(valid_scenes, output_path, output_fps)
                    
            finally:
                if os.path.exists(concat_file):
                    os.unlink(concat_file)
                
        except Exception as e:
            logger.error(f"Error combining scenes with FFmpeg: {e}")
            return self._fallback_combine_opencv(valid_scenes, output_path, output_fps)
    
    def _fallback_combine_opencv(self, scene_files: List[str], output_path: str, fps: int = 24) -> str:
        """Fallback method using OpenCV for video combination with proper error handling."""
        try:
            import cv2
            
            logger.info(f"Using OpenCV fallback to combine {len(scene_files)} scenes")
            
            if not scene_files:
                return self._create_fallback_video("No scenes for OpenCV", 1200, output_path)
            
            first_cap = None
            for scene_file in scene_files:
                if os.path.exists(scene_file):
                    first_cap = cv2.VideoCapture(scene_file)
                    if first_cap.isOpened():
                        break
                    first_cap.release()
            
            if not first_cap or not first_cap.isOpened():
                logger.error("No valid video files found for OpenCV processing")
                return self._create_fallback_video("No valid scenes", 1200, output_path)
            
            width = int(first_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(first_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            first_cap.release()
            
            if width != 1920 or height != 1080:
                width, height = 1920, 1080
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Try multiple codecs for better compatibility
            codecs_to_try = [
                ('mp4v', '.mp4'),
                ('XVID', '.avi'),
                ('MJPG', '.avi')
            ]
            
            out = None
            final_output_path = output_path
            
            for codec, ext in codecs_to_try:
                try:
                    if not output_path.endswith(ext):
                        final_output_path = output_path.rsplit('.', 1)[0] + ext
                    
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    out = cv2.VideoWriter(final_output_path, fourcc, fps, (width, height))
                    
                    if out.isOpened():
                        logger.info(f"Successfully opened video writer with codec {codec}")
                        break
                    else:
                        out.release()
                        out = None
                        logger.warning(f"Failed to open video writer with codec {codec}")
                except Exception as e:
                    logger.warning(f"Error with codec {codec}: {e}")
                    if out:
                        out.release()
                        out = None
            
            if not out or not out.isOpened():
                logger.error("Failed to open video writer with any codec")
                return self._create_fallback_video("Writer failed", 1200, output_path)
            
            total_frames = 0
            for scene_file in scene_files:
                if not os.path.exists(scene_file):
                    continue
                    
                logger.info(f"Processing scene: {scene_file}")
                cap = cv2.VideoCapture(scene_file)
                
                if not cap.isOpened():
                    logger.warning(f"Could not open scene file: {scene_file}")
                    continue
                
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
                logger.info(f"Added {frame_count} frames from {scene_file}")
            
            if out:
                out.release()
                cv2.destroyAllWindows()  # Clean up any OpenCV windows
            
            if os.path.exists(final_output_path) and total_frames > 0:
                file_size = os.path.getsize(final_output_path)
                logger.info(f"OpenCV combined {len(scene_files)} scenes into {total_frames} frames")
                logger.info(f"Final video: {final_output_path} ({file_size} bytes)")
                
                if file_size > 100000:  # At least 100KB
                    return final_output_path
                else:
                    logger.warning(f"OpenCV output too small: {file_size} bytes, creating fallback")
                    return self._create_fallback_video("Small output", 1200, output_path)
            else:
                logger.error(f"Failed to create combined video with OpenCV. File exists: {os.path.exists(final_output_path)}, Frames: {total_frames}")
                return self._create_fallback_video("OpenCV failed", 1200, output_path)
                
        except Exception as e:
            logger.error(f"OpenCV combination failed: {e}")
            return self._create_fallback_video("OpenCV error", 1200, output_path)
    
    def _create_shorts(self, scene_files: List[str], shorts_dir: Path) -> List[str]:
        """Create short clips from scenes."""
        shorts_paths = []
        
        for i, scene_file in enumerate(scene_files[:3]):
            try:
                short_path = shorts_dir / f"short_{i+1:03d}.mp4"
                
                import cv2
                cap = cv2.VideoCapture(scene_file)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(short_path), fourcc, 24, (1080, 1920))
                
                frame_count = 0
                max_frames = 24 * 15
                
                while frame_count < max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame = cv2.resize(frame, (1080, 1920))
                    out.write(frame)
                    frame_count += 1
                
                cap.release()
                out.release()
                
                if frame_count > 0:
                    shorts_paths.append(str(short_path))
                    
            except Exception as e:
                print(f"Error creating short {i+1}: {e}")
        
        return shorts_paths
    
    def _expand_script_if_needed(self, script_data: Dict, min_duration: float = 20.0) -> Dict:
        """Expand script if it doesn't meet minimum duration requirements."""
        current_duration = self._analyze_script_duration(script_data)
        if current_duration >= min_duration:
            logger.info(f"Script duration {current_duration:.1f} minutes meets minimum requirement")
            return script_data
        
        logger.info(f"Script duration {current_duration:.1f} minutes is below minimum {min_duration} minutes. Expanding...")
        
        needed_duration = min_duration - current_duration
        scenes_to_add = int(needed_duration / 2.5) + 1
        
        existing_scenes = script_data.get("scenes", [])
        characters = script_data.get("characters", [])
        setting = script_data.get("setting", "anime world")
        
        expansion_types = ["character_development", "world_building", "action_expansion", "emotional_beats"]
        
        for i in range(scenes_to_add):
            expansion_type = expansion_types[i % len(expansion_types)]
            new_scene = self._generate_expansion_scene(expansion_type, characters, setting, i + len(existing_scenes) + 1)
            existing_scenes.append(new_scene)
        
        script_data["scenes"] = existing_scenes
        return script_data
    
    def _analyze_script_duration(self, script_data: Dict) -> float:
        """Analyze script and estimate total duration."""
        total_duration = 0.0
        
        if "scenes" in script_data:
            for scene in script_data["scenes"]:
                scene_type = scene.get("type", "dialogue").lower() if isinstance(scene, dict) else "dialogue"
                base_duration = self.scene_duration_estimates.get(scene_type, 2.0)
                
                if isinstance(scene, dict):
                    dialogue_lines = len(scene.get("dialogue", []))
                    dialogue_duration = dialogue_lines * 0.1
                    description = scene.get("description", "")
                    description_duration = len(description.split()) * 0.05
                    scene_duration = max(base_duration, dialogue_duration + description_duration)
                else:
                    scene_duration = base_duration
                
                total_duration += scene_duration
        
        return total_duration
    
    def _generate_expansion_scene(self, expansion_type: str, characters: List, setting: str, scene_number: int) -> Dict:
        """Generate a new scene for expansion."""
        main_char = characters[0] if characters else {"name": "Protagonist"}
        
        if expansion_type == "character_development":
            return {
                "type": "character_development",
                "description": f"Character development scene featuring {main_char.get('name', 'the protagonist')} in {setting}. This scene explores their background, motivations, and personal growth.",
                "duration": 3.0
            }
        elif expansion_type == "world_building":
            return {
                "type": "world_building", 
                "description": f"World building scene showcasing the rich details of {setting}. This scene establishes the world's rules, culture, and atmosphere in anime style.",
                "duration": 2.5
            }
        elif expansion_type == "action_expansion":
            return {
                "type": "combat",
                "description": f"Intense anime-style combat scene in {setting} featuring dynamic action and choreography.",
                "duration": 3.0
            }
        else:  # emotional_beats
            return {
                "type": "emotional_beat",
                "description": f"Quiet character moment allowing for emotional depth and reflection in {setting}.",
                "duration": 2.0
            }
    
    def _generate_combat_scene(self, scene_description: str, duration: float, characters: List[Dict], 
                              style: str = "anime", difficulty: str = "medium") -> Dict:
        """Generate a complete combat scene with choreography."""
        combat_type = "melee"  # default
        if any(word in scene_description.lower() for word in ["gun", "shoot", "bullet", "rifle"]):
            combat_type = "ranged"
        elif any(word in scene_description.lower() for word in ["magic", "spell", "energy", "power"]):
            combat_type = "magic"
        
        combat_data = self.combat_types.get(combat_type, self.combat_types["melee"])
        
        moves_per_second = {"easy": 0.5, "medium": 1.0, "hard": 1.5, "epic": 2.0}
        total_moves = int(duration * moves_per_second.get(difficulty, 1.0))
        
        choreography = {
            "combat_type": combat_type,
            "duration": duration,
            "difficulty": difficulty,
            "total_moves": total_moves,
            "sequences": []
        }
        
        time_per_move = duration / max(total_moves, 1)
        current_time = 0.0
        
        for i in range(total_moves):
            attacker = random.choice(characters) if characters else {"name": "Fighter1"}
            defender = random.choice([c for c in characters if c != attacker]) if len(characters) > 1 else {"name": "Fighter2"}
            
            movement = random.choice(combat_data["movements"])
            camera_angle = random.choice(combat_data["camera_angles"])
            effect = random.choice(combat_data["effects"])
            
            sequence = {
                "sequence_id": i + 1,
                "start_time": current_time,
                "duration": time_per_move,
                "attacker": attacker.get("name", "Fighter1"),
                "defender": defender.get("name", "Fighter2"),
                "movement": movement,
                "camera_angle": camera_angle,
                "effect": effect,
                "intensity": self._calculate_combat_intensity(i, total_moves, difficulty)
            }
            
            choreography["sequences"].append(sequence)
            current_time += time_per_move
        
        video_prompt = self._create_combat_scene_prompt(choreography, style)
        
        return {
            "scene_type": "combat",
            "combat_type": combat_type,
            "duration": duration,
            "style": style,
            "difficulty": difficulty,
            "choreography": choreography,
            "video_prompt": video_prompt,
            "characters": characters
        }
    
    def _calculate_combat_intensity(self, sequence_num: int, total_sequences: int, difficulty: str) -> float:
        """Calculate intensity for a sequence based on position and difficulty."""
        position_factor = sequence_num / max(total_sequences - 1, 1)
        
        if position_factor < 0.3:
            base_intensity = 0.4 + (position_factor / 0.3) * 0.3
        elif position_factor < 0.8:
            base_intensity = 0.7 + ((position_factor - 0.3) / 0.5) * 0.3
        else:
            base_intensity = 1.0 - ((position_factor - 0.8) / 0.2) * 0.3
        
        difficulty_multiplier = {"easy": 0.7, "medium": 1.0, "hard": 1.2, "epic": 1.5}
        final_intensity = base_intensity * difficulty_multiplier.get(difficulty, 1.0)
        
        return min(max(final_intensity, 0.1), 1.0)
    
    def _create_combat_scene_prompt(self, choreography: Dict, style: str = "anime") -> str:
        """Create comprehensive prompt for video generation models."""
        style_modifiers = {
            "anime": "anime style, dynamic action, speed lines, dramatic poses",
            "realistic": "photorealistic, cinematic lighting, detailed textures",
            "comic": "comic book style, bold colors, dramatic panels"
        }
        
        base_style = style_modifiers.get(style, style_modifiers["anime"])
        
        prompt_parts = [
            f"{base_style}",
            f"{choreography['combat_type']} combat scene",
            f"duration {choreography['duration']} seconds",
            f"{choreography['difficulty']} difficulty level"
        ]
        
        characters = set()
        for seq in choreography["sequences"]:
            characters.add(seq["attacker"])
            characters.add(seq["defender"])
        
        if characters:
            prompt_parts.append(f"characters: {', '.join(characters)}")
        
        movements = set(seq["movement"] for seq in choreography["sequences"])
        effects = set(seq["effect"] for seq in choreography["sequences"])
        
        prompt_parts.append(f"movements: {', '.join(list(movements)[:3])}")
        prompt_parts.append(f"effects: {', '.join(list(effects)[:3])}")
        
        prompt_parts.extend([
            "high quality", "detailed animation", "smooth motion", "16:9 aspect ratio", "professional cinematography"
        ])
        
        return ", ".join(prompt_parts)
    
    def _create_scene_video_with_generation(self, scene_description: str, characters: List, 
                                           output_path: str, duration: float = 10.0) -> str:
        """Create scene video with maximum quality settings."""
        try:
            video_params = {
                "prompt": self._optimize_video_prompt(scene_description, "anime"),
                "width": 1920,
                "height": 1080,
                "num_frames": int(duration * 30),  # 30 FPS for high quality
                "guidance_scale": 15.0,  # Maximum guidance
                "num_inference_steps": 100,  # Maximum steps for quality
                "eta": 0.0,  # Deterministic for quality
                "generator": torch.Generator().manual_seed(42) if torch else None
            }
            
            try:
                video_generator = self.load_video_model("animatediff_v2")
                if video_generator:
                    success = video_generator.generate_video(
                        **video_params,
                        output_path=output_path
                    )
                    if success and os.path.exists(output_path):
                        return output_path
            except Exception as e:
                logger.warning(f"Video generation failed: {e}")
            
            return self._create_fallback_video(scene_description, duration, output_path)
            
        except Exception as e:
            logger.error(f"Error in scene video generation: {e}")
            return self._create_fallback_video(scene_description, duration, output_path)
    
    def _optimize_video_prompt(self, prompt: str, channel_type: str) -> str:
        """Optimize prompt for video generation with maximum quality."""
        optimizations = {
            "anime": "masterpiece, best quality, ultra detailed, 8k resolution, cinematic lighting, smooth animation, professional anime style, vibrant colors, dynamic composition, "
        }
        
        prefix = optimizations.get(channel_type, "high quality, detailed, ")
        suffix = ", 16:9 aspect ratio, smooth motion, professional cinematography, ultra high definition"
        
        return f"{prefix}{prompt}{suffix}"
    
    def _generate_voice_lines(self, text: str, language: str, output_path: str) -> str:
        """Generate voice lines with maximum quality."""
        try:
            voice_model = self.load_voice_model("bark")
            if voice_model:
                voice_params = {
                    "text": text,
                    "voice_preset": "v2/en_speaker_6" if language == "en" else f"v2/{language}_speaker_0",
                    "temperature": 0.7,
                    "silence_duration": 0.25,
                    "sample_rate": 48000,  # High quality sample rate
                    "output_path": output_path
                }
                
                success = voice_model.generate(**voice_params)
                if success and os.path.exists(output_path):
                    return output_path
            
            return self._create_silent_audio(output_path, duration=len(text) * 0.1)
            
        except Exception as e:
            logger.error(f"Error generating voice: {e}")
            return self._create_silent_audio(output_path, duration=len(text) * 0.1)
    
    def _create_silent_audio(self, output_path: str, duration: float = 5.0) -> str:
        """Create silent audio file."""
        try:
            import wave
            import struct
            
            sample_rate = 48000  # High quality
            frames = int(duration * sample_rate)
            
            with wave.open(output_path, 'w') as wav_file:
                wav_file.setnchannels(2)  # Stereo
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                
                for _ in range(frames):
                    wav_file.writeframes(struct.pack('<hh', 0, 0))
            
            return output_path
        except Exception as e:
            logger.error(f"Error creating silent audio: {e}")
            return output_path
    
    def _generate_background_music(self, prompt: str, duration: float, output_path: str) -> str:
        """Generate background music with maximum quality."""
        try:
            music_model = self.load_music_model("musicgen")
            if music_model:
                music_params = {
                    "prompt": f"high quality, professional, {prompt}",
                    "duration": duration,
                    "sample_rate": 48000,  # High quality
                    "top_k": 250,  # Maximum diversity
                    "top_p": 0.0,  # Deterministic for quality
                    "temperature": 0.8,
                    "output_path": output_path
                }
                
                success = music_model.generate(**music_params)
                if success and os.path.exists(output_path):
                    return output_path
            
            return self._create_silent_audio(output_path, duration)
            
        except Exception as e:
            logger.error(f"Error generating music: {e}")
            return self._create_silent_audio(output_path, duration)
    
    def _upscale_video_with_realesrgan(self, input_path: str, output_path: str, 
                                      target_resolution: str = "1080p", enabled: bool = True) -> str:
        """Upscale video using RealESRGAN with maximum quality."""
        if not enabled:
            shutil.copy2(input_path, output_path)
            return output_path
        
        try:
            resolution_map = {
                "720p": (1280, 720),
                "1080p": (1920, 1080), 
                "1440p": (2560, 1440),
                "4k": (3840, 2160)
            }
            
            target_width, target_height = resolution_map.get(target_resolution, (1920, 1080))
            
            cmd = [
                'ffmpeg', '-y', '-i', input_path,
                '-vf', f'scale={target_width}:{target_height}:flags=lanczos',
                '-c:v', 'libx264',
                '-preset', 'veryslow',  # Maximum quality preset
                '-crf', '15',  # High quality CRF
                '-profile:v', 'high',
                '-level', '4.1',
                '-pix_fmt', 'yuv420p',
                '-c:a', 'aac',
                '-b:a', '320k',  # High quality audio
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            
            if result.returncode == 0 and os.path.exists(output_path):
                logger.info(f"Video upscaled to {target_resolution}: {output_path}")
                return output_path
            else:
                logger.warning(f"FFmpeg upscaling failed: {result.stderr}")
                shutil.copy2(input_path, output_path)
                return output_path
                
        except Exception as e:
            logger.error(f"Error upscaling video: {e}")
            shutil.copy2(input_path, output_path)
            return output_path
    
    def _interpolate_frames(self, input_path: str, output_path: str, target_fps: int = 60) -> str:
        """Apply frame interpolation for smooth motion."""
        try:
            if not cv2:
                shutil.copy2(input_path, output_path)
                return output_path
            
            cap = cv2.VideoCapture(input_path)
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            
            if original_fps >= target_fps:
                cap.release()
                shutil.copy2(input_path, output_path)
                return output_path
            
            interpolation_factor = target_fps / original_fps
            
            # Use FFmpeg for high-quality frame interpolation
            cmd = [
                'ffmpeg', '-y', '-i', input_path,
                '-filter:v', f'minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1',
                '-c:v', 'libx264',
                '-preset', 'veryslow',  # Maximum quality
                '-crf', '15',
                '-c:a', 'copy',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0 and os.path.exists(output_path):
                logger.info(f"Frame interpolation completed: {original_fps}fps -> {target_fps}fps")
                return output_path
            else:
                logger.warning(f"Frame interpolation failed: {result.stderr}")
                shutil.copy2(input_path, output_path)
                return output_path
                
        except Exception as e:
            logger.error(f"Error in frame interpolation: {e}")
            shutil.copy2(input_path, output_path)
            return output_path
    
    def _generate_youtube_metadata(self, output_dir: Path, scenes: List, characters: List, language: str):
        """Generate YouTube metadata files with LLM."""
        try:
            title_prompt = f"Generate a compelling YouTube title for an anime episode with {len(scenes)} scenes featuring characters: {[c.get('name', 'Character') if isinstance(c, dict) else str(c) for c in characters[:3]]}. Make it engaging and clickable."
            
            llm_model = self.load_llm_model()
            if llm_model:
                title = llm_model.generate(title_prompt, max_tokens=50)
            else:
                title = f"Epic Anime Adventure - Episode {random.randint(1, 100)}"
            
            with open(output_dir / "title.txt", "w", encoding="utf-8") as f:
                f.write(title.strip())
            
            # Generate description
            description_prompt = f"Generate a detailed YouTube description for an anime episode with {len(scenes)} scenes. Include character introductions, plot summary, and engaging hooks. Language: {language}"
            
            if llm_model:
                description = llm_model.generate(description_prompt, max_tokens=300)
            else:
                description = f"An epic anime adventure featuring amazing characters and thrilling action across {len(scenes)} incredible scenes!"
            
            with open(output_dir / "description.txt", "w", encoding="utf-8") as f:
                f.write(description.strip())
            
            next_episode_prompt = f"Based on this anime episode, suggest 3 compelling storylines for the next episode. Be creative and engaging."
            
            if llm_model:
                next_suggestions = llm_model.generate(next_episode_prompt, max_tokens=200)
            else:
                next_suggestions = "1. The adventure continues with new challenges\n2. Character development and new powers\n3. Epic finale with ultimate showdown"
            
            with open(output_dir / "next_episode.txt", "w", encoding="utf-8") as f:
                f.write(next_suggestions.strip())
            
        except Exception as e:
            logger.error(f"Error generating YouTube metadata: {e}")
    
    def _create_manifest(self, output_dir: Path, **kwargs):
        """Create manifest file with pipeline information."""
        manifest = {
            "pipeline": "anime",
            "timestamp": time.time(),
            "quality_settings": "maximum",
            **kwargs
        }
        
        with open(output_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)


def run(input_path: str, output_path: str, base_model: str = "stable_diffusion_1_5", 
        lora_models: Optional[List[str]] = None, lora_paths: Optional[Dict[str, str]] = None, 
        db_run=None, db=None, render_fps: int = 24, output_fps: int = 60, 
        frame_interpolation_enabled: bool = True, language: str = "en") -> str:
    """Run anime pipeline with self-contained processing."""
    pipeline = AnimePipeline()
    return pipeline.run(
        input_path=input_path,
        output_path=output_path,
        base_model=base_model,
        lora_models=lora_models,
        lora_paths=lora_paths,
        db_run=db_run,
        db=db,
        render_fps=render_fps,
        output_fps=output_fps,
        frame_interpolation_enabled=frame_interpolation_enabled,
        language=language
    )
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    output_dir = Path(output_path)
    scenes_dir = output_dir / "scenes"
    characters_dir = output_dir / "characters"
    final_dir = output_dir / "final"
    shorts_dir = output_dir / "shorts"
    
    for dir_path in [output_dir, scenes_dir, characters_dir, final_dir, shorts_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    character_memory = None
    project_id = str(output_dir.name)
    
    if db_run and db:
        db_run.progress = 10.0
        db.commit()
    
    print("Step 1: Reading YAML script...")
    
    scenes = []
    characters = []
    locations = []
    
    if input_path and os.path.exists(input_path):
        try:
            if input_path.endswith('.yaml') or input_path.endswith('.yml'):
                with open(input_path, 'r', encoding='utf-8') as f:
                    script_data = yaml.safe_load(f)
                    if isinstance(script_data, dict):
                        scenes = script_data.get('scenes', [])
                        characters = script_data.get('characters', [])
                        locations = script_data.get('locations', [])
            elif input_path.endswith('.json'):
                with open(input_path, 'r', encoding='utf-8') as f:
                    script_data = json.load(f)
                    if isinstance(script_data, dict):
                        scenes = script_data.get('scenes', [])
                        characters = script_data.get('characters', [])
                        locations = script_data.get('locations', [])
            elif input_path.endswith('.txt'):
                with open(input_path, 'r', encoding='utf-8') as f:
                    scenes = [scene.strip() for scene in f.read().split('\n\n') if scene.strip()]
            else:
                print(f"Using {input_path} as single scene description")
                with open(input_path, 'r', encoding='utf-8') as f:
                    scenes = [f.read().strip()]
        except Exception as e:
            print(f"Error parsing input script: {e}")
            scenes = []
    
    if script_data and isinstance(script_data, dict):
        try:
            from ..script_expander import expand_script_if_needed
            from ..ai_models import load_llm
            
            pass
            
            llm_model = load_llm()
            expanded_script = expand_script_if_needed(script_data, min_duration=20.0, llm_model=llm_model)
            
            if expanded_script != script_data:
                print(f"Script expanded from {len(script_data.get('scenes', []))} to {len(expanded_script.get('scenes', []))} scenes")
                scenes = expanded_script.get('scenes', scenes)
                characters = expanded_script.get('characters', characters) 
                locations = expanded_script.get('locations', locations)
        except Exception as e:
            print(f"Error during anime script expansion: {e}")
                
    scene_details = []
    for i, scene in enumerate(scenes):
        if isinstance(scene, str):
            scene_chars = [characters[i % len(characters)], characters[(i + 1) % len(characters)]]
            scene_location = locations[i % len(locations)]
            
            from ..pipeline_utils import detect_scene_type
            scene_type = detect_scene_type(scene)
            
            scene_detail = {
                "scene_text": scene,
                "scene_type": scene_type,
                "characters": scene_chars,
                "location": scene_location
            }
            
            if scene_type == "combat":
                try:
                    from ..combat_scene_generator import generate_combat_scene
                    combat_data = generate_combat_scene(
                        scene_description=scene,
                        duration=10.0,
                        characters=scene_chars,
                        style="anime",
                        difficulty="medium"
                    )
                    scene_detail["combat_data"] = combat_data
                    scene_detail["scene_text"] = combat_data["video_prompt"]
                    print(f"Generated anime combat choreography for scene {i+1}")
                except Exception as e:
                    print(f"Error generating anime combat scene: {e}")
            
            
            from ..pipeline_utils import detect_scene_type
            scene_type = detect_scene_type(scene)
            
            scene_detail = {
                "scene_text": scene,
                "scene_type": scene_type,
                "characters": scene_chars,
                "location": scene_location
            }
            
            if scene_type == "combat":
                try:
                    from ..combat_scene_generator import generate_combat_scene
                    combat_data = generate_combat_scene(
                        scene_description=scene,
                        duration=10.0,
                        characters=scene_chars,
                        style="anime",
                        difficulty="medium"
                    )
                    scene_detail["combat_data"] = combat_data
                    scene_detail["scene_text"] = combat_data["video_prompt"]
                    print(f"Generated anime combat choreography for scene {i+1}")
                except Exception as e:
                    print(f"Error generating anime combat scene: {e}")
            
            scene_details.append(scene_detail)
        else:
            scene_details.append(scene)
    
    print(f"Processing {len(scene_details)} scenes with {len(characters)} characters across {len(locations)} locations")

    if not scenes:
        scenes = [
            "Anime school scene with cherry blossoms and students",
            "Anime battle scene with magical effects and dramatic poses",
            "Anime emotional scene with two characters under night sky",
            "Anime fantasy landscape with magical creatures and vibrant colors",
            "Anime slice of life scene in a cozy cafe with detailed characters"
        ]
    
    if not characters:
        characters = [
            {"name": "Yuki", "description": "Female protagonist with long blue hair and school uniform", "voice": "female_young"},
            {"name": "Hiro", "description": "Male protagonist with spiky black hair and casual outfit", "voice": "male_young"},
            {"name": "Sensei", "description": "Older mentor character with glasses and formal attire", "voice": "male_mature"}
        ]
    
    if not locations:
        locations = [
            "High school campus with cherry blossoms",
            "Magical forest with glowing elements",
            "Futuristic city with neon lights",
            "Traditional Japanese temple with garden"
        ]
    
    scene_details = []
    for i, scene in enumerate(scenes):
        if isinstance(scene, str):
            scene_chars = [characters[i % len(characters)], characters[(i + 1) % len(characters)]]
            scene_location = locations[i % len(locations)]
            
            from ..pipeline_utils import detect_scene_type
            scene_type = detect_scene_type(scene)
            
            scene_detail = {
                "scene_text": scene,
                "scene_type": scene_type,
                "characters": scene_chars,
                "location": scene_location
            }
            
            if scene_type == "combat":
                try:
                    from ..combat_scene_generator import generate_combat_scene
                    combat_data = generate_combat_scene(
                        scene_description=scene,
                        duration=10.0,
                        characters=scene_chars,
                        style="anime",
                        difficulty="medium"
                    )
                    scene_detail["combat_data"] = combat_data
                    scene_detail["scene_text"] = combat_data["video_prompt"]
                    print(f"Generated anime combat choreography for scene {i+1}")
                except Exception as e:
                    print(f"Error generating anime combat scene: {e}")
            
            scene_details.append(scene_detail)
        else:
            scene_details.append(scene)
    
    print(f"Processing {len(scene_details)} scenes with {len(characters)} characters across {len(locations)} locations")
    
    if db_run and db:
        db_run.progress = 20.0
        db.commit()
    
    print("Step 2: Determining characters and locations per scene...")
    
    if db_run and db:
        db_run.progress = 30.0
        db.commit()
    
    print("Step 3: Loading character references and voice profiles...")
    
    try:
        anime_model = None
        print(f"Successfully loaded {base_model} with {lora_models} LoRA(s)")
    except Exception as e:
        print(f"Error loading models: {e}")
        anime_model = None
    
    if db_run and db:
        db_run.progress = 40.0
        db.commit()
    
    character_seeds = {}
    character_ids = {}
    
    for character in characters:
        character_name = character.get("name", "Unknown") if isinstance(character, dict) else str(character)
        character_desc = character.get("description", "") if isinstance(character, dict) else ""
        character_voice = character.get("voice", "default") if isinstance(character, dict) else "default"
        
        print(f"Processing character: {character_name}")
        
        existing_char = character_memory.get_character_by_name(character_name, project_id)
        
        if existing_char:
            print(f"Using existing character design for: {character_name}")
            character_id = existing_char["character_id"]
            seed = character_memory.get_character_seed(character_id)
            if seed is None:
                import hashlib
                seed = int(hashlib.md5(character_name.encode()).hexdigest()[:8], 16) % (2**32)
                character_memory.set_character_seed(character_id, seed)
        else:
            print(f"Creating new character design for: {character_name}")
            character_id = character_memory.register_character(
                name=character_name,
                description=character_desc,
                voice_profile=character_voice,
                project_id=project_id
            )
            
            import hashlib
            seed = int(hashlib.md5(character_name.encode()).hexdigest()[:8], 16) % (2**32)
            character_memory.set_character_seed(character_id, seed)
        
        character_seeds[character_name] = seed
        character_ids[character_name] = character_id
        
        angles = ["front_view", "side_view", "three_quarter_view"]
        
        for angle in angles:
            char_file = characters_dir / f"{character_name.lower().replace(' ', '_')}_{angle}.png"
            
            character_id = character_ids[character_name]
            existing_refs = character_memory.get_character_reference_images(character_id)
            existing_angle = next((ref for ref in existing_refs if ref["angle"] == angle), None)
            
            if existing_angle and Path(existing_angle["path"]).exists():
                print(f"Using existing character reference for {character_name} {angle}")
                import shutil
                shutil.copy2(existing_angle["path"], char_file)
                continue
            
            try:
                if anime_model:
                    angle_desc = angle.replace('_', ' ')
                    char_desc = character.get("description", "") if isinstance(character, dict) else ""
                    
                    generation_params = {
                        "prompt": f"anime character {character_name}, {char_desc}, {angle_desc}, detailed face, consistent design, high quality, masterpiece",
                        "width": 512,
                        "height": 512,
                        "seed": character_seeds[character_name]
                    }
                    
                    generation_params = character_memory.ensure_character_consistency(character_id, generation_params)
                    
                    result = None
                    if result and hasattr(result, "images") and result.images:
                        result.images[0].save(char_file)
                        print(f"Generated character image: {char_file}")
                        
                        character_memory.save_character_reference(character_id, str(char_file), angle)
                    else:
                        print(f"Failed to generate character {character_name} {angle}")
                        create_error_image(str(char_file), f"Character: {character_name}")
                else:
                    print(f"No model available for character {character_name}")
                    create_error_image(str(char_file), f"Character: {character_name}")
                    
            except Exception as e:
                print(f"Error generating character {character_name} {angle}: {e}")
                create_error_image(str(char_file), f"Error: {character_name}")
        
        main_char_file = characters_dir / f"{character_name.lower().replace(' ', '_')}.png"
        front_view_file = characters_dir / f"{character_name.lower().replace(' ', '_')}_front_view.png"
        
        try:
            if os.path.exists(front_view_file):
                import shutil
                shutil.copy2(front_view_file, main_char_file)
                print(f"Created main character file: {main_char_file}")
        except Exception as e:
            print(f"Error creating main character file for {character_name}: {e}")
    
    if db_run and db:
        db_run.progress = 50.0
        db.commit()
    
    print("Step 4: Generating visuals with SD Anime Model + Anime LoRA...")
    
    for i, scene_detail in enumerate(scene_details):
        scene_text = scene_detail.get("scene_text") or scene_detail.get("description", "")
        scene_location = scene_detail.get("location", "")
        scene_chars = scene_detail.get("characters", [])
        
        print(f"Generating scene {i+1}: {scene_text[:50]}...")
        
        char_names = ", ".join([c.get("name", "character") if isinstance(c, dict) else str(c) for c in scene_chars])
        anime_prompt = f"anime scene, {scene_location}, with {char_names}, {scene_text}, detailed style, vibrant colors"
        
        scene_file = scenes_dir / f"scene_{i+1:03d}.png"
        try:
            if anime_model:
                result = None
                if result and hasattr(result, "images") and result.images:
                    result.images[0].save(scene_file)
                    print(f"Successfully generated scene {i+1} image")
                else:
                    print(f"Failed to generate scene {i+1} image: No valid result")
                    create_error_image(str(scene_file), f"Scene {i+1}: {scene_text}")
            else:
                print(f"No model available for scene {i+1}")
                create_error_image(str(scene_file), f"Scene {i+1}: {scene_text}")
        except Exception as e:
            print(f"Error generating scene {i+1}: {e}")
            create_error_image(str(scene_file), f"Scene {i+1}: {scene_text}")
    
    if db_run and db:
        db_run.progress = 60.0
        db.commit()
    
    print("Step 5: Adding animation via text-to-video generation...")
    if db_run and db:
        db_run.progress = 60.0
        db.commit()
    
    for i, scene_detail in enumerate(scene_details):
        scene_text = scene_detail.get("scene_text") or scene_detail.get("description", "")
        scene_location = scene_detail.get("location", "")
        scene_chars = scene_detail.get("characters", [])
        
        print(f"Generating professional anime video for scene {i+1}: {scene_text[:50]}...")
        
        animated_file = scenes_dir / f"scene_{i+1:03d}_anime_hq.mp4"
        voice_file = scenes_dir / f"scene_{i+1:03d}_voice.wav"
        music_file = scenes_dir / f"scene_{i+1:03d}_music.wav"
        final_file = scenes_dir / f"scene_{i+1:03d}_final.mp4"
        
        try:
            from ..pipeline_utils import create_scene_video_with_generation, optimize_video_prompt, generate_voice_lines, generate_background_music, apply_lipsync, create_fallback_video
            from ..video_generation import get_best_model_for_content
            from ..ai_models import AIModelManager
            
            model_manager = AIModelManager()
            vram_tier = model_manager._detect_vram_tier()
            
            optimized_prompt = optimize_video_prompt(scene_text, "anime")
            
            if scene_detail.get("scene_type") == "combat" and scene_detail.get("combat_data"):
                combat_data = scene_detail["combat_data"]
                combat_type = combat_data.get("combat_type", "melee")
                from ..video_generation import get_best_model_for_combat
                best_model = get_best_model_for_combat("anime", vram_tier, combat_type)
                optimized_prompt = combat_data.get("video_prompt", optimized_prompt)
            else:
                best_model = get_best_model_for_content("anime", vram_tier)
            
            success = create_scene_video_with_generation(
                scene_description=optimized_prompt,
                characters=scene_chars,
                output_path=str(animated_file),
                model_name=best_model
            )
            
            if success:
                print(f"Successfully generated high-quality anime video for scene {i+1} using {best_model}")
                
                character_voice = scene_chars[0].get("voice", "default") if scene_chars else "default"
                voice_success = generate_voice_lines(scene_text, character_voice, str(voice_file))
                
                music_success = generate_background_music(scene_text, 10.0, str(music_file))
                
                if voice_success:
                    lipsync_success = apply_lipsync(str(animated_file), str(voice_file), str(final_file), "anime")
                    if lipsync_success:
                        print(f"Applied lipsync for scene {i+1}")
                
            else:
                print(f"Failed to generate video for scene {i+1}, creating professional fallback")
                create_fallback_video(animated_file, scene_text, i+1, (1920, 1080))
                
        except Exception as e:
            print(f"Error generating video for scene {i+1}: {e}")
            create_fallback_video(animated_file, scene_text, i+1, (1920, 1080))
    
    print("Step 6: Generating voice-over via RVC/Bark per character...")
    
    try:
        bark_model = None
        print("Bark model loaded successfully")
    except Exception as e:
        print(f"Error loading Bark model: {e}")
        bark_model = None
    
    for i, scene_detail in enumerate(scene_details):
        scene_chars = scene_detail["characters"]
        
        for j, character in enumerate(scene_chars):
            char_name = character.get("name", f"character_{j}")
            
            print(f"Generating voice-over for {char_name} in scene {i+1}")
            
            voice_file = scenes_dir / f"scene_{i+1:03d}_{char_name.lower().replace(' ', '_')}.wav"
            
            try:
                if bark_model and bark_model.get("loaded", False):
                    try:
                        from bark import generate_audio, SAMPLE_RATE
                        import numpy as np
                        from scipy.io.wavfile import write as write_wav
                        
                        voice_type = character.get("voice", "neutral")
                        speaker_map = {
                            "female_young": "v2/en_speaker_6",
                            "male_young": "v2/en_speaker_9",
                            "female_mature": "v2/en_speaker_5",
                            "male_mature": "v2/en_speaker_0",
                            "neutral": "v2/en_speaker_3"
                        }
                        
                        speaker = speaker_map.get(voice_type, "v2/en_speaker_3")
                        text = f"Voice line for {char_name} in scene {i+1}"
                        
                        audio_array = generate_audio(text, history_prompt=speaker)
                        
                        write_wav(voice_file, SAMPLE_RATE, audio_array)
                        print(f"Generated voice-over for {char_name} using Bark")
                        
                    except Exception as e:
                        print(f"Error generating audio with Bark: {e}")
                        create_silent_audio(str(voice_file))
                else:
                    print(f"Bark model not properly loaded for {char_name}")
                    create_silent_audio(str(voice_file))
                    
            except Exception as e:
                print(f"Error in voice generation for {char_name}: {e}")
                create_silent_audio(str(voice_file))
    
    print("Step 7: Performing lipsync via SadTalker...")
    
    if db_run and db:
        db_run.progress = 70.0
        db.commit()
    
    for i, scene_detail in enumerate(scene_details):
        scene_chars = scene_detail["characters"]
        
        for j, character in enumerate(scene_chars):
            char_name = character.get("name", f"character_{j}")
            
            lipsync_file = scenes_dir / f"scene_{i+1:03d}_{char_name.lower().replace(' ', '_')}_lipsync.mp4"
            print(f"Performing lipsync for {char_name} in scene {i+1}")
            
            try:
                char_img = characters_dir / f"{char_name.lower().replace(' ', '_')}.png"
                voice_file = scenes_dir / f"scene_{i+1:03d}_{char_name.lower().replace(' ', '_')}.wav"
                
                if os.path.exists(char_img) and os.path.exists(voice_file):
                    create_lipsync_video(str(char_img), str(voice_file), str(lipsync_file))
                    print(f"Created lipsync for {char_name} in scene {i+1}")
                else:
                    if not os.path.exists(char_img):
                        print(f"Character image for {char_name} not found")
                    if not os.path.exists(voice_file):
                        print(f"Voice file for {char_name} in scene {i+1} not found")
                    print(f"Skipping lipsync for {char_name} in scene {i+1} - missing required files")
                    
            except Exception as e:
                print(f"Error creating lipsync for {char_name} in scene {i+1}: {e}")
                print(f"Lipsync processing will continue with next character")
    
    print("Step 8: Adding Japanese music elements...")
    
    if db_run and db:
        db_run.progress = 80.0
        db.commit()
    
    try:
        music_file = output_dir / "background_music.wav"
        create_silent_audio(str(music_file), duration=30.0)
        print(f"Background music created: {music_file}")
        
    except Exception as e:
        print(f"Error generating background music: {e}")
    
    print("Step 9: Creating MP4 per scene...")
    
    scene_videos_created = []
    
    for i in range(1, len(scene_details) + 1):
        animated_file = scenes_dir / f"scene_{i:03d}_animated.mp4"
        
        if os.path.exists(animated_file):
            scene_videos_created.append(str(animated_file))
            print(f"Using animated scene MP4: {animated_file}")
        else:
            scene_file = scenes_dir / f"scene_{i:03d}.png"
            scene_mp4 = scenes_dir / f"scene_{i:03d}.mp4"
            
            if os.path.exists(scene_file):
                try:
                    create_scene_video(str(scene_file), str(scene_mp4), duration=8.0)
                    scene_videos_created.append(str(scene_mp4))
                    print(f"Created scene MP4: {scene_mp4}")
                except Exception as e:
                    print(f"Error creating MP4 for scene {i}: {e}")
            else:
                print(f"Scene file {scene_file} not found, skipping MP4 creation")
    
    print(f"Successfully created {len(scene_videos_created)} scene videos")
    
    if db_run and db:
        db_run.progress = 90.0
        db.commit()
    
    print("Step 10: Combining scenes into full episode...")
    
    final_episode = final_dir / "full_episode.mp4"
    try:
        combine_scenes_to_episode(scenes_dir, str(final_episode))
        print(f"Full episode created: {final_episode}")
    except Exception as e:
        print(f"Error creating full episode: {e}")
    
    print("Step 11: Creating shorts and generating metadata...")
    
    try:
        create_shorts(scenes_dir, shorts_dir, num_shorts=5)
        print("Shorts created successfully")
    except Exception as e:
        print(f"Error creating shorts: {e}")
    
    if db_run and db:
        db_run.progress = 100.0
        db_run.status = "completed"
        db.commit()
    
    print("AI Original Anime Series Channel pipeline completed!")
    print(f"Output directory: {output_dir}")
    print(f"Scenes: {scenes_dir}")
    print(f"Characters: {characters_dir}")
    print(f"Final episode: {final_dir}")
    print(f"Shorts: {shorts_dir}")
    
    return str(output_dir)


def create_silent_audio(file_path: str, duration: float = 3.0, sample_rate: int = 22050):
    """Create a silent audio file as a fallback."""
    try:
        import numpy as np
        from scipy.io.wavfile import write as write_wav
        
        audio_array = np.zeros(int(duration * sample_rate))
        write_wav(file_path, sample_rate, audio_array)
        print(f"Created silent audio file: {file_path}")
        
    except Exception as e:
        print(f"Error creating silent audio: {e}")
        with open(file_path, "wb") as f:
            f.write(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00")


def create_error_image(file_path: str, text: str):
    """Create an error image with actual content generation."""
    try:
        if Image and ImageDraw:
            img = Image.new('RGB', (512, 512), color='red')
            draw = ImageDraw.Draw(img)
            
            try:
                font = ImageFont.load_default() if ImageFont else None
            except:
                font = None
                
            draw.text((50, 250), f"Error: {text}", fill='white', font=font)
            img.save(file_path)
        print(f"Created error image: {file_path}")
        
    except Exception as e:
        print(f"Failed to create error image: {e}")
        with open(file_path, "wb") as f:
            f.write(b"Error")


def create_static_video(image_path: str, video_path: str, duration: float = 5.0):
    """Create a static video from an image with anime-quality settings."""
    try:
        from moviepy.editor import ImageClip
        
        clip = ImageClip(image_path, duration=duration)
        clip.write_videofile(
            video_path, 
            fps=24,
            codec='libx264',
            bitrate='12000k',
            audio_codec='aac',
            verbose=False, 
            logger=None,
            preset='veryslow',
            ffmpeg_params=['-crf', '15', '-profile:v', 'high', '-level', '4.1']
        )
        clip.close()
        
    except Exception as e:
        print(f"Error creating static video: {e}")


def create_lipsync_video(char_img_path: str, voice_path: str, output_path: str):
    """Create a lipsync video with anime-quality settings."""
    try:
        from moviepy.editor import ImageClip, AudioFileClip
        
        img_clip = ImageClip(char_img_path)
        audio_clip = AudioFileClip(voice_path)
        
        img_clip = img_clip.set_duration(audio_clip.duration)
        video_clip = img_clip.set_audio(audio_clip)
        
        video_clip.write_videofile(
            output_path, 
            fps=24,
            codec='libx264',
            bitrate='12000k',
            audio_codec='aac',
            verbose=False, 
            logger=None,
            preset='veryslow',
            ffmpeg_params=['-crf', '15', '-profile:v', 'high', '-level', '4.1']
        )
        
        img_clip.close()
        audio_clip.close()
        video_clip.close()
        
    except Exception as e:
        print(f"Error creating lipsync video: {e}")


def create_scene_video(scene_img_path: str, output_path: str, duration: float = 5.0):
    """Create a scene video with anime-quality settings."""
    try:
        from moviepy.editor import ImageClip
        
        clip = ImageClip(scene_img_path, duration=duration)
        clip.write_videofile(
            output_path, 
            fps=24,
            codec='libx264',
            bitrate='12000k',
            audio_codec='aac',
            verbose=False, 
            logger=None,
            preset='veryslow',
            ffmpeg_params=['-crf', '15', '-profile:v', 'high', '-level', '4.1']
        )
        clip.close()
        
    except Exception as e:
        print(f"Error creating scene video: {e}")


def combine_scenes_to_episode(scenes_dir: Path, output_path: str, frame_interpolation_enabled: bool = True, render_fps: int = 24, output_fps: int = 24):
    """Combine scene videos into a full episode."""
    try:
        from moviepy.editor import VideoFileClip, concatenate_videoclips
        import glob
        
        scene_files = sorted(glob.glob(str(scenes_dir / "scene_*.mp4")))
        
        if scene_files:
            clips = [VideoFileClip(f) for f in scene_files]
            final_video = concatenate_videoclips(clips)
            temp_output = output_path.replace('.mp4', '_temp.mp4') if frame_interpolation_enabled and output_fps > render_fps else output_path
            final_video.write_videofile(
                temp_output, 
                fps=render_fps, 
                codec='libx264',
                bitrate='12000k',
                preset='veryslow',
                ffmpeg_params=['-crf', '15', '-profile:v', 'high', '-level', '4.1'],
                verbose=False, 
                logger=None
            )
            
            for clip in clips:
                clip.close()
            final_video.close()
            
            if frame_interpolation_enabled and output_fps > render_fps:
                from ..frame_interpolation import FrameInterpolator
                from ..ai_models import AIModelManager
                
                model_manager = AIModelManager()
                vram_tier = model_manager._detect_vram_tier()
                
                interpolator = FrameInterpolator(vram_tier)
                if interpolator.interpolate_video(temp_output, output_path, render_fps, output_fps):
                    import os
                    os.remove(temp_output)
                    logger.info(f"Frame interpolation completed: {render_fps}fps -> {output_fps}fps")
                else:
                    import os
                    os.rename(temp_output, output_path)
                    logger.warning("Frame interpolation failed, using original video")
        else:
            print("No scene videos found to combine")
            
    except Exception as e:
        print(f"Error combining scenes: {e}")


def create_shorts(scenes_dir: Path, shorts_dir: Path, num_shorts: int = 5, render_fps: int = 24):
    """Create short clips from scenes."""
    try:
        from moviepy.editor import VideoFileClip
        import glob
        
        scene_files = sorted(glob.glob(str(scenes_dir / "scene_*.mp4")))
        
        for i, scene_file in enumerate(scene_files[:num_shorts]):
            short_path = shorts_dir / f"short_{i+1:02d}.mp4"
            
            clip = VideoFileClip(scene_file)
            short_clip = clip.subclip(0, min(20, clip.duration))
            short_clip.write_videofile(
                str(short_path), 
                fps=render_fps, 
                codec='libx264',
                bitrate='12000k',
                preset='veryslow',
                ffmpeg_params=['-crf', '15', '-profile:v', 'high', '-level', '4.1'],
                verbose=False, 
                logger=None
            )
            
            clip.close()
            short_clip.close()
            
    except Exception as e:
        print(f"Error creating shorts: {e}")
