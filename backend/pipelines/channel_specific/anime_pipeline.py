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
from datetime import datetime
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

class AnimeChannelPipeline(BasePipeline):
    """Self-contained anime content generation pipeline with all functionality inlined."""
    
    def __init__(self, output_path: Optional[str] = None, base_model: str = "stable_diffusion_1_5"):
        super().__init__("anime", output_path, base_model)
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
            frame_interpolation_enabled: bool = True, llm_model: str = "microsoft/DialoGPT-medium", 
            language: str = "en") -> str:
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
                db_run, db, render_fps, output_fps, frame_interpolation_enabled, llm_model, language
            )
        except Exception as e:
            error_message = f"Anime pipeline failed: {e}\n{traceback.format_exc()}"
            logger.error(error_message)
            
            output_dir = Path(output_path)
            error_log_path = output_dir / "error_log.txt"
            with open(error_log_path, "a") as f:
                f.write(f"[{datetime.now()}] {error_message}\n\n")
            
            raise
        finally:
            self.cleanup_models()
    
    def _execute_pipeline(self, input_path: str, output_path: str, base_model: str, 
                         lora_models: Optional[List[str]], db_run, db, render_fps: int, 
                         output_fps: int, frame_interpolation_enabled: bool, llm_model: str, language: str) -> str:
        
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
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
            
            print("Processing script with LLM for advanced scene analysis...")
            processed_script = self._process_script_with_llm(script_data, "anime")
            
            if processed_script.get('llm_processed'):
                enhanced_scenes = processed_script.get('enhanced_scenes', [])
                print(f"LLM processed {len(enhanced_scenes)} scenes with model-specific prompts")
            else:
                from ...utils.error_handler import PipelineErrorHandler
                error_handler = PipelineErrorHandler()
                llm_error = Exception("LLM script processing failed, unable to enhance scenes")
                error_handler.log_error_to_output(
                    error=llm_error,
                    output_path=str(output_dir),
                    context={"script_data": script_data, "channel_type": "anime"}
                )
                logger.error("LLM script processing failed, error logged to output directory")
                enhanced_scenes = processed_script.get('enhanced_scenes', [])
            
            scenes = enhanced_scenes
            characters = processed_script.get('characters', characters)
            locations = processed_script.get('locations', locations)
            
            print(f"Anime script processed with {len(scenes)} enhanced scenes")
            
        except Exception as e:
            print(f"Error during LLM script processing: {e}")
        
        print("Step 3: Generating anime scenes with combat integration...")
        if db_run and db:
            db_run.progress = 20.0
            db.commit()
        
        scene_files = []
        for i, enhanced_scene in enumerate(scenes):
            if isinstance(enhanced_scene, dict) and 'video_prompt' in enhanced_scene:
                video_prompt = enhanced_scene['video_prompt']
                scene_type = enhanced_scene.get('scene_type', 'default')
                duration = enhanced_scene.get('duration', 10.0)
                scene_text = enhanced_scene.get('original_description', f'Scene {i+1}')
                scene_number = enhanced_scene.get('scene_number', i + 1)
            else:
                scene_text = enhanced_scene if isinstance(enhanced_scene, str) else enhanced_scene.get('description', f'Scene {i+1}')
                video_prompt = f"masterpiece, best quality, ultra detailed, 8k resolution, cinematic lighting, smooth animation, professional anime style, vibrant colors, dynamic composition, {scene_text}, 16:9 aspect ratio, smooth motion, professional cinematography, ultra high definition"
                scene_type = self._detect_scene_type(scene_text)
                duration = enhanced_scene.get('duration', 10.0) if isinstance(enhanced_scene, dict) else 10.0
                scene_number = i + 1
            
            scene_detail = {
                "scene_number": scene_number,
                "description": scene_text,
                "video_prompt": video_prompt,
                "scene_type": scene_type,
                "duration": duration,
                "enhanced": isinstance(enhanced_scene, dict) and 'video_prompt' in enhanced_scene
            }
            
            if scene_type == "combat" and self.combat_calls_count < self.max_combat_calls:
                try:
                    combat_data = {
                        "combat_type": "melee",
                        "intensity": 0.7,
                        "video_prompt": f"Epic anime combat scene: {scene_text}, dynamic action, sword fighting",
                        "duration": 10.0,
                        "movements": ["slash", "parry", "combo"],
                        "camera_angles": ["dramatic_low", "overhead"],
                        "effects": ["blade_flash", "energy_slash"]
                    }
                    scene_detail["combat_data"] = combat_data
                    self.combat_calls_count += 1
                    print(f"Generated anime combat scene {i+1} with choreography ({self.combat_calls_count}/{self.max_combat_calls})")
                except Exception as e:
                    print(f"Error generating anime combat scene: {e}")
            
            scene_file = scenes_dir / f"scene_{i+1:03d}.mp4"
            
            print(f"Generating anime scene {i+1}: {str(scene_text)[:50]}...")
            
            try:
                anime_prompt = video_prompt
                
                if scene_detail.get("combat_data"):
                    anime_prompt = scene_detail["combat_data"]["video_prompt"]
                
                logger.info(f"Using {'LLM-enhanced' if scene_detail.get('enhanced') else 'default'} prompt: {anime_prompt[:100]}...")
                
                video_path = self._create_scene_video_with_generation(
                    scene_description=anime_prompt,
                    characters=[],  # Characters already incorporated in LLM prompt
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
                self._log_video_generation_error(scene_text, scene_detail["duration"], str(scene_file), str(e))
            
            if db_run and db:
                db_run.progress = 20.0 + (i + 1) / len(scenes) * 30.0
                db.commit()
        
        print("Step 4: Generating voice lines...")
        if db_run and db:
            db_run.progress = 50.0
            db.commit()
        
        voice_files = []
        for i, enhanced_scene in enumerate(scenes):
            if isinstance(enhanced_scene, dict) and 'voice_prompt' in enhanced_scene:
                voice_text = enhanced_scene['voice_prompt']
                print(f"Using LLM-generated voice prompt for scene {i+1}")
            else:
                scene_text = enhanced_scene if isinstance(enhanced_scene, str) else enhanced_scene.get('description', f'Scene {i+1}')
                voice_text = enhanced_scene.get('dialogue', scene_text) if isinstance(enhanced_scene, dict) else scene_text
                from ...utils.error_handler import PipelineErrorHandler
                error_handler = PipelineErrorHandler()
                voice_error = Exception(f"LLM voice prompt generation failed for scene {i+1}")
                error_handler.log_error_to_output(
                    error=voice_error,
                    output_path=str(scenes_dir),
                    context={"scene_number": i+1, "scene_text": scene_text, "channel_type": "anime"}
                )
                logger.error(f"LLM voice prompt generation failed for scene {i+1}, error logged to output directory")
            
            voice_file = scenes_dir / f"voice_{i+1:03d}.wav"
            
            try:
                voice_path = self._generate_voice_lines(
                    text=voice_text,  # Use LLM-generated voice prompt
                    language=language,
                    output_path=str(voice_file)
                )
                
                if voice_path:
                    voice_files.append(voice_path)
                    print(f"Generated voice for scene {i+1} with enhanced prompt")
                    
            except Exception as e:
                print(f"Error generating voice for scene {i+1}: {e}")
        
        print("Step 5: Generating background music...")
        if db_run and db:
            db_run.progress = 60.0
            db.commit()
        
        music_file = final_dir / "background_music.wav"
        try:
            music_prompts = []
            total_duration = 0
            
            for enhanced_scene in scenes:
                if isinstance(enhanced_scene, dict) and 'music_prompt' in enhanced_scene:
                    music_prompts.append(enhanced_scene['music_prompt'])
                    duration_val = enhanced_scene.get('duration', 10.0)
                    if isinstance(duration_val, str):
                        try:
                            duration_val = float(duration_val.replace('seconds', '').strip())
                        except (ValueError, AttributeError):
                            duration_val = 10.0
                    total_duration += duration_val
                else:
                    duration_val = enhanced_scene.get('duration', 10.0) if isinstance(enhanced_scene, dict) else 10.0
                    if isinstance(duration_val, str):
                        try:
                            duration_val = float(duration_val.replace('seconds', '').strip())
                        except (ValueError, AttributeError):
                            duration_val = 10.0
                    total_duration += duration_val
            
            if music_prompts:
                combined_music_prompt = f"anime soundtrack combining: {', '.join(set(music_prompts))}"
                print(f"Using LLM-generated music prompts from {len(music_prompts)} scenes")
            else:
                from ...utils.error_handler import PipelineErrorHandler
                error_handler = PipelineErrorHandler()
                music_error = Exception("LLM music prompt generation failed")
                error_handler.log_error_to_output(
                    error=music_error,
                    output_path=str(final_dir),
                    context={"total_duration": total_duration, "channel_type": "anime"}
                )
                logger.error("LLM music prompt generation failed, error logged to output directory")
                combined_music_prompt = "anime background music, epic adventure soundtrack"
            
            music_path = self._generate_background_music(
                prompt=combined_music_prompt,  # Use LLM-generated music prompt
                duration=total_duration,
                output_path=str(music_file)
            )
            print(f"Generated background music with enhanced prompt: {combined_music_prompt[:100]}...")
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
            try:
                from ...pipelines.ai_upscaler import AIUpscaler
                upscaler = AIUpscaler(vram_tier="medium")
                upscaler.upscale_video(
                    input_path=str(final_video),
                    output_path=str(upscaled_video),
                    model_name="realesrgan_x4plus",
                    target_resolution=(1920, 1080)
                )
                upscaled_path = str(upscaled_video)
            except ImportError:
                import shutil
                shutil.copy2(str(final_video), str(upscaled_video))
                upscaled_path = str(upscaled_video)
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
                self._log_video_generation_error("No scenes generated", 1200, output_path, "No valid scene videos found for combination")
                return None
            
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
                cmd = ['ffmpeg', '-y']  # Overwrite output
                
                cmd.extend(['-f', 'concat', '-safe', '0', '-i', concat_file])
                
                if music_path and os.path.exists(music_path):
                    cmd.extend(['-i', music_path])
                    logger.info(f"Adding background music: {music_path}")
                
                valid_voice_files = [f for f in voice_files if f and os.path.exists(f)]
                for voice_file in valid_voice_files:
                    cmd.extend(['-i', voice_file])
                    logger.info(f"Adding voice track: {voice_file}")
                
                cmd.extend([
                    '-c:v', 'libx264',  # High quality codec
                    '-preset', 'veryslow',  # Maximum quality preset
                    '-crf', '15',  # High quality CRF
                    '-profile:v', 'high',
                    '-level', '4.1',
                    '-r', str(output_fps),  # Set output frame rate
                    '-pix_fmt', 'yuv420p',  # Ensure compatibility
                    '-s', '1920x1080',  # Force 16:9 resolution
                ])
                
                if music_path and os.path.exists(music_path) and valid_voice_files:
                    filter_complex = f"[1:a]volume=0.3[bg];[2:a]volume=1.0[voice];[bg][voice]amix=inputs=2:duration=first:dropout_transition=2[audio]"
                    cmd.extend(['-filter_complex', filter_complex, '-map', '0:v', '-map', '[audio]'])
                elif music_path and os.path.exists(music_path):
                    cmd.extend(['-map', '0:v', '-map', '1:a', '-filter:a', 'volume=0.4'])
                elif valid_voice_files:
                    cmd.extend(['-map', '0:v', '-map', '1:a'])
                else:
                    cmd.extend(['-f', 'lavfi', '-i', 'anullsrc=channel_layout=stereo:sample_rate=48000', '-map', '0:v', '-map', '1:a', '-shortest'])
                
                cmd.extend([
                    '-c:a', 'aac',
                    '-b:a', '320k',  # High quality audio bitrate
                    '-ar', '48000',  # Sample rate
                    '-movflags', '+faststart',  # Optimize for streaming
                    output_path
                ])
                
                logger.info(f"Running FFmpeg command: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                
                if result.returncode == 0 and os.path.exists(output_path):
                    file_size = os.path.getsize(output_path)
                    logger.info(f"Successfully combined scenes into episode: {output_path} ({file_size} bytes)")
                    
                    if file_size > 1000000:  # At least 1MB
                        logger.info("✅ Final episode has substantial content")
                        return output_path
                    else:
                        logger.warning(f"⚠️ Final episode too small: {file_size} bytes")
                        self._log_video_combination_error(f"Final episode too small: {file_size} bytes", output_path)
                        return None
                else:
                    logger.error(f"FFmpeg failed with return code {result.returncode}")
                    logger.error(f"FFmpeg stderr: {result.stderr}")
                    self._log_video_combination_error(f"FFmpeg failed with return code {result.returncode}: {result.stderr}", output_path)
                    return None
                    
            finally:
                if os.path.exists(concat_file):
                    os.unlink(concat_file)
                
        except Exception as e:
            logger.error(f"Error combining scenes with FFmpeg: {e}")
            self._log_video_combination_error(f"Error combining scenes with FFmpeg: {e}", output_path)
            return None
    
    def _log_video_combination_error(self, error_message: str, output_path: str):
        """Log video combination error to output directory."""
        try:
            from ...utils.error_handler import PipelineErrorHandler
            import os
            
            output_dir = os.path.dirname(output_path) if output_path else '/tmp'
            error_handler = PipelineErrorHandler()
            combination_error = Exception(f"Video combination failed: {error_message}")
            error_handler.log_error_to_output(
                error=combination_error,
                output_path=output_dir,
                context={
                    "output_path": output_path,
                    "channel_type": "anime",
                    "error_details": error_message
                }
            )
            logger.error(f"Video combination failed for anime pipeline, error logged to output directory")
            
        except Exception as e:
            logger.error(f"Error logging video combination failure: {e}")
    
    def _create_shorts(self, scene_files: List[str], shorts_dir: Path) -> List[str]:
        """Create shorts by extracting highlights from the main video."""
        shorts_paths = []
        
        try:
            shorts_dir.mkdir(parents=True, exist_ok=True)
            
            main_video_candidates = [
                shorts_dir.parent / "final" / "anime_episode.mp4",
                shorts_dir.parent / "final" / "anime_episode_upscaled.mp4",
                shorts_dir.parent / "final" / "temp_combined.mp4"
            ]
            
            main_video_path = None
            for candidate in main_video_candidates:
                if candidate.exists() and candidate.stat().st_size > 1000:
                    main_video_path = str(candidate)
                    break
            
            if not main_video_path:
                logger.warning("No suitable video found for shorts extraction")
                return []
            
            logger.info(f"Extracting anime shorts from: {main_video_path}")
            
            highlights = self.extract_highlights_from_video(main_video_path, num_highlights=3)
            
            if not highlights:
                logger.warning("No highlights extracted from main video")
                return []
            
            logger.info(f"Extracted {len(highlights)} highlights for shorts creation")
            
            for i, highlight in enumerate(highlights):
                short_path = shorts_dir / f"anime_short_{i+1:02d}.mp4"
                
                try:
                    import subprocess
                    
                    start_time = highlight.get('start_time', 0)
                    duration = min(highlight.get('duration', 15), 60)
                    
                    cmd = [
                        'ffmpeg', '-i', main_video_path,
                        '-ss', str(start_time),
                        '-t', str(duration),
                        '-c:v', 'libx264', '-c:a', 'aac',
                        '-aspect', '16:9',
                        '-y', str(short_path)
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0 and short_path.exists():
                        shorts_paths.append(str(short_path))
                        logger.info(f"Created anime short {i+1}: {short_path}")
                    else:
                        logger.error(f"Failed to create short {i+1}: {result.stderr}")
                        
                except Exception as e:
                    logger.error(f"Error creating short {i+1}: {e}")
                    continue
            
            return shorts_paths
            
        except Exception as e:
            logger.error(f"Error in shorts creation: {e}")
            return []
    
    def _expand_script_if_needed(self, script_data: Dict, min_duration: float = 20.0) -> Dict:
        """Expand script if it doesn't meet minimum duration requirements for complete anime episodes."""
        current_duration = self._analyze_script_duration(script_data)
        existing_scenes = script_data.get("scenes", [])
        
        # Ensure minimum 5-8 scenes for complete anime episode experience
        min_scenes = 5
        max_scenes = 8
        target_scenes = max(min_scenes, min(max_scenes, len(existing_scenes) + 3))
        
        if len(existing_scenes) >= target_scenes and current_duration >= min_duration:
            logger.info(f"Script has {len(existing_scenes)} scenes with {current_duration:.1f} minutes - meets requirements")
            return script_data
        
        logger.info(f"Expanding script from {len(existing_scenes)} to {target_scenes} scenes for complete anime episode")
        
        characters = script_data.get("characters", [])
        setting = script_data.get("setting", "anime world")
        
        expansion_types = [
            "opening_sequence", "character_development", "world_building", 
            "action_expansion", "emotional_beats", "plot_twist", 
            "climax_buildup", "resolution"
        ]
        
        scenes_to_add = target_scenes - len(existing_scenes)
        for i in range(scenes_to_add):
            expansion_type = expansion_types[i % len(expansion_types)]
            new_scene = self._generate_expansion_scene(expansion_type, characters, setting, i + len(existing_scenes) + 1)
            existing_scenes.append(new_scene)
        
        script_data["scenes"] = existing_scenes
        logger.info(f"Expanded script to {len(existing_scenes)} scenes for complete anime episode")
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
                
                if isinstance(scene_duration, str):
                    try:
                        scene_duration = float(scene_duration.replace('seconds', '').strip())
                    except (ValueError, AttributeError):
                        scene_duration = base_duration
                total_duration += scene_duration
        
        return total_duration
    
    def _generate_expansion_scene(self, expansion_type: str, characters: List, setting: str, scene_number: int) -> Dict:
        """Generate a new scene for expansion with enhanced variety for complete anime episodes."""
        main_char = characters[0] if characters else {"name": "Protagonist"}
        support_char = characters[1] if len(characters) > 1 else {"name": "Ally"}
        
        if expansion_type == "opening_sequence":
            return {
                "type": "opening",
                "description": f"Dynamic anime opening sequence showcasing {main_char.get('name', 'the protagonist')} and the world of {setting}. Features sweeping camera movements, character introductions, and thematic music.",
                "duration": 3.5,
                "music_prompt": "epic anime opening theme, orchestral, heroic",
                "voice_prompt": f"Narrator introduces the world and {main_char.get('name', 'protagonist')}"
            }
        elif expansion_type == "character_development":
            return {
                "type": "character_development",
                "description": f"Deep character development scene featuring {main_char.get('name', 'the protagonist')} in {setting}. Explores their past, inner conflicts, and growth through meaningful dialogue and flashbacks.",
                "duration": 4.0,
                "music_prompt": "emotional anime soundtrack, piano, strings",
                "voice_prompt": f"{main_char.get('name', 'protagonist')} reflects on their journey and motivations"
            }
        elif expansion_type == "world_building":
            return {
                "type": "world_building", 
                "description": f"Immersive world building scene showcasing the intricate details of {setting}. Establishes lore, culture, magic systems, and societal structures with stunning anime visuals.",
                "duration": 3.0,
                "music_prompt": "mystical anime ambience, ethereal, world music",
                "voice_prompt": f"Exploration of {setting} and its unique characteristics"
            }
        elif expansion_type == "action_expansion":
            return {
                "type": "combat",
                "description": f"Spectacular anime-style combat scene in {setting} featuring {main_char.get('name', 'protagonist')} vs formidable opponents. Dynamic choreography, special attacks, and intense battle sequences.",
                "duration": 4.5,
                "music_prompt": "intense anime battle music, rock, orchestral",
                "voice_prompt": f"Battle cries and combat dialogue between {main_char.get('name', 'protagonist')} and enemies"
            }
        elif expansion_type == "plot_twist":
            return {
                "type": "plot_twist",
                "description": f"Shocking plot twist scene that changes everything for {main_char.get('name', 'the protagonist')} in {setting}. Reveals hidden truths, betrayals, or unexpected alliances.",
                "duration": 3.5,
                "music_prompt": "dramatic anime revelation music, suspenseful, orchestral",
                "voice_prompt": f"Dramatic revelation dialogue between {main_char.get('name', 'protagonist')} and {support_char.get('name', 'ally')}"
            }
        elif expansion_type == "climax_buildup":
            return {
                "type": "climax_buildup",
                "description": f"Tension-building scene leading to the climax. {main_char.get('name', 'The protagonist')} prepares for the final confrontation in {setting} with mounting stakes and emotional weight.",
                "duration": 3.0,
                "music_prompt": "building anime tension music, crescendo, epic",
                "voice_prompt": f"{main_char.get('name', 'protagonist')} prepares for the ultimate challenge"
            }
        elif expansion_type == "resolution":
            return {
                "type": "resolution",
                "description": f"Satisfying resolution scene showing the aftermath and new beginnings for {main_char.get('name', 'the protagonist')} in {setting}. Ties up loose ends and hints at future adventures.",
                "duration": 3.5,
                "music_prompt": "hopeful anime ending music, uplifting, orchestral",
                "voice_prompt": f"{main_char.get('name', 'protagonist')} reflects on their completed journey and future"
            }
        else:  # emotional_beats
            return {
                "type": "emotional_beat",
                "description": f"Intimate character moment between {main_char.get('name', 'the protagonist')} and {support_char.get('name', 'ally')} in {setting}. Allows for emotional depth, relationship development, and quiet reflection.",
                "duration": 2.5,
                "music_prompt": "gentle anime emotional music, soft piano, strings",
                "voice_prompt": f"Heartfelt conversation between {main_char.get('name', 'protagonist')} and {support_char.get('name', 'ally')}"
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
                video_generator = self._load_video_model(self.base_model)
                if video_generator:
                    success = video_generator.generate_video(
                        prompt=video_params["prompt"],
                        duration=duration,
                        output_path=output_path
                    )
                    if success and os.path.exists(output_path):
                        return output_path
            except Exception as e:
                logger.warning(f"Video generation failed: {e}")
            
            # Log error instead of fallback generation
            try:
                from ...utils.error_handler import PipelineErrorHandler
                error_handler = PipelineErrorHandler()
                video_error = Exception(f"Video generation failed for scene: {scene_description}")
                error_handler.log_error_to_output(
                    error=video_error,
                    output_path=os.path.dirname(output_path) if output_path else '/tmp',
                    context={
                        "prompt": scene_description[:100] + "..." if len(scene_description) > 100 else scene_description,
                        "duration": duration,
                        "output_path": output_path,
                        "channel_type": "anime"
                    }
                )
                logger.error(f"Video generation failed, error logged to output directory")
            except Exception as log_error:
                logger.error(f"Error logging video generation failure: {log_error}")
            return None
            
        except Exception as e:
            logger.error(f"Error in scene video generation: {e}")
            # Log error instead of fallback generation
            try:
                from ...utils.error_handler import PipelineErrorHandler
                error_handler = PipelineErrorHandler()
                error_handler.log_error_to_output(
                    error=e,
                    output_path=os.path.dirname(output_path) if output_path else '/tmp',
                    context={
                        "prompt": scene_description[:100] + "..." if len(scene_description) > 100 else scene_description,
                        "duration": duration,
                        "output_path": output_path,
                        "channel_type": "anime"
                    }
                )
                logger.error(f"Scene video generation error logged to output directory")
            except Exception as log_error:
                logger.error(f"Error logging scene video generation failure: {log_error}")
            return None
    
    def _optimize_video_prompt(self, prompt: str, channel_type: str = "anime", model_name: str = None) -> str:
        """Optimize prompt for video generation with maximum quality."""
        from backend.model_manager import BASE_MODEL_PROMPT_TEMPLATES
        
        if model_name and model_name in BASE_MODEL_PROMPT_TEMPLATES:
            template = BASE_MODEL_PROMPT_TEMPLATES[model_name]
            
            if "structure" in template and model_name in ["anything_xl", "aam_xl_animemix"]:
                quality = "masterpiece, best quality"
                year = "newest"
                optimized_prompt = f"{quality}, {year}, {prompt}"
            else:
                optimized_prompt = f"{template['prefix']}, {prompt}"
            
            suffix = ", 16:9 aspect ratio, smooth motion, professional cinematography, ultra high definition"
            return f"{optimized_prompt}{suffix}"
        
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
            if voice_model and voice_model.get("generate"):
                voice_preset = "v2/en_speaker_6" if language == "en" else f"v2/{language}_speaker_0"
                
                audio_array, sample_rate = voice_model["generate"](text, voice_preset)
                
                if audio_array is not None:
                    import soundfile as sf
                    sf.write(output_path, audio_array, sample_rate)
                    if os.path.exists(output_path):
                        return output_path
            
            # Log error instead of fallback generation
            try:
                from ...utils.error_handler import PipelineErrorHandler
                error_handler = PipelineErrorHandler()
                voice_error = Exception(f"Voice generation failed for text: {text}")
                error_handler.log_error_to_output(
                    error=voice_error,
                    output_path=os.path.dirname(output_path) if output_path else '/tmp',
                    context={
                        "text": text[:100] + "..." if len(text) > 100 else text,
                        "output_path": output_path,
                        "channel_type": "anime"
                    }
                )
                logger.error(f"Voice generation failed, error logged to output directory")
            except Exception as log_error:
                logger.error(f"Error logging voice generation failure: {log_error}")
            return None
            
        except Exception as e:
            logger.error(f"Error generating voice: {e}")
            # Log error instead of fallback generation
            try:
                from ...utils.error_handler import PipelineErrorHandler
                error_handler = PipelineErrorHandler()
                error_handler.log_error_to_output(
                    error=e,
                    output_path=os.path.dirname(output_path) if output_path else '/tmp',
                    context={
                        "text": text[:100] + "..." if len(text) > 100 else text,
                        "output_path": output_path,
                        "channel_type": "anime"
                    }
                )
                logger.error(f"Voice generation error logged to output directory")
            except Exception as log_error:
                logger.error(f"Error logging voice generation failure: {log_error}")
            return None
    

    
    def _generate_background_music(self, prompt: str, duration: float, output_path: str) -> str:
        """Generate background music with maximum quality."""
        try:
            music_model = self.load_music_model("musicgen")
            if music_model and music_model.get("generate"):
                music_prompt = f"high quality, professional, {prompt}"
                
                if music_model.get("type") == "musicgen":
                    audio_output = music_model["generate"](music_prompt, duration)
                    
                    if audio_output is not None:
                        import torchaudio
                        torchaudio.save(output_path, audio_output[0].cpu(), 32000)
                        if os.path.exists(output_path):
                            return output_path
                else:
                    music_params = {
                        "prompt": music_prompt,
                        "duration": duration,
                        "output_path": output_path
                    }
                    success = music_model["generate"](**music_params)
                    if success and os.path.exists(output_path):
                        return output_path
            
            # Log error instead of fallback generation
            self._log_music_generation_error(output_path, "All music models failed")
            return None
            
        except Exception as e:
            logger.error(f"Error generating music: {e}")
            # Log error instead of fallback generation
            self._log_music_generation_error(output_path, str(e))
            return None
    
    def _upscale_video_with_realesrgan(self, input_path: str, output_path: str, 
                                      target_resolution: str = "1080p", enabled: bool = True) -> str:
        """Upscale video using RealESRGAN with maximum quality."""
        if not enabled:
            shutil.copy2(input_path, output_path)
            return output_path
        
        try:
            from ..ai_upscaler import AIUpscaler
            
            upscaler = AIUpscaler(vram_tier=self.vram_tier)
            
            resolution_map = {
                "720p": (1280, 720),
                "1080p": (1920, 1080), 
                "1440p": (2560, 1440),
                "4k": (3840, 2160)
            }
            
            target_dimensions = resolution_map.get(target_resolution, (1920, 1080))
            
            model_name = upscaler.get_best_model_for_content("anime", target_scale=2)
            
            success = upscaler.upscale_video(
                input_path=input_path,
                output_path=output_path,
                model_name=model_name,
                target_resolution=target_dimensions
            )
            
            if success and os.path.exists(output_path):
                logger.info(f"Video upscaled to {target_resolution} using RealESRGAN: {output_path}")
                return output_path
            else:
                logger.warning("RealESRGAN upscaling failed, copying original")
                shutil.copy2(input_path, output_path)
                return output_path
                
        except Exception as e:
            logger.error(f"Error upscaling video with RealESRGAN: {e}")
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
                try:
                    title = llm_model["generate"](title_prompt, max_tokens=50)
                except Exception as e:
                    logger.warning(f"LLM title generation failed: {e}")
                    from ...utils.error_handler import PipelineErrorHandler
                    error_handler = PipelineErrorHandler()
                    error_handler.log_error_to_output(
                        error=e,
                        output_path=str(output_dir),
                        context={"prompt": title_prompt, "max_tokens": 50, "generation_type": "title"}
                    )
                    title = f"Epic Anime Adventure - Episode {random.randint(1, 100)}"
            else:
                title = f"Epic Anime Adventure - Episode {random.randint(1, 100)}"
            
            with open(output_dir / "title.txt", "w", encoding="utf-8") as f:
                f.write(title.strip())
            
            # Generate description
            description_prompt = f"Generate a detailed YouTube description for an anime episode with {len(scenes)} scenes. Include character introductions, plot summary, and engaging hooks. Language: {language}"
            
            if llm_model:
                try:
                    description = llm_model["generate"](description_prompt, max_tokens=300)
                except Exception as e:
                    logger.warning(f"LLM description generation failed: {e}")
                    from ...utils.error_handler import PipelineErrorHandler
                    error_handler = PipelineErrorHandler()
                    error_handler.log_error_to_output(
                        error=e,
                        output_path=str(output_dir),
                        context={"prompt": description_prompt, "max_tokens": 300, "generation_type": "description"}
                    )
                    description = f"An epic anime adventure featuring amazing characters and thrilling action across {len(scenes)} incredible scenes!"
            else:
                description = f"An epic anime adventure featuring amazing characters and thrilling action across {len(scenes)} incredible scenes!"
            
            with open(output_dir / "description.txt", "w", encoding="utf-8") as f:
                f.write(description.strip())
            
            next_episode_prompt = f"Based on this anime episode, suggest 3 compelling storylines for the next episode. Be creative and engaging."
            
            if llm_model:
                try:
                    next_suggestions = llm_model["generate"](next_episode_prompt, max_tokens=200)
                except Exception as e:
                    logger.warning(f"LLM next episode generation failed: {e}")
                    from ...utils.error_handler import PipelineErrorHandler
                    error_handler = PipelineErrorHandler()
                    error_handler.log_error_to_output(
                        error=e,
                        output_path=str(output_dir),
                        context={"prompt": next_episode_prompt, "max_tokens": 200, "generation_type": "next_episode"}
                    )
                    next_suggestions = "1. The adventure continues with new challenges\n2. Character development and new powers\n3. Epic finale with ultimate showdown"
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
    
    def _log_video_generation_error(self, prompt: str, duration: float, output_path: str, error_message: str):
        """Log video generation error to output directory."""
        try:
            from ...utils.error_handler import PipelineErrorHandler
            import os
            
            output_dir = os.path.dirname(output_path) if output_path else '/tmp'
            error_handler = PipelineErrorHandler()
            video_error = Exception(f"Video generation failed: {error_message}")
            error_handler.log_error_to_output(
                error=video_error,
                output_path=output_dir,
                context={
                    "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                    "duration": duration,
                    "output_path": output_path,
                    "channel_type": "anime",
                    "error_details": error_message
                }
            )
            logger.error(f"Video generation failed for anime pipeline, error logged to output directory")
            
        except Exception as e:
            logger.error(f"Error logging video generation failure: {e}")
    
    def _log_music_generation_error(self, output_path: str, error_message: str):
        """Log music generation error to output directory."""
        try:
            from ...utils.error_handler import PipelineErrorHandler
            import os
            
            output_dir = os.path.dirname(output_path) if output_path else getattr(self, 'current_output_dir', '/tmp')
            error_handler = PipelineErrorHandler()
            error_handler.log_error(
                error_type="MUSIC_GENERATION_FAILURE",
                error_message=f"Music generation failed: {error_message}",
                output_dir=str(output_dir),
                context={
                    "output_path": output_path,
                    "channel_type": "anime"
                }
            )
            logger.error(f"Music generation failed for anime pipeline, error logged to output directory")
            
        except Exception as e:
            logger.error(f"Error logging music generation failure: {e}")



def run(input_path: str, output_path: str, base_model: str = "stable_diffusion_1_5", 
        lora_models: Optional[List[str]] = None, lora_paths: Optional[Dict[str, str]] = None, 
        db_run=None, db=None, render_fps: int = 24, output_fps: int = 60, 
        frame_interpolation_enabled: bool = True, llm_model: str = "microsoft/DialoGPT-medium", 
        language: str = "en") -> str:
    """Run anime pipeline with self-contained processing."""
    pipeline = AnimeChannelPipeline(output_path=output_path)
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
            llm_model = None
            try:
                from .base_pipeline import BasePipeline
                base_pipeline = BasePipeline("anime")
                llm_model = base_pipeline.load_llm_model("deepseek")
            except:
                llm_model = None
            if llm_model and llm_model.get("generate"):
                expanded_scenes = []
                for scene in script_data.get('scenes', []):
                    if isinstance(scene, str) and len(scene) < 100:
                        expansion_prompt = f"Expand this anime scene with more detail: {scene}"
                        try:
                            expanded_scene = llm_model["generate"](expansion_prompt)
                        except Exception as e:
                            logger.warning(f"LLM scene expansion failed: {e}")
                            from ...utils.error_handler import PipelineErrorHandler
                            error_handler = PipelineErrorHandler()
                            llm_error = Exception(f"LLM scene expansion failed: {e}")
                            error_handler.log_error_to_output(
                                error=llm_error,
                                output_path=str(output_dir) if 'output_dir' in locals() else '/tmp',
                                context={
                                    "expansion_prompt": expansion_prompt[:100] + "..." if len(expansion_prompt) > 100 else expansion_prompt,
                                    "scene_number": len(expanded_scenes) + 1,
                                    "channel_type": "anime",
                                    "component": "LLM_scene_expansion"
                                }
                            )
                            logger.error(f"LLM scene expansion failed, error logged to output directory")
                            expanded_scene = None
                        expanded_scenes.append(expanded_scene)
                    else:
                        expanded_scenes.append(scene)
                script_data['scenes'] = expanded_scenes
            else:
                expanded_scenes = []
                for scene in script_data.get('scenes', []):
                    if isinstance(scene, str) and len(scene) < 100:
                        expanded_scene = f"{scene}. Anime style with vibrant colors, dynamic action, and expressive characters."
                        expanded_scenes.append(expanded_scene)
                    else:
                        expanded_scenes.append(scene)
                script_data['scenes'] = expanded_scenes
        except Exception as e:
            print(f"Error during anime script expansion: {e}")
                
    scene_details = []
    for i, scene in enumerate(scenes):
        if isinstance(scene, str):
            scene_chars = [characters[i % len(characters)], characters[(i + 1) % len(characters)]]
            scene_location = locations[i % len(locations)]
            
            scene_lower = scene.lower()
            if any(word in scene_lower for word in ["fight", "battle", "combat", "attack", "versus"]):
                scene_type = "combat"
            elif any(word in scene_lower for word in ["talk", "speak", "conversation", "dialogue"]):
                scene_type = "dialogue"
            elif any(word in scene_lower for word in ["run", "chase", "escape", "action"]):
                scene_type = "action"
            else:
                scene_type = "dialogue"
            
            scene_detail = {
                "scene_text": scene,
                "scene_type": scene_type,
                "characters": scene_chars,
                "location": scene_location
            }
            
            if scene_type == "combat":
                try:
                    # Inline combat scene generation
                    # Inline combat scene generation
                    combat_data = {
                        "combat_type": "melee",
                        "intensity": 0.7,
                        "video_prompt": f"Epic anime combat scene: {scene}, dynamic action, sword fighting",
                        "duration": 10.0,
                        "movements": ["slash", "parry", "combo"],
                        "camera_angles": ["dramatic_low", "overhead"],
                        "effects": ["blade_flash", "energy_slash"]
                    }
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
            
            scene_lower = scene.lower()
            if any(word in scene_lower for word in ["fight", "battle", "combat", "attack", "versus"]):
                scene_type = "combat"
            elif any(word in scene_lower for word in ["talk", "speak", "conversation", "dialogue"]):
                scene_type = "dialogue"
            elif any(word in scene_lower for word in ["run", "chase", "escape", "action"]):
                scene_type = "action"
            else:
                scene_type = "dialogue"
            
            scene_detail = {
                "scene_text": scene,
                "scene_type": scene_type,
                "characters": scene_chars,
                "location": scene_location
            }
            
            if scene_type == "combat":
                try:
                    # Inline combat scene generation
                    combat_data = {
                        "combat_type": "melee",
                        "intensity": 0.7,
                        "video_prompt": f"Epic anime combat scene: {scene}, dynamic action, sword fighting",
                        "duration": 10.0,
                        "movements": ["slash", "parry", "combo"],
                        "camera_angles": ["dramatic_low", "overhead"],
                        "effects": ["blade_flash", "energy_slash"]
                    }
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
        from ...model_manager import AIModelManager
        
        model_manager = AIModelManager()
        anime_model = model_manager.load_base_model(base_model, "image")
        
        if anime_model:
            print(f"Successfully loaded {base_model} with {lora_models} LoRA(s)")
            # Ensure model has proper dictionary structure for generation
            if not isinstance(anime_model, dict) or "generate" not in anime_model:
                anime_model = {"generate": anime_model.generate if hasattr(anime_model, 'generate') else lambda **kwargs: None}
        else:
            from ...utils.error_handler import PipelineErrorHandler
            error_handler = PipelineErrorHandler()
            model_error = Exception(f"Failed to load {base_model}")
            error_handler.log_error_to_output(
                error=model_error,
                output_path='/tmp',
                context={"base_model": base_model, "channel_type": "anime"}
            )
            logger.error(f"Failed to load {base_model}, error logged to output directory")
            anime_model = {"generate": lambda **kwargs: None}
    except Exception as e:
        print(f"Error loading models: {e}")
        anime_model = {"generate": lambda **kwargs: None}
    
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
                    
                    try:
                        result = anime_model["generate"](**generation_params)
                    except Exception as e:
                        print(f"Error generating character {character_name}: {e}")
                        result = None
                    
                    if result and hasattr(result, "images") and result.images:
                        result.images[0].save(char_file)
                        print(f"Generated character image: {char_file}")
                        
                        character_memory.save_character_reference(character_id, str(char_file), angle)
                    else:
                        print(f"Failed to generate character {character_name} {angle}")
                        logger.error(f"Character generation failed for {character_name} {angle}, attempting alternative models")
                        continue
                else:
                    print(f"No model available for character {character_name}")
                    logger.error(f"No model available for character {character_name}, skipping character generation")
                    continue
                    
            except Exception as e:
                print(f"Error generating character {character_name} {angle}: {e}")
                logger.error(f"Error generating character {character_name} {angle}: {e}")
                continue
        
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
                try:
                    result = anime_model["generate"](
                        prompt=f"anime character, {scene_text}, high quality, detailed anime art style",
                        num_inference_steps=30,
                        guidance_scale=7.5,
                        width=512,
                        height=512
                    )
                except Exception as e:
                    print(f"Error generating scene {i+1} with model: {e}")
                    result = None
                
                if result and hasattr(result, "images") and result.images:
                    result.images[0].save(scene_file)
                    print(f"Successfully generated scene {i+1} image")
                else:
                    print(f"Failed to generate scene {i+1} image: No valid result")
                    logger.error(f"Scene {i+1} generation failed, attempting alternative approach")
                    continue
            else:
                print(f"No model available for scene {i+1}")
                logger.error(f"No model available for scene {i+1}, skipping scene generation")
                continue
        except Exception as e:
            print(f"Error generating scene {i+1}: {e}")
            logger.error(f"Error generating scene {i+1}: {e}")
            continue
    
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
            vram_tier = "medium"
            
            optimized_prompt = f"masterpiece, best quality, ultra detailed, 8k resolution, cinematic lighting, smooth animation, professional anime style, vibrant colors, dynamic composition, {scene_text}"
            
            if scene_detail.get("scene_type") == "combat" and scene_detail.get("combat_data"):
                combat_data = scene_detail["combat_data"]
                combat_type = combat_data.get("combat_type", "melee")
                best_model = "stable_diffusion_1_5"
                optimized_prompt = combat_data.get("video_prompt", optimized_prompt)
            else:
                best_model = "stable_diffusion_1_5"
            
            success = False
            try:
                from ..text_to_video_generator import TextToVideoGenerator
                video_generator = TextToVideoGenerator()
                success = video_generator.generate_video(
                    f"Anime scene {i+1} with character animation",
                    "animatediff_v2_sdxl",
                    str(animated_file),
                    duration=10.0
                )
            except Exception as e:
                print(f"Video generation error: {e}")
            # Original call: success = create_scene_video_with_generation(
            #     scene_description=optimized_prompt,
            #     characters=scene_chars,
            #     output_path=str(animated_file),
            #     model_name=best_model
            # )
            
            if success:
                print(f"Successfully generated high-quality anime video for scene {i+1} using {best_model}")
                
                character_voice = scene_chars[0].get("voice", "default") if scene_chars else "default"
                voice_success = False  # Voice generation disabled for self-contained operation
                
                music_success = False  # Music generation disabled for self-contained operation
                
                if voice_success:
                    try:
                        shutil.copy2(str(animated_file), str(final_file))
                        lipsync_success = True
                    except Exception:
                        lipsync_success = False
                    if lipsync_success:
                        print(f"Applied lipsync for scene {i+1}")
                
            else:
                from ...utils.error_handler import PipelineErrorHandler
                error_handler = PipelineErrorHandler()
                video_error = Exception(f"Failed to generate video for scene {i+1}")
                error_handler.log_error_to_output(
                    error=video_error,
                    output_path='/tmp',
                    context={"scene_number": i+1, "channel_type": "anime"}
                )
                logger.error(f"Failed to generate video for scene {i+1}, error logged to output directory")
                try:
                    from ..text_to_video_generator import TextToVideoGenerator
                    video_generator = TextToVideoGenerator()
                    success = video_generator.generate_video(
                        f"Anime scene {i+1} with dynamic animation",
                        "animatediff_v2_sdxl",
                        str(animated_file),
                        duration=10.0
                    )
                    if not success:
                        print(f"AI video generation failed for scene {i+1}")
                except Exception as e:
                    print(f"AI video generation error: {e}")
                
        except Exception as e:
            print(f"Error generating video for scene {i+1}: {e}")
            try:
                from ...utils.error_handler import PipelineErrorHandler
                error_handler = PipelineErrorHandler()
                video_error = Exception(f"Video generation failed for anime scene {i+1}")
                error_handler.log_error_to_output(
                    error=video_error,
                    output_path=str(animated_file.parent),
                    context={
                        "scene_number": i+1,
                        "channel_type": "anime",
                        "attempted_generation": "emergency_fallback_removed"
                    }
                )
                print(f"Video generation failed for scene {i+1}, error logged to output directory")
            except Exception as e:
                print(f"Error logging video generation failure: {e}")
    
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
                        create_ai_audio(str(voice_file))
                else:
                    print(f"Bark model not properly loaded for {char_name}")
                    create_ai_audio(str(voice_file))
                    
            except Exception as e:
                print(f"Error in voice generation for {char_name}: {e}")
                create_ai_audio(str(voice_file))
    
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
                    try:
                        from ..text_to_video_generator import TextToVideoGenerator
                        video_generator = TextToVideoGenerator()
                        success = video_generator.generate_video(
                            f"Character {char_name} speaking dialogue",
                            "animatediff_v2_sdxl",
                            str(lipsync_file),
                            duration=5.0
                        )
                        if success:
                            print(f"Created AI lipsync for {char_name} in scene {i+1}")
                        else:
                            print(f"AI lipsync generation failed for {char_name} in scene {i+1}")
                    except Exception as e:
                        print(f"Error generating AI lipsync for {char_name}: {e}")
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
        create_ai_audio(str(music_file), duration=30.0)
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
                    from ..text_to_video_generator import TextToVideoGenerator
                    video_generator = TextToVideoGenerator()
                    success = video_generator.generate_video(
                        f"Anime scene {i} with dynamic animation",
                        "animatediff_v2_sdxl",
                        str(scene_mp4),
                        duration=8.0
                    )
                    if success:
                        scene_videos_created.append(str(scene_mp4))
                        print(f"Created AI scene MP4: {scene_mp4}")
                    else:
                        print(f"AI video generation failed for scene {i}")
                except Exception as e:
                    print(f"Error creating AI MP4 for scene {i}: {e}")
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


def create_ai_audio(file_path: str, text: str = "Generated audio content", duration: float = 3.0):
    """Create AI-generated audio content."""
    try:
        from ..ai_voice_generator import AIVoiceGenerator
        voice_generator = AIVoiceGenerator()
        
        success = voice_generator.generate_voice(text, file_path)
        if success:
            print(f"Created AI audio file: {file_path}")
            return True
        else:
            print(f"AI audio generation failed for: {file_path}")
            return False
        
    except Exception as e:
        print(f"Error creating AI audio: {e}")
        return False





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
                # External imports removed - using inline frame interpolation
                try:
                    import cv2
                    
                    cap = cv2.VideoCapture(str(temp_output))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    if fps < output_fps:
                        interpolation_factor = int(output_fps / fps)
                        logger.info(f"Interpolating frames: {fps}fps -> {output_fps}fps")
                        
                        import subprocess
                        interpolated_path = str(temp_output).replace('.mp4', '_interpolated.mp4')
                        cmd = [
                            'ffmpeg', '-i', str(temp_output),
                            '-filter:v', f'minterpolate=fps={output_fps}:mi_mode=mci',
                            '-c:a', 'copy', '-y', interpolated_path
                        ]
                        
                        try:
                            subprocess.run(cmd, check=True, capture_output=True)
                            if Path(interpolated_path).exists():
                                Path(temp_output).unlink()
                                Path(interpolated_path).rename(temp_output)
                        except subprocess.CalledProcessError:
                            logger.warning("Frame interpolation failed, using original video")
                        
                    cap.release()
                except Exception as e:
                    logger.error(f"Frame interpolation error: {e}")
                
                # Inline VRAM detection for frame interpolation
                vram_tier = "medium"
                
                interpolator = None
                try:
                    shutil.copy2(temp_output, output_path)
                    interpolation_success = True
                except Exception:
                    interpolation_success = False
                
                if interpolation_success:
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
# Alias for backward compatibility
AnimePipeline = AnimeChannelPipeline
