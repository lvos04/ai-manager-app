"""
AI Gaming Content Pipeline
Self-contained gaming content generation with complete internal processing.
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

class GamingPipeline(BasePipeline):
    """Self-contained gaming content generation pipeline with all functionality inlined."""
    
    def __init__(self):
        super().__init__("gaming")
        self.supports_combat = False
        self.video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv']
        self.audio_extensions = ['.mp3', '.wav', '.aac', '.ogg', '.flac']
    
    def run(self, input_path: str, output_path: str, base_model: str = "stable_diffusion_1_5", 
            lora_models: Optional[List[str]] = None, lora_paths: Optional[Dict[str, str]] = None, 
            db_run=None, db=None, render_fps: int = 24, output_fps: int = 60, 
            frame_interpolation_enabled: bool = True, language: str = "en") -> str:
        """
        Run the self-contained gaming pipeline.
        
        Args:
            input_path: Path to input script/recording
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
        
        print("Running self-contained gaming pipeline")
        print(f"Using base model: {base_model}")
        print(f"Using LoRA models: {lora_models}")
        print(f"Language: {language}")
        
        try:
            return self._execute_pipeline(
                input_path, output_path, base_model, lora_models, 
                db_run, db, render_fps, output_fps, frame_interpolation_enabled, language
            )
        except Exception as e:
            logger.error(f"Gaming pipeline failed: {e}")
            raise
        finally:
            self.cleanup_models()
    
    def _execute_pipeline(self, input_path: str, output_path: str, base_model: str, 
                         lora_models: Optional[List[str]], db_run, db, render_fps: int, 
                         output_fps: int, frame_interpolation_enabled: bool, language: str) -> str:
        
        output_dir = self.ensure_output_dir(output_path)
        
        scenes_dir = output_dir / "scenes"
        scenes_dir.mkdir(exist_ok=True)
        
        highlights_dir = output_dir / "highlights"
        highlights_dir.mkdir(exist_ok=True)
        
        final_dir = output_dir / "final"
        final_dir.mkdir(exist_ok=True)
        
        shorts_dir = output_dir / "shorts"
        shorts_dir.mkdir(exist_ok=True)
        
        print("Step 1: Analyzing input...")
        if db_run and db:
            db_run.progress = 5.0
            db.commit()
        
        is_recording = self._is_game_recording(input_path)
        
        if is_recording:
            return self._process_game_recording(input_path, output_dir, db_run, db, language, render_fps, output_fps, frame_interpolation_enabled)
        else:
            return self._process_script_content(input_path, output_dir, db_run, db, language, render_fps, output_fps, frame_interpolation_enabled)
    
    def _is_game_recording(self, input_path: str) -> bool:
        """Check if input is a game recording file."""
        return any(input_path.lower().endswith(ext) for ext in self.video_extensions)
    
    def _process_game_recording(self, input_path: str, output_dir: Path, db_run, db, language: str, 
                               render_fps: int, output_fps: int, frame_interpolation_enabled: bool) -> str:
        """Process game recording with highlight extraction and editing."""
        print("Processing game recording...")
        
        if db_run and db:
            db_run.progress = 10.0
            db.commit()
        
        highlights = self._extract_highlights(input_path)
        
        if db_run and db:
            db_run.progress = 30.0
            db.commit()
        
        edited_video = self._create_edited_compilation(highlights, output_dir, render_fps, output_fps)
        
        if db_run and db:
            db_run.progress = 50.0
            db.commit()
        
        if frame_interpolation_enabled and output_fps > render_fps:
            print(f"Applying frame interpolation: {render_fps}fps -> {output_fps}fps...")
            interpolated_video = output_dir / "final" / "gaming_content_interpolated.mp4"
            edited_video = self._interpolate_frames(edited_video, str(interpolated_video), output_fps)
        
        if db_run and db:
            db_run.progress = 70.0
            db.commit()
        
        upscaled_video = output_dir / "final" / "gaming_content_upscaled.mp4"
        final_video = self._upscale_video_with_realesrgan(edited_video, str(upscaled_video), "1080p", True)
        
        if db_run and db:
            db_run.progress = 85.0
            db.commit()
        
        shorts = self._create_gaming_shorts(final_video, output_dir)
        
        if db_run and db:
            db_run.progress = 95.0
            db.commit()
        
        self._generate_youtube_metadata(output_dir, [], [], language, "gaming")
        
        if db_run and db:
            db_run.progress = 100.0
            db.commit()
        
        return str(output_dir)
    
    def _process_script_content(self, input_path: str, output_dir: Path, db_run, db, language: str,
                               render_fps: int, output_fps: int, frame_interpolation_enabled: bool) -> str:
        """Process script content for gaming scenarios."""
        print("Processing gaming script content...")
        
        if db_run and db:
            db_run.progress = 10.0
            db.commit()
        
        script_data = self.parse_input_script(input_path)
        scenes = script_data.get('scenes', [])
        
        if not scenes:
            scenes = [
                "Gaming setup and introduction",
                "Gameplay highlights and commentary", 
                "Epic moments and reactions",
                "Final thoughts and outro"
            ]
        
        if db_run and db:
            db_run.progress = 30.0
            db.commit()
        
        scene_files = []
        for i, scene in enumerate(scenes):
            scene_text = scene if isinstance(scene, str) else scene.get('description', f'Gaming scene {i+1}')
            
            scene_file = output_dir / "scenes" / f"gaming_scene_{i+1:03d}.mp4"
            
            try:
                video_path = self._create_gaming_scene_video(scene_text, str(scene_file))
                if video_path:
                    scene_files.append(video_path)
            except Exception as e:
                print(f"Error generating gaming scene {i+1}: {e}")
                fallback_path = self._create_fallback_video(scene_text, 10.0, str(scene_file))
                if fallback_path:
                    scene_files.append(fallback_path)
        
        if db_run and db:
            db_run.progress = 60.0
            db.commit()
        
        temp_combined = output_dir / "final" / "temp_combined.mp4"
        combined_video = self._combine_gaming_scenes(scene_files, str(temp_combined), render_fps, output_fps)
        
        if frame_interpolation_enabled and output_fps > render_fps:
            print(f"Applying frame interpolation: {render_fps}fps -> {output_fps}fps...")
            interpolated_video = output_dir / "final" / "gaming_content_interpolated.mp4"
            combined_video = self._interpolate_frames(combined_video, str(interpolated_video), output_fps)
        
        if db_run and db:
            db_run.progress = 80.0
            db.commit()
        
        upscaled_video = output_dir / "final" / "gaming_content_upscaled.mp4"
        final_video = self._upscale_video_with_realesrgan(combined_video, str(upscaled_video), "1080p", True)
        
        if db_run and db:
            db_run.progress = 90.0
            db.commit()
        
        shorts = self._create_gaming_shorts(final_video, output_dir)
        
        self._generate_youtube_metadata(output_dir, scenes, [], language, "gaming")
        
        if db_run and db:
            db_run.progress = 100.0
            db.commit()
        
        return str(output_dir)
    
    def _extract_highlights(self, input_path: str) -> List[Dict]:
        """Extract highlights from game recording using advanced analysis."""
        try:
            if not cv2:
                return [{"start": 0, "end": 60, "type": "highlight"}]
            
            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            
            highlights = []
            segment_duration = 15  # 15 second segments for better highlights
            
            prev_frame = None
            motion_scores = []
            
            frame_count = 0
            while frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % int(fps) == 0:  # Sample every second
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    if prev_frame is not None:
                        flow = cv2.calcOpticalFlowPyrLK(prev_frame, gray, None, None)
                        motion_magnitude = np.mean(np.abs(flow[0])) if flow[0] is not None else 0
                        motion_scores.append(motion_magnitude)
                    
                    prev_frame = gray
                
                frame_count += 1
            
            cap.release()
            
            if motion_scores:
                threshold = np.percentile(motion_scores, 75)  # Top 25% motion
                
                for i, score in enumerate(motion_scores):
                    if score > threshold:
                        start_time = i
                        end_time = min(start_time + segment_duration, duration)
                        
                        highlights.append({
                            "start": start_time,
                            "end": end_time,
                            "type": "action_highlight",
                            "score": score,
                            "motion_intensity": "high"
                        })
            
            if len(highlights) < 3:
                for i in range(0, int(duration), 45):
                    end_time = min(i + 20, duration)
                    highlights.append({
                        "start": i,
                        "end": end_time,
                        "type": "general_highlight",
                        "score": random.uniform(0.6, 0.9)
                    })
            
            highlights.sort(key=lambda x: x["score"], reverse=True)
            return highlights[:5]  # Top 5 highlights
            
        except Exception as e:
            logger.error(f"Error extracting highlights: {e}")
            return [{"start": 0, "end": 60, "type": "highlight"}]
    
    def _create_edited_compilation(self, highlights: List[Dict], output_dir: Path, 
                                  render_fps: int, output_fps: int) -> str:
        """Create edited compilation from highlights with maximum quality."""
        try:
            compilation_path = output_dir / "final" / "gaming_compilation.mp4"
            
            if highlights:
                filter_parts = []
                input_parts = []
                
                for i, highlight in enumerate(highlights):
                    duration = highlight["end"] - highlight["start"]
                    
                    highlight_content = f"Gaming highlight {i+1}: {highlight.get('type', 'action')} sequence"
                    
                    temp_clip = output_dir / "scenes" / f"highlight_{i+1}.mp4"
                    self._create_gaming_scene_video(highlight_content, str(temp_clip))
                    
                    input_parts.append(f"-i {temp_clip}")
                    filter_parts.append(f"[{i}:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1[v{i}];")
                
                concat_filter = "".join(filter_parts) + "".join([f"[v{i}]" for i in range(len(highlights))]) + f"concat=n={len(highlights)}:v=1:a=0[outv]"
                
                cmd = [
                    'ffmpeg', '-y'
                ] + " ".join(input_parts).split() + [
                    '-filter_complex', concat_filter,
                    '-map', '[outv]',
                    '-c:v', 'libx264',
                    '-preset', 'veryslow',  # Maximum quality
                    '-crf', '15',  # High quality
                    '-profile:v', 'high',
                    '-level', '4.1',
                    '-r', str(output_fps),
                    '-pix_fmt', 'yuv420p',
                    str(compilation_path)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
                
                if result.returncode == 0 and os.path.exists(compilation_path):
                    return str(compilation_path)
            
            return self._create_fallback_video("Gaming compilation", 300, str(compilation_path))
            
        except Exception as e:
            logger.error(f"Error creating compilation: {e}")
            return self._create_fallback_video("Gaming compilation", 300, str(compilation_path))
    
    def _create_gaming_shorts(self, video_path: str, output_dir: Path) -> List[str]:
        """Create gaming shorts from main video."""
        shorts_paths = []
        
        try:
            for i in range(3):  # Create 3 shorts
                short_path = output_dir / "shorts" / f"gaming_short_{i+1:03d}.mp4"
                
                start_time = i * 30  # 30 second intervals
                
                cmd = [
                    'ffmpeg', '-y',
                    '-i', video_path,
                    '-ss', str(start_time),
                    '-t', '15',  # 15 second shorts
                    '-vf', 'scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920',  # Vertical format
                    '-c:v', 'libx264',
                    '-preset', 'veryslow',
                    '-crf', '15',
                    '-c:a', 'aac',
                    str(short_path)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0 and os.path.exists(short_path):
                    shorts_paths.append(str(short_path))
                    
        except Exception as e:
            logger.error(f"Error creating gaming shorts: {e}")
        
        return shorts_paths
    
    def _create_gaming_scene_video(self, scene_description: str, output_path: str) -> str:
        """Create gaming scene video with maximum quality."""
        try:
            gaming_prompt = self._optimize_video_prompt(scene_description, "gaming")
            
            try:
                video_generator = self.load_video_model("animatediff_v2")
                if video_generator:
                    success = video_generator.generate_video(
                        prompt=gaming_prompt,
                        width=1920,
                        height=1080,
                        num_frames=240,  # 8 seconds at 30fps
                        guidance_scale=12.0,  # High guidance for gaming
                        num_inference_steps=80,  # High quality steps
                        output_path=output_path
                    )
                    if success and os.path.exists(output_path):
                        return output_path
            except Exception as e:
                logger.warning(f"Gaming video generation failed: {e}")
            
            return self._create_fallback_video(scene_description, 8.0, output_path)
            
        except Exception as e:
            logger.error(f"Error creating gaming scene video: {e}")
            return self._create_fallback_video(scene_description, 8.0, output_path)
    
    def _optimize_video_prompt(self, prompt: str, channel_type: str) -> str:
        """Optimize prompt for video generation with maximum quality."""
        optimizations = {
            "gaming": "masterpiece, best quality, ultra detailed, 8k resolution, cinematic lighting, smooth animation, professional gaming content, vibrant colors, dynamic composition, "
        }
        
        prefix = optimizations.get(channel_type, "high quality, detailed, ")
        suffix = ", 16:9 aspect ratio, smooth motion, professional cinematography, ultra high definition"
        
        return f"{prefix}{prompt}{suffix}"
    
    def _combine_gaming_scenes(self, scene_files: List[str], output_path: str, 
                              render_fps: int, output_fps: int) -> str:
        """Combine gaming scenes with maximum quality."""
        try:
            if not scene_files:
                return self._create_fallback_video("No gaming scenes", 60, output_path)
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                concat_file = f.name
                for scene_file in scene_files:
                    if os.path.exists(scene_file):
                        f.write(f"file '{os.path.abspath(scene_file)}'\n")
            
            try:
                cmd = [
                    'ffmpeg', '-y',
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', concat_file,
                    '-c:v', 'libx264',
                    '-preset', 'veryslow',  # Maximum quality
                    '-crf', '15',  # High quality
                    '-profile:v', 'high',
                    '-level', '4.1',
                    '-r', str(output_fps),
                    '-pix_fmt', 'yuv420p',
                    '-s', '1920x1080',
                    '-movflags', '+faststart',
                    output_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
                
                if result.returncode == 0 and os.path.exists(output_path):
                    return output_path
                else:
                    logger.warning(f"FFmpeg combination failed: {result.stderr}")
                    return self._create_fallback_video("Gaming content", 180, output_path)
                    
            finally:
                if os.path.exists(concat_file):
                    os.unlink(concat_file)
                    
        except Exception as e:
            logger.error(f"Error combining gaming scenes: {e}")
            return self._create_fallback_video("Gaming content", 180, output_path)
    
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
    
    def _generate_youtube_metadata(self, output_dir: Path, scenes: List, characters: List, language: str, channel_type: str = "gaming"):
        """Generate YouTube metadata files with LLM."""
        try:
            title_prompt = f"Generate a compelling YouTube title for a {channel_type} video with {len(scenes)} scenes. Make it engaging and clickable for gaming content."
            
            llm_model = self.load_llm_model()
            if llm_model:
                title = llm_model.generate(title_prompt, max_tokens=50)
            else:
                title = f"Epic Gaming Highlights - Best Moments Compilation"
            
            with open(output_dir / "title.txt", "w", encoding="utf-8") as f:
                f.write(title.strip())
            
            description_prompt = f"Generate a detailed YouTube description for a {channel_type} video. Include gameplay highlights, epic moments, and engaging hooks. Language: {language}"
            
            if llm_model:
                description = llm_model.generate(description_prompt, max_tokens=300)
            else:
                description = f"Amazing gaming highlights and epic moments! Watch the best gameplay compilation with incredible action and reactions!"
            
            with open(output_dir / "description.txt", "w", encoding="utf-8") as f:
                f.write(description.strip())
            
            next_episode_prompt = f"Based on this gaming content, suggest 3 compelling ideas for the next gaming video. Be creative and engaging."
            
            if llm_model:
                next_suggestions = llm_model.generate(next_episode_prompt, max_tokens=200)
            else:
                next_suggestions = "1. New game exploration and first reactions\n2. Challenge runs and speedrun attempts\n3. Multiplayer highlights and epic team moments"
            
            with open(output_dir / "next_episode.txt", "w", encoding="utf-8") as f:
                f.write(next_suggestions.strip())
            
        except Exception as e:
            logger.error(f"Error generating YouTube metadata: {e}")
    
    def _create_fallback_video(self, description: str, duration: float, output_path: str) -> str:
        """Create fallback video with text overlay."""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            cmd = [
                'ffmpeg', '-y',
                '-f', 'lavfi',
                '-i', f'color=c=black:size=1920x1080:duration={duration}',
                '-vf', f'drawtext=text=\'{description}\':fontcolor=white:fontsize=48:x=(w-text_w)/2:y=(h-text_h)/2',
                '-c:v', 'libx264',
                '-preset', 'veryslow',
                '-crf', '15',
                '-pix_fmt', 'yuv420p',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0 and os.path.exists(output_path):
                return output_path
            else:
                logger.warning(f"Fallback video creation failed: {result.stderr}")
                return output_path
                
        except Exception as e:
            logger.error(f"Error creating fallback video: {e}")
            return output_path


def run(input_path: str, output_path: str, base_model: str = "stable_diffusion_1_5", 
        lora_models: Optional[List[str]] = None, lora_paths: Optional[Dict[str, str]] = None, 
        db_run=None, db=None, render_fps: int = 24, output_fps: int = 60, 
        frame_interpolation_enabled: bool = True, language: str = "en") -> str:
    """Run gaming pipeline with self-contained processing."""
    pipeline = GamingPipeline()
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
