"""
AI-powered shorts generator for creating engaging short-form content.
Integrates with LLM models for intelligent content creation.
"""
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class AIShortsGenerator:
    """Generate engaging shorts using AI models."""
    
    def __init__(self, vram_tier: str = "medium"):
        self.vram_tier = vram_tier
        self.loaded_models = {}
    
    def generate_ai_shorts(self, content_description: str, output_dir: str, num_shorts: int = 5) -> List[Dict]:
        """
        Generate multiple shorts using AI content generation.
        
        Args:
            content_description: Description of content to generate
            output_dir: Directory to save shorts
            num_shorts: Number of shorts to generate
            
        Returns:
            List of generated shorts with metadata
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            shorts = []
            for i in range(num_shorts):
                short_prompt = self.generate_short_prompt(content_description, i + 1)
                
                short_path = output_path / f"ai_short_{i+1:02d}.mp4"
                
                short_data = self.create_ai_short(
                    short_prompt,
                    str(short_path),
                    i + 1
                )
                
                if short_data:
                    shorts.append(short_data)
            
            return shorts
            
        except Exception as e:
            logger.error(f"Error generating AI shorts: {e}")
            return []
    
    def generate_short_prompt(self, content_description: str, short_number: int) -> str:
        """Generate optimized prompt for short video creation."""
        try:
            from .ai_models import load_llm
            
            llm_model = load_llm()
            
            prompt = f"""
            Create an engaging short video concept based on: {content_description}
            
            Short #{short_number} should be:
            - 30-60 seconds long
            - Visually striking and attention-grabbing
            - Perfect for social media platforms
            - Include dynamic camera movements
            - Have clear visual storytelling
            
            Generate a detailed scene description for this short:
            """
            
            if isinstance(llm_model, dict) and "generate" in llm_model:
                optimized_prompt = llm_model["generate"](prompt, max_tokens=200)
            else:
                optimized_prompt = f"Dynamic short video: {content_description}, cinematic style, engaging visuals"
            
            return optimized_prompt
            
        except Exception as e:
            logger.error(f"Error generating short prompt: {e}")
            return f"Engaging short video: {content_description}"
    
    def create_ai_short(self, prompt: str, output_path: str, short_number: int) -> Optional[Dict]:
        """Create a single AI-generated short using latest video models."""
        try:
            from .video_generation import TextToVideoGenerator, get_best_model_for_content
            from .pipeline_utils import optimize_video_prompt
            
            optimized_prompt = optimize_video_prompt(prompt, "shorts")
            
            generator = TextToVideoGenerator(self.vram_tier, (1080, 1920))
            best_model = get_best_model_for_content("shorts", self.vram_tier)
            
            success = generator.generate_video(
                optimized_prompt,
                best_model,
                output_path
            )
            
            if success:
                title = self.generate_short_title(prompt, short_number)
                
                return {
                    "path": output_path,
                    "title": title,
                    "prompt": optimized_prompt,
                    "type": "ai_generated",
                    "model": best_model,
                    "duration": 45
                }
            else:
                self.create_fallback_short(output_path, prompt, short_number)
                return {
                    "path": output_path,
                    "title": f"Short #{short_number}",
                    "prompt": prompt,
                    "type": "fallback",
                    "duration": 5
                }
            
        except Exception as e:
            logger.error(f"Error creating AI short: {e}")
            return None
    
    def generate_short_title(self, prompt: str, short_number: int) -> str:
        """Generate engaging title for short."""
        try:
            from .ai_models import load_llm
            
            llm_model = load_llm()
            
            title_prompt = f"""
            Create a catchy, engaging title for this short video:
            {prompt[:100]}...
            
            Title should be:
            - Under 60 characters
            - Attention-grabbing
            - Perfect for social media
            
            Title:
            """
            
            if isinstance(llm_model, dict) and "generate" in llm_model:
                title = llm_model["generate"](title_prompt, max_tokens=20)
                if title and len(title.strip()) > 5:
                    return title.strip()
            
            return f"Amazing Short #{short_number}"
            
        except Exception as e:
            logger.error(f"Error generating title: {e}")
            return f"Epic Short #{short_number}"
    
    def create_fallback_short(self, output_path: str, prompt: str, short_number: int):
        """Create fallback short when AI generation fails."""
        try:
            from moviepy.editor import ColorClip, TextClip, CompositeVideoClip
            
            bg_clip = ColorClip(size=(1080, 1920), color=(30, 30, 30), duration=5)
            
            title_text = f"Short #{short_number}"
            title_clip = TextClip(title_text, fontsize=60, color='white', bg_color='black')
            title_clip = title_clip.set_duration(2).set_position(('center', 'top'))
            
            desc_text = prompt[:100] + "..."
            desc_clip = TextClip(desc_text, fontsize=30, color='white', 
                               size=(1000, None), method='caption')
            desc_clip = desc_clip.set_duration(5).set_position('center')
            
            final_clip = CompositeVideoClip([bg_clip, title_clip, desc_clip])
            final_clip.write_videofile(output_path, codec='libx264', fps=30)
            
            bg_clip.close()
            title_clip.close()
            desc_clip.close()
            final_clip.close()
            
            logger.info(f"Created fallback short: {output_path}")
            
        except Exception as e:
            logger.error(f"Error creating fallback short: {e}")
            with open(output_path, "wb") as f:
                f.write(b"")

def generate_ai_shorts(content_description: str, output_dir: str, num_shorts: int = 5, vram_tier: str = "medium") -> List[Dict]:
    """Generate AI-powered shorts from content description."""
    generator = AIShortsGenerator(vram_tier)
    return generator.generate_ai_shorts(content_description, output_dir, num_shorts)
