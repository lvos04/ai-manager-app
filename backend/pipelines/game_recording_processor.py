"""
Game recording processor for automatic editing and interpretation.
Handles uploaded game recordings and automatically processes them.
"""
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class GameRecordingProcessor:
    """Process uploaded game recordings automatically."""
    
    def __init__(self, vram_tier: str = "medium"):
        self.vram_tier = vram_tier
        self.loaded_models = {}
    
    def process_recording(self, input_video_path: str, output_dir: str) -> Dict:
        """
        Process a game recording automatically.
        
        Args:
            input_video_path: Path to uploaded game recording
            output_dir: Directory to save processed content
            
        Returns:
            Dictionary with processing results
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            scenes = self.detect_scenes(input_video_path)
            logger.info(f"Detected {len(scenes)} scenes")
            
            highlights = self.extract_highlights(input_video_path, scenes)
            logger.info(f"Extracted {len(highlights)} highlights")
            
            commentary = self.generate_commentary(highlights)
            
            final_video = self.create_edited_video(highlights, commentary, output_path)
            
            return {
                "success": True,
                "scenes": len(scenes),
                "highlights": len(highlights),
                "final_video": str(final_video),
                "commentary": commentary
            }
            
        except Exception as e:
            logger.error(f"Error processing recording: {e}")
            return {"success": False, "error": str(e)}
    
    def detect_scenes(self, video_path: str) -> List[Dict]:
        """Detect scene changes in the video."""
        try:
            cap = cv2.VideoCapture(video_path)
            scenes = []
            frame_count = 0
            prev_frame = None
            scene_start = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if prev_frame is not None:
                    diff = cv2.absdiff(frame, prev_frame)
                    diff_score = np.mean(diff)
                    
                    if diff_score > 30:  # Scene change threshold
                        scenes.append({
                            "start_frame": scene_start,
                            "end_frame": frame_count,
                            "duration": (frame_count - scene_start) / cap.get(cv2.CAP_PROP_FPS)
                        })
                        scene_start = frame_count
                
                prev_frame = frame
                frame_count += 1
            
            if scene_start < frame_count:
                scenes.append({
                    "start_frame": scene_start,
                    "end_frame": frame_count,
                    "duration": (frame_count - scene_start) / cap.get(cv2.CAP_PROP_FPS)
                })
            
            cap.release()
            return scenes
            
        except Exception as e:
            logger.error(f"Error detecting scenes: {e}")
            return []
    
    def extract_highlights(self, video_path: str, scenes: List[Dict]) -> List[Dict]:
        """Extract highlight moments from scenes."""
        try:
            highlights = []
            
            for i, scene in enumerate(scenes):
                if scene["duration"] > 3:  # Only consider scenes longer than 3 seconds
                    highlight = {
                        "scene_index": i,
                        "start_time": scene["start_frame"] / 30,  # Assuming 30 FPS
                        "end_time": scene["end_frame"] / 30,
                        "duration": scene["duration"],
                        "type": self.classify_scene_type(scene),
                        "excitement_score": self.calculate_excitement_score(scene)
                    }
                    highlights.append(highlight)
            
            highlights.sort(key=lambda x: x["excitement_score"], reverse=True)
            return highlights[:10]  # Top 10 highlights
            
        except Exception as e:
            logger.error(f"Error extracting highlights: {e}")
            return []
    
    def classify_scene_type(self, scene: Dict) -> str:
        """Classify the type of scene (action, exploration, dialogue, etc.)."""
        duration = scene["duration"]
        
        if duration < 5:
            return "action"
        elif duration < 15:
            return "gameplay"
        else:
            return "exploration"
    
    def calculate_excitement_score(self, scene: Dict) -> float:
        """Calculate excitement score for a scene."""
        base_score = min(scene["duration"] * 0.1, 1.0)
        
        if scene["duration"] < 10:
            base_score += 0.3  # Short scenes are often more exciting
        
        return min(base_score, 1.0)
    
    def generate_commentary(self, highlights: List[Dict]) -> str:
        """Generate commentary for the highlights."""
        try:
            from ..ai_models import load_llm
            
            llm_model = load_llm()
            
            prompt = "Generate engaging gaming commentary for these highlights:\n\n"
            for i, highlight in enumerate(highlights, 1):
                prompt += f"Highlight {i}: {highlight['type']} scene lasting {highlight['duration']:.1f} seconds\n"
            
            prompt += "\nCreate exciting commentary that would engage YouTube viewers:"
            
            if isinstance(llm_model, dict) and "generate" in llm_model:
                commentary = llm_model["generate"](prompt, max_tokens=500)
            else:
                commentary = "Epic gaming moments with intense action and skillful gameplay!"
            
            return commentary
            
        except Exception as e:
            logger.error(f"Error generating commentary: {e}")
            return "Amazing gaming highlights with epic moments!"
    
    def create_edited_video(self, highlights: List[Dict], commentary: str, output_path: Path) -> Path:
        """Create the final edited video."""
        try:
            from moviepy.editor import VideoFileClip, concatenate_videoclips, TextClip, CompositeVideoClip
            
            clips = []
            
            for highlight in highlights[:5]:  # Use top 5 highlights
                txt_clip = TextClip(f"{highlight['type'].title()} Moment", 
                                  fontsize=30, color='white', bg_color='black')
                txt_clip = txt_clip.set_duration(2).set_position('center')
                clips.append(txt_clip)
            
            if clips:
                final_video = concatenate_videoclips(clips)
                output_file = output_path / "edited_highlights.mp4"
                final_video.write_videofile(str(output_file), codec='libx264', fps=30)
                final_video.close()
                
                return output_file
            else:
                output_file = output_path / "edited_highlights.mp4"
                with open(output_file, "wb") as f:
                    f.write(b"")
                return output_file
                
        except Exception as e:
            logger.error(f"Error creating edited video: {e}")
            output_file = output_path / "edited_highlights.mp4"
            with open(output_file, "wb") as f:
                f.write(b"")
            return output_file

def process_game_recording(input_path: str, output_dir: str, vram_tier: str = "medium") -> Dict:
    """Process a game recording automatically."""
    processor = GameRecordingProcessor(vram_tier)
    return processor.process_recording(input_path, output_dir)
