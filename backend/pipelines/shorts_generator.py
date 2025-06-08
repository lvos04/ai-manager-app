"""
AI-powered shorts generator for creating engaging short-form content.
"""
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class ShortsGenerator:
    """Generate engaging shorts from longer content."""
    
    def __init__(self, vram_tier: str = "medium"):
        self.vram_tier = vram_tier
        self.loaded_models = {}
    
    def generate_shorts(self, input_video_path: str, output_dir: str, num_shorts: int = 5) -> List[Dict]:
        """
        Generate multiple shorts from input video.
        
        Args:
            input_video_path: Path to input video
            output_dir: Directory to save shorts
            num_shorts: Number of shorts to generate
            
        Returns:
            List of generated shorts with metadata
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            highlights = self.extract_highlights(input_video_path)
            
            shorts = []
            for i, highlight in enumerate(highlights[:num_shorts]):
                short_path = output_path / f"short_{i+1:02d}.mp4"
                
                short_data = self.create_short(
                    input_video_path,
                    highlight,
                    str(short_path),
                    i + 1
                )
                
                if short_data:
                    shorts.append(short_data)
            
            return shorts
            
        except Exception as e:
            logger.error(f"Error generating shorts: {e}")
            return []
    
    def extract_highlights(self, video_path: str) -> List[Dict]:
        """Extract highlight moments from video."""
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            
            highlights = []
            
            segment_duration = 60
            num_segments = int(duration // segment_duration)
            
            for i in range(num_segments):
                start_time = i * segment_duration
                end_time = min((i + 1) * segment_duration, duration)
                
                excitement_score = self.calculate_segment_excitement(
                    video_path, start_time, end_time
                )
                
                highlights.append({
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": end_time - start_time,
                    "excitement_score": excitement_score,
                    "type": self.classify_segment_type(excitement_score)
                })
            
            highlights.sort(key=lambda x: x["excitement_score"], reverse=True)
            cap.release()
            
            return highlights
            
        except Exception as e:
            logger.error(f"Error extracting highlights: {e}")
            return []
    
    def calculate_segment_excitement(self, video_path: str, start_time: float, end_time: float) -> float:
        """Calculate excitement score for a video segment."""
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
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
            
            cap.release()
            
            if motion_scores:
                avg_motion = np.mean(motion_scores)
                max_motion = np.max(motion_scores)
                excitement = (avg_motion * 0.7 + max_motion * 0.3) / 255.0
                return min(excitement, 1.0)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating excitement: {e}")
            return 0.0
    
    def classify_segment_type(self, excitement_score: float) -> str:
        """Classify segment type based on excitement score."""
        if excitement_score > 0.7:
            return "high_action"
        elif excitement_score > 0.4:
            return "medium_action"
        else:
            return "low_action"
    
    def create_short(self, input_video: str, highlight: Dict, output_path: str, short_number: int) -> Optional[Dict]:
        """Create a single short from highlight data."""
        try:
            from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
            
            start_time = highlight["start_time"]
            duration = min(highlight["duration"], 60)
            
            clip = VideoFileClip(input_video).subclip(start_time, start_time + duration)
            
            title = self.generate_short_title(highlight, short_number)
            
            title_clip = TextClip(title, fontsize=40, color='white', bg_color='black')
            title_clip = title_clip.set_duration(3).set_position(('center', 'top'))
            
            final_clip = CompositeVideoClip([clip, title_clip])
            
            final_clip.write_videofile(
                output_path,
                codec='libx264',
                fps=30,
                preset='medium'
            )
            
            clip.close()
            title_clip.close()
            final_clip.close()
            
            return {
                "path": output_path,
                "title": title,
                "duration": duration,
                "type": highlight["type"],
                "excitement_score": highlight["excitement_score"]
            }
            
        except Exception as e:
            logger.error(f"Error creating short: {e}")
            return None
    
    def generate_short_title(self, highlight: Dict, short_number: int) -> str:
        """Generate engaging title for short."""
        segment_type = highlight["type"]
        excitement = highlight["excitement_score"]
        
        if segment_type == "high_action":
            titles = [
                f"Epic Moment #{short_number}!",
                f"Insane Action #{short_number}",
                f"Mind-Blowing #{short_number}",
                f"Incredible Play #{short_number}"
            ]
        elif segment_type == "medium_action":
            titles = [
                f"Great Moment #{short_number}",
                f"Nice Play #{short_number}",
                f"Cool Scene #{short_number}",
                f"Solid Gameplay #{short_number}"
            ]
        else:
            titles = [
                f"Highlight #{short_number}",
                f"Moment #{short_number}",
                f"Scene #{short_number}",
                f"Clip #{short_number}"
            ]
        
        import random
        return random.choice(titles)

def generate_shorts_from_video(input_video: str, output_dir: str, num_shorts: int = 5) -> List[Dict]:
    """Generate shorts from input video."""
    generator = ShortsGenerator()
    return generator.generate_shorts(input_video, output_dir, num_shorts)
