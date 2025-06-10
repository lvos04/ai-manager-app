"""
Duration parsing utilities to handle various duration formats and prevent string concatenation bugs.
"""

import re
import logging
from typing import Union, Any

logger = logging.getLogger(__name__)

def parse_duration(duration: Any, default: float = 10.0) -> float:
    """
    Parse duration from various formats and ensure it's a valid float.
    
    Args:
        duration: Duration value in various formats (float, int, str, etc.)
        default: Default value to return if parsing fails
        
    Returns:
        float: Parsed duration value
    """
    if isinstance(duration, (int, float)):
        return max(float(duration), 0.1)  # Minimum 0.1 seconds
    
    if isinstance(duration, str):
        try:
            if "seconds" in duration:
                duration_clean = duration.split("seconds")[0].strip()
            else:
                duration_clean = duration.strip()
            
            duration_clean = re.sub(r'[^\d.]', '', duration_clean)
            
            if duration_clean:
                parsed = float(duration_clean)
                return max(parsed, 0.1)  # Minimum 0.1 seconds
            else:
                logger.warning(f"Empty duration string after cleaning: '{duration}'")
                return default
                
        except (ValueError, AttributeError, IndexError) as e:
            logger.warning(f"Failed to parse duration '{duration}': {e}")
            return default
    
    try:
        return max(float(duration), 0.1)
    except (ValueError, TypeError):
        logger.warning(f"Cannot convert duration to float: {duration} (type: {type(duration)})")
        return default

def validate_scene_durations(scenes: list) -> list:
    """
    Validate and fix duration values in a list of scenes.
    
    Args:
        scenes: List of scene dictionaries or strings
        
    Returns:
        list: Scenes with validated duration values
    """
    validated_scenes = []
    
    for i, scene in enumerate(scenes):
        if isinstance(scene, dict):
            scene_copy = scene.copy()
            if "duration" in scene_copy:
                scene_copy["duration"] = parse_duration(scene_copy["duration"])
            else:
                scene_copy["duration"] = 10.0
            validated_scenes.append(scene_copy)
        else:
            validated_scenes.append({
                "description": str(scene),
                "duration": 10.0,
                "scene_id": str(i)
            })
    
    return validated_scenes

def calculate_total_duration(scenes: list) -> float:
    """
    Calculate total duration from a list of scenes with proper error handling.
    
    Args:
        scenes: List of scene dictionaries
        
    Returns:
        float: Total duration in seconds
    """
    total = 0.0
    
    for scene in scenes:
        if isinstance(scene, dict):
            duration = parse_duration(scene.get("duration", 10.0))
        else:
            duration = 10.0  # Default for string scenes
        
        total += duration
    
    return total
