"""
Channel-specific pipeline mapping module.
"""
from pathlib import Path

from . import gaming_pipeline
from . import anime_pipeline
from . import marvel_dc_pipeline
from . import manga_pipeline
from . import superhero_pipeline
from . import original_manga_pipeline

CHANNEL_PIPELINES = {
    "gaming": gaming_pipeline.GamingPipeline,
    "anime": anime_pipeline.AnimePipeline,
    "marvel_dc": marvel_dc_pipeline.MarvelDCPipeline,
    "manga": manga_pipeline.MangaPipeline,
    "superhero": superhero_pipeline.SuperheroPipeline,
    "original_manga": original_manga_pipeline.OriginalMangaPipeline
}

def get_pipeline_for_channel(channel_type):
    """
    Get the appropriate pipeline module for a given channel type.
    
    Args:
        channel_type: The channel type (gaming, anime, etc.)
        
    Returns:
        The pipeline module for the specified channel type, or None if not found.
    """
    pipeline_mapping = {
        "gaming": gaming_pipeline,
        "anime": anime_pipeline,
        "marvel_dc": marvel_dc_pipeline,
        "manga": manga_pipeline,
        "superhero": superhero_pipeline,
        "original_manga": original_manga_pipeline
    }
    
    return pipeline_mapping.get(channel_type.lower(), None)
