"""
Pipeline modules for AI Project Manager.
"""

import logging
logger = logging.getLogger(__name__)

try:
    from . import pipeline_utils
    logger.info("Successfully imported pipeline_utils")
except ImportError as e:
    logger.warning(f"Could not import pipeline_utils: {e}")
    pipeline_utils = None

try:
    from .channel_specific import get_pipeline_for_channel
except ImportError:
    def get_pipeline_for_channel(channel_type):
        return None

__all__ = ['pipeline_utils']
