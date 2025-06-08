"""
Pipeline modules for AI Project Manager.
"""

import logging
logger = logging.getLogger(__name__)

pipeline_utils = None

try:
    from .channel_specific import get_pipeline_for_channel
except ImportError:
    def get_pipeline_for_channel(channel_type):
        return None

__all__ = []
