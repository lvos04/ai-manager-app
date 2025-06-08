"""
Adaptive Quality & Performance Scaling for AI Project Manager.
"""

import logging
import psutil
import torch
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class AdaptiveQualityManager:
    """Dynamic quality adjustment based on hardware performance."""
    
    def __init__(self):
        self.performance_monitor = self._create_performance_monitor()
        self.quality_presets = self._load_quality_presets()
        self.current_settings = {}
        
    def _create_performance_monitor(self):
        """Create performance monitoring system."""
        return {
            'cpu_percent': 0,
            'memory_percent': 0,
            'gpu_memory_percent': 0,
            'processing_time': 0
        }
        
    def _load_quality_presets(self):
        """Load quality presets for different performance levels."""
        return {
            'ultra': {
                'resolution_scale': 1.0,
                'batch_size': 8,
                'steps': 50,
                'guidance_scale': 7.5
            },
            'high': {
                'resolution_scale': 0.8,
                'batch_size': 4,
                'steps': 30,
                'guidance_scale': 7.0
            },
            'medium': {
                'resolution_scale': 0.6,
                'batch_size': 2,
                'steps': 20,
                'guidance_scale': 6.5
            },
            'low': {
                'resolution_scale': 0.4,
                'batch_size': 1,
                'steps': 15,
                'guidance_scale': 6.0
            }
        }
        
    def adjust_quality_for_performance(self, target_fps: float = 24.0, current_performance: Dict[str, float] = None) -> Dict[str, Any]:
        """Dynamically adjust quality settings for target performance."""
        if current_performance is None:
            current_performance = self._measure_current_performance()
            
        if current_performance['gpu_memory_percent'] > 90 or current_performance['processing_time'] > 10:
            quality_level = 'low'
        elif current_performance['gpu_memory_percent'] > 70 or current_performance['processing_time'] > 5:
            quality_level = 'medium'
        elif current_performance['gpu_memory_percent'] > 50:
            quality_level = 'high'
        else:
            quality_level = 'ultra'
            
        settings = self.quality_presets[quality_level].copy()
        logger.info(f"Adaptive quality: Using {quality_level} settings for performance optimization")
        
        return settings
        
    def get_optimal_settings(self, hardware_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Get optimal settings for specific hardware."""
        vram_gb = hardware_profile.get('vram_gb', 4)
        cpu_cores = hardware_profile.get('cpu_cores', 4)
        
        if vram_gb >= 16:
            return self.quality_presets['ultra']
        elif vram_gb >= 8:
            return self.quality_presets['high']
        elif vram_gb >= 4:
            return self.quality_presets['medium']
        else:
            return self.quality_presets['low']
            
    def _measure_current_performance(self) -> Dict[str, float]:
        """Measure current system performance."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            gpu_memory_percent = 0
            if torch.cuda.is_available():
                try:
                    allocated = torch.cuda.memory_allocated()
                    total = torch.cuda.get_device_properties(0).total_memory
                    gpu_memory_percent = (allocated / total) * 100
                except:
                    gpu_memory_percent = 0
                
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'gpu_memory_percent': gpu_memory_percent,
                'processing_time': 0
            }
        except Exception as e:
            logger.warning(f"Performance measurement failed: {e}")
            return self.performance_monitor

_adaptive_quality_manager = None

def get_adaptive_quality_manager():
    """Get global adaptive quality manager instance."""
    global _adaptive_quality_manager
    if _adaptive_quality_manager is None:
        _adaptive_quality_manager = AdaptiveQualityManager()
    return _adaptive_quality_manager
