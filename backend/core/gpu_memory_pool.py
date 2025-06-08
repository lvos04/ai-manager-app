"""
GPU Memory Pool Management for AI Project Manager
Advanced GPU memory management to eliminate fragmentation and OOM errors.
"""

import gc
import logging
import threading
import time
import weakref
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
import psutil

logger = logging.getLogger(__name__)

class MemoryPriority(Enum):
    """Memory allocation priorities."""
    CRITICAL = 1    # Must stay in memory
    HIGH = 2        # Important to keep
    MEDIUM = 3      # Can be evicted if needed
    LOW = 4         # First to be evicted

@dataclass
class MemorySegment:
    """Represents a GPU memory segment."""
    model_name: str
    size_mb: float
    priority: MemoryPriority
    last_accessed: float
    access_count: int
    model_ref: Any
    
class GPUMemoryMonitor:
    """Monitors GPU memory usage and availability."""
    
    def __init__(self):
        self.gpu_available = False
        self.total_memory_mb = 0
        self.used_memory_mb = 0
        self.free_memory_mb = 0
        self.utilization_percent = 0
        self.last_update = 0
        self.update_interval = 1.0  # Update every second
        
        self._detect_gpu()
        
    def _detect_gpu(self):
        """Detect GPU and initialize memory monitoring."""
        try:
            import torch
            if torch.cuda.is_available():
                self.gpu_available = True
                device_props = torch.cuda.get_device_properties(0)
                self.total_memory_mb = device_props.total_memory / (1024 * 1024)
                logger.info(f"GPU detected: {device_props.name}, {self.total_memory_mb:.0f}MB")
            else:
                logger.warning("No CUDA GPU available")
        except ImportError:
            logger.warning("PyTorch not available for GPU monitoring")
        except Exception as e:
            logger.error(f"GPU detection failed: {e}")
            
    def update_memory_stats(self):
        """Update current memory statistics."""
        if not self.gpu_available:
            return
            
        current_time = time.time()
        if current_time - self.last_update < self.update_interval:
            return
            
        try:
            import torch
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(0) / (1024 * 1024)
                memory_reserved = torch.cuda.memory_reserved(0) / (1024 * 1024)
                
                self.used_memory_mb = memory_reserved
                self.free_memory_mb = self.total_memory_mb - memory_reserved
                self.utilization_percent = (memory_reserved / self.total_memory_mb) * 100
                
                self.last_update = current_time
                
        except Exception as e:
            logger.error(f"Failed to update GPU memory stats: {e}")
            
    def get_memory_pressure(self) -> float:
        """Get current memory pressure (0.0 = no pressure, 1.0 = critical)."""
        self.update_memory_stats()
        if not self.gpu_available:
            return 0.0
        return min(1.0, self.utilization_percent / 100.0)
        
    def get_available_memory_mb(self) -> float:
        """Get available GPU memory in MB."""
        self.update_memory_stats()
        return self.free_memory_mb if self.gpu_available else 0.0
        
    def can_allocate(self, size_mb: float, safety_margin: float = 0.1) -> bool:
        """Check if we can allocate the requested memory size."""
        available = self.get_available_memory_mb()
        required = size_mb * (1 + safety_margin)  # Add safety margin
        return available >= required

class ModelSizeEstimator:
    """Estimates memory requirements for different model types."""
    
    MODEL_SIZE_ESTIMATES = {
        'stable_diffusion_1_5': 4200,
        'stable_diffusion_xl': 6900,
        'anythingv5': 5800,
        'counterfeitv3': 5600,
        'realisticvision': 7200,
        
        'svd_xt': 8500,
        'zeroscope_v2_xl': 6800,
        'animatediff_v2_sdxl': 5200,
        'modelscope_t2v': 3800,
        
        'deepseek_llama_8b': 16000,
        'deepseek_r1': 14000,
        
        'bark': 2800,
        'xtts': 1200,
        'musicgen': 3500,
        
        'lora_default': 150,
        
        'real_esrgan': 800,
        'swinir': 600,
    }
    
    @classmethod
    def estimate_model_size(cls, model_name: str, model_type: str = None) -> float:
        """Estimate memory requirement for a model in MB."""
        if model_name in cls.MODEL_SIZE_ESTIMATES:
            return cls.MODEL_SIZE_ESTIMATES[model_name]
            
        if model_type:
            if 'lora' in model_type.lower():
                return cls.MODEL_SIZE_ESTIMATES['lora_default']
            elif 'video' in model_type.lower():
                return 5000  # Average video model size
            elif 'text' in model_type.lower() or 'llm' in model_type.lower():
                return 8000  # Average LLM size
            elif 'audio' in model_type.lower():
                return 2000  # Average audio model size
            elif 'upscaling' in model_type.lower():
                return 700   # Average upscaling model size
                
        name_lower = model_name.lower()
        if 'xl' in name_lower or 'large' in name_lower:
            return 8000
        elif 'small' in name_lower or 'mini' in name_lower:
            return 1000
        elif 'lora' in name_lower:
            return 150
        else:
            return 4000  # Default estimate

class GPUMemoryPool:
    """Advanced GPU memory pool with intelligent allocation and defragmentation."""
    
    def __init__(self, max_memory_usage: float = 0.9):
        self.max_memory_usage = max_memory_usage  # Use up to 90% of GPU memory
        self.memory_monitor = GPUMemoryMonitor()
        self.size_estimator = ModelSizeEstimator()
        
        self.allocated_segments: Dict[str, MemorySegment] = {}
        self.allocation_order = OrderedDict()  # Track allocation order
        self.access_history = defaultdict(list)  # Track access patterns
        self.lock = threading.RLock()
        
        self.allocation_count = 0
        self.eviction_count = 0
        self.fragmentation_events = 0
        
        logger.info("GPU Memory Pool initialized")
        
    def allocate_model_memory(self, model_name: str, model_type: str = None, 
                            priority: MemoryPriority = MemoryPriority.MEDIUM) -> bool:
        """Allocate memory for a model."""
        with self.lock:
            if model_name in self.allocated_segments:
                self._update_access(model_name)
                return True
                
            estimated_size = self.size_estimator.estimate_model_size(model_name, model_type)
            
            if not self._can_allocate_size(estimated_size):
                if not self._free_memory_for_allocation(estimated_size):
                    logger.warning(f"Cannot allocate memory for {model_name} ({estimated_size}MB)")
                    return False
                    
            segment = MemorySegment(
                model_name=model_name,
                size_mb=estimated_size,
                priority=priority,
                last_accessed=time.time(),
                access_count=1,
                model_ref=None  # Will be set when model is loaded
            )
            
            self.allocated_segments[model_name] = segment
            self.allocation_order[model_name] = time.time()
            self.allocation_count += 1
            
            logger.info(f"Allocated {estimated_size}MB for {model_name}")
            return True
            
    def deallocate_model_memory(self, model_name: str):
        """Deallocate memory for a model."""
        with self.lock:
            if model_name in self.allocated_segments:
                segment = self.allocated_segments.pop(model_name)
                self.allocation_order.pop(model_name, None)
                
                self._clear_gpu_memory(segment)
                
                logger.info(f"Deallocated {segment.size_mb}MB for {model_name}")
                
    def set_model_reference(self, model_name: str, model_ref: Any):
        """Set the actual model reference for a memory segment."""
        with self.lock:
            if model_name in self.allocated_segments:
                self.allocated_segments[model_name].model_ref = model_ref
                self._update_access(model_name)
                
    def get_model_reference(self, model_name: str) -> Optional[Any]:
        """Get the model reference if available."""
        with self.lock:
            if model_name in self.allocated_segments:
                self._update_access(model_name)
                return self.allocated_segments[model_name].model_ref
            return None
            
    def _can_allocate_size(self, size_mb: float) -> bool:
        """Check if we can allocate the requested size."""
        if not self.memory_monitor.gpu_available:
            return True  # Allow allocation if no GPU (CPU fallback)
            
        current_usage = sum(seg.size_mb for seg in self.allocated_segments.values())
        max_allowed = self.memory_monitor.total_memory_mb * self.max_memory_usage
        
        return (current_usage + size_mb) <= max_allowed
        
    def _free_memory_for_allocation(self, required_mb: float) -> bool:
        """Free memory to make space for new allocation."""
        with self.lock:
            current_usage = sum(seg.size_mb for seg in self.allocated_segments.values())
            max_allowed = self.memory_monitor.total_memory_mb * self.max_memory_usage
            available = max_allowed - current_usage
            
            if available >= required_mb:
                return True
                
            need_to_free = required_mb - available
            freed = 0
            
            candidates = self._get_eviction_candidates()
            
            for model_name in candidates:
                if freed >= need_to_free:
                    break
                    
                segment = self.allocated_segments.get(model_name)
                if segment and segment.priority != MemoryPriority.CRITICAL:
                    freed += segment.size_mb
                    self.deallocate_model_memory(model_name)
                    self.eviction_count += 1
                    
            return freed >= need_to_free
            
    def _get_eviction_candidates(self) -> List[str]:
        """Get list of models that can be evicted, sorted by priority."""
        candidates = []
        
        for model_name, segment in self.allocated_segments.items():
            if segment.priority != MemoryPriority.CRITICAL:
                priority_score = segment.priority.value
                time_score = time.time() - segment.last_accessed
                access_score = 1.0 / max(1, segment.access_count)
                
                total_score = priority_score + (time_score / 3600) + access_score
                candidates.append((total_score, model_name))
                
        candidates.sort(reverse=True)
        return [model_name for _, model_name in candidates]
        
    def _update_access(self, model_name: str):
        """Update access statistics for a model."""
        if model_name in self.allocated_segments:
            segment = self.allocated_segments[model_name]
            segment.last_accessed = time.time()
            segment.access_count += 1
            
            self.access_history[model_name].append(time.time())
            cutoff = time.time() - 3600  # Last hour
            self.access_history[model_name] = [
                t for t in self.access_history[model_name] if t > cutoff
            ]
            
    def _clear_gpu_memory(self, segment: MemorySegment):
        """Clear GPU memory for a segment."""
        try:
            if segment.model_ref:
                del segment.model_ref
                
            gc.collect()
            
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
                
        except Exception as e:
            logger.error(f"Error clearing GPU memory: {e}")
            
    def optimize_memory_layout(self):
        """Optimize memory layout to reduce fragmentation."""
        with self.lock:
            try:
                gc.collect()
                
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.info("GPU memory cache cleared")
                except ImportError:
                    pass
                    
                self.fragmentation_events += 1
                
            except Exception as e:
                logger.error(f"Memory optimization failed: {e}")
                
    def get_memory_info(self) -> Dict[str, Any]:
        """Get memory information for compatibility."""
        return self.get_memory_stats()
    
    def allocate_memory(self, model_name: str, size_mb: float) -> bool:
        """Allocate memory for a model."""
        return self.allocate_model_memory(model_name, model_type=None)
    
    def deallocate_memory(self, model_name: str):
        """Deallocate memory for a model."""
        return self.deallocate_model_memory(model_name)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        with self.lock:
            self.memory_monitor.update_memory_stats()
            
            allocated_memory = sum(seg.size_mb for seg in self.allocated_segments.values())
            
            return {
                'gpu_available': self.memory_monitor.gpu_available,
                'total_memory_mb': self.memory_monitor.total_memory_mb,
                'used_memory_mb': self.memory_monitor.used_memory_mb,
                'free_memory_mb': self.memory_monitor.free_memory_mb,
                'utilization_percent': self.memory_monitor.utilization_percent,
                'allocated_segments': len(self.allocated_segments),
                'allocated_memory_mb': allocated_memory,
                'allocation_count': self.allocation_count,
                'eviction_count': self.eviction_count,
                'fragmentation_events': self.fragmentation_events,
                'memory_pressure': self.memory_monitor.get_memory_pressure()
            }
            
    def cleanup(self):
        """Cleanup all allocated memory."""
        with self.lock:
            model_names = list(self.allocated_segments.keys())
            for model_name in model_names:
                self.deallocate_model_memory(model_name)
                
            self.optimize_memory_layout()
            logger.info("GPU memory pool cleaned up")

_gpu_memory_pool = None

def get_gpu_memory_pool() -> GPUMemoryPool:
    """Get the global GPU memory pool instance."""
    global _gpu_memory_pool
    if _gpu_memory_pool is None:
        _gpu_memory_pool = GPUMemoryPool()
    return _gpu_memory_pool

def cleanup_gpu_memory_pool():
    """Cleanup the global GPU memory pool."""
    global _gpu_memory_pool
    if _gpu_memory_pool:
        _gpu_memory_pool.cleanup()
        _gpu_memory_pool = None
