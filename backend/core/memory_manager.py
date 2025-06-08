"""
Memory management system for AI Project Manager with automatic cleanup and VRAM monitoring.
"""

try:
    import psutil
except ImportError:
    psutil = None
import threading
import time
from typing import Dict, List, Callable, Optional
from .logging_config import get_logger

logger = get_logger("memory_manager")

class MemoryManager:
    """Manages system memory and VRAM with automatic cleanup triggers."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MemoryManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self.cleanup_callbacks: List[Callable] = []
            self.monitoring_thread = None
            self.monitoring_active = False
            self.memory_threshold = 85.0  # Trigger cleanup at 85% memory usage
            self.vram_threshold = 90.0    # Trigger cleanup at 90% VRAM usage
            self.check_interval = 30      # Check every 30 seconds
            self._initialized = True
    
    def start_monitoring(self) -> None:
        """Start memory monitoring thread."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitor_memory, daemon=True)
        self.monitoring_thread.start()
        logger.info("Memory monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop memory monitoring thread."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Memory monitoring stopped")
    
    def register_cleanup_callback(self, callback: Callable) -> None:
        """Register a callback function to be called when cleanup is needed."""
        self.cleanup_callbacks.append(callback)
        logger.debug(f"Registered cleanup callback: {callback.__name__}")
    
    def _monitor_memory(self) -> None:
        """Monitor memory usage and trigger cleanup when needed."""
        while self.monitoring_active:
            try:
                if psutil:
                    memory_percent = psutil.virtual_memory().percent
                    vram_percent = self._get_vram_usage_percent()
                    
                    if memory_percent > self.memory_threshold:
                        logger.warning(f"High memory usage detected: {memory_percent:.1f}%")
                        self._trigger_cleanup("memory")
                    
                    if vram_percent > self.vram_threshold:
                        logger.warning(f"High VRAM usage detected: {vram_percent:.1f}%")
                        self._trigger_cleanup("vram")
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")
                time.sleep(self.check_interval)
    
    def _get_vram_usage_percent(self) -> float:
        """Get current VRAM usage percentage."""
        try:
            import torch
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory
                allocated_memory = torch.cuda.memory_allocated(0)
                return (allocated_memory / total_memory) * 100
        except:
            pass
        return 0.0
    
    def _trigger_cleanup(self, reason: str) -> None:
        """Trigger all registered cleanup callbacks."""
        logger.info(f"Triggering memory cleanup due to high {reason} usage")
        
        for callback in self.cleanup_callbacks:
            try:
                callback()
                logger.debug(f"Executed cleanup callback: {callback.__name__}")
            except Exception as e:
                logger.error(f"Error in cleanup callback {callback.__name__}: {e}")
        
        import gc
        gc.collect()
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("Cleared CUDA cache")
        except:
            pass
    
    def force_cleanup(self) -> None:
        """Force immediate cleanup."""
        self._trigger_cleanup("manual")
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        stats = {"vram_percent": self._get_vram_usage_percent()}
        
        if psutil:
            memory = psutil.virtual_memory()
            stats.update({
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "memory_used_gb": memory.used / (1024**3),
                "memory_total_gb": memory.total / (1024**3)
            })
        else:
            stats.update({
                "memory_percent": 50.0,
                "memory_available_gb": 8.0,
                "memory_used_gb": 8.0,
                "memory_total_gb": 16.0
            })
        
        return stats

_memory_manager = None

def get_memory_manager() -> MemoryManager:
    """Get the singleton memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager
