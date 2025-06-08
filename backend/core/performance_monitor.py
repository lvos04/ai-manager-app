"""
Performance monitoring system with VRAM tracking and automatic optimization.
"""

import time
import threading
try:
    import psutil
except ImportError:
    psutil = None

try:
    import GPUtil
except ImportError:
    GPUtil = None
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from collections import deque
import json
from pathlib import Path
from .logging_config import get_logger

logger = get_logger("performance_monitor")

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    gpu_memory_used_mb: float
    gpu_memory_total_mb: float
    gpu_utilization_percent: float
    active_models: int
    pipeline_stage: str
    processing_time: float

class VRAMMonitor:
    """VRAM monitoring and management."""
    
    def __init__(self, warning_threshold: float = 0.85, critical_threshold: float = 0.95):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.callbacks = {
            'warning': [],
            'critical': [],
            'normal': []
        }
    
    def get_vram_usage(self) -> Dict[str, Any]:
        """Get current VRAM usage."""
        try:
            if GPUtil:
                gpus = GPUtil.getGPUs()
                if not gpus:
                    return {"available": False, "message": "No GPU detected"}
                
                gpu = gpus[0]  # Use first GPU
                used_mb = gpu.memoryUsed
                total_mb = gpu.memoryTotal
                usage_percent = used_mb / total_mb
                
                return {
                    "available": True,
                    "used_mb": used_mb,
                    "total_mb": total_mb,
                    "free_mb": total_mb - used_mb,
                    "usage_percent": usage_percent,
                    "utilization_percent": gpu.load * 100,
                    "temperature": gpu.temperature,
                    "name": gpu.name
                }
            else:
                try:
                    import torch
                    if torch.cuda.is_available():
                        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**2)  # MB
                        allocated_memory = torch.cuda.memory_allocated(0) / (1024**2)  # MB
                        usage_percent = allocated_memory / total_memory
                        return {
                            "available": True,
                            "used_mb": allocated_memory,
                            "total_mb": total_memory,
                            "free_mb": total_memory - allocated_memory,
                            "usage_percent": usage_percent,
                            "utilization_percent": 50.0,
                            "temperature": 65,
                            "name": "CUDA Device"
                        }
                except:
                    pass
                
                return {
                    "available": False, 
                    "message": "GPUtil not available and no CUDA detected",
                    "used_mb": 0,
                    "total_mb": 8192,
                    "free_mb": 8192,
                    "usage_percent": 0.0
                }
        except Exception as e:
            logger.error(f"Error getting VRAM usage: {e}")
            return {"available": False, "error": str(e)}
    
    def check_vram_status(self) -> str:
        """Check VRAM status and trigger callbacks if needed."""
        vram_info = self.get_vram_usage()
        
        if not vram_info.get("available"):
            return "unavailable"
        
        usage_percent = vram_info["usage_percent"]
        
        if usage_percent >= self.critical_threshold:
            self._trigger_callbacks('critical', vram_info)
            return "critical"
        elif usage_percent >= self.warning_threshold:
            self._trigger_callbacks('warning', vram_info)
            return "warning"
        else:
            self._trigger_callbacks('normal', vram_info)
            return "normal"
    
    def register_callback(self, status: str, callback: Callable[[Dict], None]) -> None:
        """Register callback for VRAM status changes."""
        if status in self.callbacks:
            self.callbacks[status].append(callback)
    
    def _trigger_callbacks(self, status: str, vram_info: Dict) -> None:
        """Trigger callbacks for status."""
        for callback in self.callbacks[status]:
            try:
                callback(vram_info)
            except Exception as e:
                logger.error(f"Error in VRAM callback: {e}")

class PerformanceMonitor:
    """Comprehensive performance monitoring system."""
    
    def __init__(self, history_size: int = 1000, monitoring_interval: float = 1.0):
        self.history_size = history_size
        self.monitoring_interval = monitoring_interval
        self.metrics_history = deque(maxlen=history_size)
        self.vram_monitor = VRAMMonitor()
        self.monitoring_thread = None
        self.monitoring_active = False
        self.current_pipeline_stage = "idle"
        self.stage_start_time = time.time()
        
        self.vram_monitor.register_callback('warning', self._on_vram_warning)
        self.vram_monitor.register_callback('critical', self._on_vram_critical)
    
    def start_monitoring(self) -> None:
        """Start performance monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                self.vram_monitor.check_vram_status()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        if psutil:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_gb = memory.available / (1024 ** 3)
        else:
            cpu_percent = 50.0
            memory_percent = 60.0
            memory_available_gb = 8.0
        
        vram_info = self.vram_monitor.get_vram_usage()
        gpu_memory_used_mb = vram_info.get("used_mb", 0)
        gpu_memory_total_mb = vram_info.get("total_mb", 0)
        gpu_utilization_percent = vram_info.get("utilization_percent", 0)
        
        processing_time = time.time() - self.stage_start_time
        
        return PerformanceMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_available_gb=memory_available_gb,
            gpu_memory_used_mb=gpu_memory_used_mb,
            gpu_memory_total_mb=gpu_memory_total_mb,
            gpu_utilization_percent=gpu_utilization_percent,
            active_models=self._count_active_models(),
            pipeline_stage=self.current_pipeline_stage,
            processing_time=processing_time
        )
    
    def _count_active_models(self) -> int:
        """Count currently active models."""
        try:
            from .advanced_cache_manager import get_cache_manager
            cache_manager = get_cache_manager()
            return cache_manager.model_cache.get_memory_usage()["total_models"]
        except:
            return 0
    
    def set_pipeline_stage(self, stage: str) -> None:
        """Set current pipeline stage."""
        self.current_pipeline_stage = stage
        self.stage_start_time = time.time()
        logger.debug(f"Pipeline stage changed to: {stage}")
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get current performance metrics."""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None
    
    def get_metrics_summary(self, last_n: int = 100) -> Dict[str, Any]:
        """Get performance metrics summary."""
        if not self.metrics_history:
            return {"error": "No metrics available"}
        
        recent_metrics = list(self.metrics_history)[-last_n:]
        
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]
        gpu_memory_values = [m.gpu_memory_used_mb for m in recent_metrics if m.gpu_memory_total_mb > 0]
        
        return {
            "time_range": {
                "start": recent_metrics[0].timestamp,
                "end": recent_metrics[-1].timestamp,
                "duration_minutes": (recent_metrics[-1].timestamp - recent_metrics[0].timestamp) / 60
            },
            "cpu": {
                "avg": sum(cpu_values) / len(cpu_values),
                "max": max(cpu_values),
                "min": min(cpu_values)
            },
            "memory": {
                "avg_percent": sum(memory_values) / len(memory_values),
                "max_percent": max(memory_values),
                "min_percent": min(memory_values),
                "current_available_gb": recent_metrics[-1].memory_available_gb
            },
            "gpu": {
                "avg_memory_mb": sum(gpu_memory_values) / len(gpu_memory_values) if gpu_memory_values else 0,
                "max_memory_mb": max(gpu_memory_values) if gpu_memory_values else 0,
                "total_memory_mb": recent_metrics[-1].gpu_memory_total_mb,
                "current_utilization": recent_metrics[-1].gpu_utilization_percent
            },
            "pipeline": {
                "current_stage": recent_metrics[-1].pipeline_stage,
                "stage_duration": recent_metrics[-1].processing_time,
                "active_models": recent_metrics[-1].active_models
            }
        }
    
    def _on_vram_warning(self, vram_info: Dict) -> None:
        """Handle VRAM warning threshold."""
        logger.warning(f"VRAM usage warning: {vram_info['usage_percent']:.1%} used ({vram_info['used_mb']:.0f}MB/{vram_info['total_mb']:.0f}MB)")
        
        try:
            from .advanced_cache_manager import get_cache_manager
            cache_manager = get_cache_manager()
            
            model_usage = cache_manager.model_cache.get_memory_usage()
            if model_usage["total_models"] > 2:
                cache_manager.model_cache._free_memory_if_needed(1024 * 1024 * 1024)  # Free 1GB
                logger.info("Freed models due to VRAM warning")
        except Exception as e:
            logger.error(f"Error during VRAM warning cleanup: {e}")
    
    def _on_vram_critical(self, vram_info: Dict) -> None:
        """Handle VRAM critical threshold."""
        logger.critical(f"VRAM usage critical: {vram_info['usage_percent']:.1%} used ({vram_info['used_mb']:.0f}MB/{vram_info['total_mb']:.0f}MB)")
        
        try:
            from .advanced_cache_manager import get_cache_manager
            cache_manager = get_cache_manager()
            
            cache_manager.model_cache.models.clear()
            cache_manager.model_cache.model_sizes.clear()
            
            import gc
            gc.collect()
            
            logger.warning("Cleared all model cache due to critical VRAM usage")
        except Exception as e:
            logger.error(f"Error during critical VRAM cleanup: {e}")
    
    def monitor_operation(self, operation_name: str):
        """Context manager for monitoring operations."""
        class OperationMonitor:
            def __init__(self, monitor, name):
                self.monitor = monitor
                self.name = name
                self.start_time = None
            
            def __enter__(self):
                self.start_time = time.time()
                self.monitor.set_pipeline_stage(self.name)
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                duration = time.time() - self.start_time
                logger.info(f"Operation '{self.name}' completed in {duration:.2f}s")
        
        return OperationMonitor(self, operation_name)
    
    def save_metrics_report(self, filepath: str) -> None:
        """Save performance metrics report to file."""
        try:
            report = {
                "summary": self.get_metrics_summary(),
                "vram_status": self.vram_monitor.get_vram_usage(),
                "metrics_count": len(self.metrics_history),
                "generated_at": time.time()
            }
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Performance report saved to: {filepath}")
        except Exception as e:
            logger.error(f"Error saving performance report: {e}")

_performance_monitor = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get the singleton performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor
