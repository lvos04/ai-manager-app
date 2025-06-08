"""
GUI integration for optimization features and model management.
"""

from typing import Dict, List, Any, Optional
from .core.advanced_cache_manager import get_cache_manager
from .core.performance_monitor import get_performance_monitor
from .core.memory_manager import get_memory_manager
from .core.model_version_updater import get_model_version_updater
from .core.logging_config import get_logger

logger = get_logger("gui_integration")

class OptimizationGUIInterface:
    """Interface for GUI to access optimization features."""
    
    def __init__(self):
        self.cache_manager = get_cache_manager()
        self.performance_monitor = get_performance_monitor()
        self.memory_manager = get_memory_manager()
        self.version_updater = get_model_version_updater()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics for GUI display."""
        try:
            perf_stats = self.performance_monitor.get_performance_stats()
            memory_stats = self.memory_manager.get_memory_stats()
            cache_stats = self.cache_manager.get_cache_stats()
            
            return {
                "performance": perf_stats,
                "memory": memory_stats,
                "cache": cache_stats,
                "status": "active"
            }
        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_model_updates(self) -> Dict[str, List[Dict]]:
        """Get available model updates for GUI display."""
        try:
            updates = self.version_updater.check_for_updates()
            update_summary = self.version_updater.get_update_summary()
            
            return {
                "updates": updates,
                "summary": update_summary,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Error getting model updates: {e}")
            return {"status": "error", "error": str(e)}
    
    def apply_model_update(self, model_name: str, category: str) -> Dict[str, Any]:
        """Apply a model update through GUI."""
        try:
            success = self.version_updater.apply_update(model_name, category)
            
            return {
                "success": success,
                "model_name": model_name,
                "category": category,
                "status": "completed" if success else "failed"
            }
        except Exception as e:
            logger.error(f"Error applying model update: {e}")
            return {"status": "error", "error": str(e)}
    
    def force_cache_cleanup(self) -> Dict[str, Any]:
        """Force cache cleanup through GUI."""
        try:
            self.cache_manager.clear_all_caches()
            self.memory_manager.force_cleanup()
            
            return {
                "status": "success",
                "message": "Cache cleanup completed"
            }
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_optimization_settings(self) -> Dict[str, Any]:
        """Get current optimization settings."""
        try:
            from .core.config_manager import get_config_manager
            config_manager = get_config_manager()
            
            return {
                "cache_settings": config_manager.get('performance.cache_settings', {}),
                "async_settings": config_manager.get('performance.async_settings', {}),
                "memory_settings": config_manager.get('performance.memory_management', {}),
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Error getting optimization settings: {e}")
            return {"status": "error", "error": str(e)}
    
    def update_optimization_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Update optimization settings through GUI."""
        try:
            if "memory_threshold" in settings:
                self.memory_manager.memory_threshold = settings["memory_threshold"]
            
            if "vram_threshold" in settings:
                self.memory_manager.vram_threshold = settings["vram_threshold"]
            
            if "check_interval" in settings:
                self.memory_manager.check_interval = settings["check_interval"]
            
            return {
                "status": "success",
                "message": "Settings updated successfully"
            }
        except Exception as e:
            logger.error(f"Error updating optimization settings: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_async_pipeline_status(self) -> Dict[str, Any]:
        """Get current async pipeline status."""
        try:
            from .core.async_pipeline_manager import get_async_pipeline_manager
            async_manager = get_async_pipeline_manager()
            
            return {
                "active_tasks": len(async_manager.active_tasks),
                "max_workers": async_manager.max_workers,
                "status": "active"
            }
        except Exception as e:
            logger.error(f"Error getting async pipeline status: {e}")
            return {"status": "error", "error": str(e)}

def get_gui_interface() -> OptimizationGUIInterface:
    """Get the GUI interface instance."""
    return OptimizationGUIInterface()
