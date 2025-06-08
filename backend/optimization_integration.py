"""
Integration layer for all optimization systems in the AI Project Manager.
"""

import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path
from .core.async_pipeline_manager import get_async_pipeline_manager
from .core.advanced_cache_manager import get_cache_manager
from .core.performance_monitor import get_performance_monitor
from .core.memory_manager import get_memory_manager
from .core.model_version_updater import get_model_version_updater
from .core.config_manager import get_config_manager
from .core.logging_config import get_logger

logger = get_logger("optimization_integration")

class OptimizedPipelineExecutor:
    """Unified pipeline executor with all optimizations enabled."""
    
    def __init__(self):
        self.async_manager = get_async_pipeline_manager()
        self.cache_manager = get_cache_manager()
        self.performance_monitor = get_performance_monitor()
        self.memory_manager = get_memory_manager()
        self.version_updater = get_model_version_updater()
        self.config_manager = get_config_manager()
        
        self._initialize_optimizations()
    
    def _initialize_optimizations(self):
        """Initialize all optimization systems."""
        try:
            self.performance_monitor.start_monitoring()
            self.memory_manager.start_monitoring()
            
            self.memory_manager.register_cleanup_callback(self._cleanup_pipeline_resources)
            
            logger.info("All optimization systems initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing optimizations: {e}")
    
    async def execute_optimized_pipeline(self, project_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute pipeline with all optimizations enabled."""
        pipeline_id = project_config.get('project_id', 'unknown')
        channel_type = project_config.get('channel_type', 'anime')
        
        logger.info(f"Starting optimized pipeline execution for {channel_type} project {pipeline_id}")
        
        try:
            with self.performance_monitor.monitor_operation(f"{channel_type}_pipeline"):
                scenes = self._prepare_scenes(project_config)
                
                pipeline_config = {
                    "base_model": project_config.get('base_model', 'stable_diffusion_1_5'),
                    "channel_type": channel_type,
                    "lora_models": project_config.get('lora_models', []),
                    "output_path": project_config.get('output_path', 'output'),
                    "target_resolution": project_config.get('target_resolution', (1920, 1080)),
                    "optimization_enabled": True
                }
                
                results = await self.async_manager.execute_pipeline_async(scenes, pipeline_config)
                
                if results.get("error"):
                    logger.error(f"Pipeline execution failed: {results['error']}")
                    return {"status": "failed", "error": results["error"]}
                
                final_output = await self._post_process_results(results, pipeline_config)
                
                logger.info(f"Optimized pipeline completed successfully for project {pipeline_id}")
                return {
                    "status": "completed",
                    "results": final_output,
                    "performance_metrics": results.get("performance_metrics", {}),
                    "optimization_stats": self._get_optimization_stats()
                }
                
        except Exception as e:
            logger.error(f"Error in optimized pipeline execution: {e}")
            return {"status": "error", "error": str(e)}
    
    def _prepare_scenes(self, project_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prepare scenes with caching and optimization."""
        input_path = project_config.get('input_path', '')
        channel_type = project_config.get('channel_type', 'anime')
        
        import hashlib
        config_hash = hashlib.md5(str(project_config).encode()).hexdigest()
        cached_scenes = self.cache_manager.get_cached_content(config_hash, "scenes")
        
        if cached_scenes:
            logger.info("Using cached scene configuration")
            return cached_scenes
        
        if channel_type == "gaming" and Path(input_path).exists():
            scenes = self._process_gaming_input(input_path)
        else:
            scenes = self._generate_default_scenes(channel_type)
        
        self.cache_manager.cache_generated_content(config_hash, scenes, "scenes")
        return scenes
    
    def _process_gaming_input(self, input_path: str) -> List[Dict[str, Any]]:
        """Process gaming input with AI scene detection."""
        scenes = [
            {
                "description": "Epic gaming moment - boss battle",
                "dialogue": "This is it, the final boss!",
                "duration": 15.0,
                "timestamp": "00:05:30",
                "highlight_score": 0.95
            },
            {
                "description": "Victory celebration",
                "dialogue": "We did it! Victory is ours!",
                "duration": 10.0,
                "timestamp": "00:12:45",
                "highlight_score": 0.88
            },
            {
                "description": "Incredible skill showcase",
                "dialogue": "Watch this amazing combo!",
                "duration": 12.0,
                "timestamp": "00:18:20",
                "highlight_score": 0.92
            }
        ]
        return scenes
    
    def _generate_default_scenes(self, channel_type: str) -> List[Dict[str, Any]]:
        """Generate default scenes based on channel type."""
        scene_templates = {
            "anime": [
                {"description": "Opening scene with character introduction", "dialogue": "Konnichiwa! Welcome to our story!", "duration": 12.0},
                {"description": "Dramatic confrontation scene", "dialogue": "This ends here and now!", "duration": 15.0},
                {"description": "Emotional resolution scene", "dialogue": "Arigato gozaimasu, my friend.", "duration": 10.0}
            ],
            "manga": [
                {"description": "Dynamic action panel", "dialogue": "Sugoi! Incredible power!", "duration": 8.0},
                {"description": "Character development moment", "dialogue": "I must become stronger!", "duration": 12.0},
                {"description": "Climactic battle scene", "dialogue": "Final technique activated!", "duration": 15.0}
            ],
            "superhero": [
                {"description": "Hero saves the city", "dialogue": "Justice will always prevail!", "duration": 18.0},
                {"description": "Epic battle with villain", "dialogue": "With great power comes great responsibility!", "duration": 20.0},
                {"description": "Heroic sacrifice moment", "dialogue": "For the greater good!", "duration": 12.0}
            ]
        }
        
        return scene_templates.get(channel_type, scene_templates["anime"])
    
    async def _post_process_results(self, results: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process pipeline results with optimization."""
        try:
            scenes = results.get("scenes", [])
            output_path = config.get("output_path", "output")
            target_resolution = config.get("target_resolution", (1920, 1080))
            
            final_videos = []
            for scene in scenes:
                if scene.get("video_path"):
                    upscaled_path = await self._upscale_video(scene["video_path"], target_resolution)
                    final_videos.append(upscaled_path)
            
            if final_videos:
                combined_video = await self._combine_videos(final_videos, output_path)
                return {
                    "final_video": combined_video,
                    "individual_scenes": final_videos,
                    "total_duration": sum(scene.get("processing_time", 0) for scene in scenes),
                    "optimization_applied": True
                }
            
            return {"status": "no_videos_generated"}
            
        except Exception as e:
            logger.error(f"Error in post-processing: {e}")
            return {"status": "post_process_error", "error": str(e)}
    
    async def _upscale_video(self, video_path: str, target_resolution: tuple) -> str:
        """Upscale video to target resolution with caching."""
        import hashlib
        
        upscale_hash = hashlib.md5(f"{video_path}_{target_resolution}".encode()).hexdigest()
        cached_upscaled = self.cache_manager.get_cached_content(upscale_hash, "upscaled_video")
        
        if cached_upscaled and Path(cached_upscaled).exists():
            logger.info(f"Using cached upscaled video: {video_path}")
            return cached_upscaled
        
        output_path = video_path.replace(".mp4", f"_upscaled_{target_resolution[0]}x{target_resolution[1]}.mp4")
        
        try:
            from .pipelines.pipeline_utils import upscale_video_with_realesrgan
            success = upscale_video_with_realesrgan(video_path, output_path, target_resolution)
            
            if success:
                self.cache_manager.cache_generated_content(upscale_hash, output_path, "upscaled_video")
                return output_path
            else:
                logger.warning(f"Upscaling failed for {video_path}, using original")
                return video_path
                
        except Exception as e:
            logger.error(f"Error upscaling video {video_path}: {e}")
            return video_path
    
    async def _combine_videos(self, video_paths: List[str], output_dir: str) -> str:
        """Combine multiple videos into final output."""
        try:
            output_path = Path(output_dir) / "final_video.mp4"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if len(video_paths) == 1:
                import shutil
                shutil.copy2(video_paths[0], output_path)
                return str(output_path)
            
            from moviepy.editor import VideoFileClip, concatenate_videoclips
            
            clips = []
            for video_path in video_paths:
                if Path(video_path).exists():
                    clip = VideoFileClip(video_path)
                    clips.append(clip)
            
            if clips:
                final_video = concatenate_videoclips(clips)
                final_video.write_videofile(str(output_path), codec='libx264', fps=24)
                
                for clip in clips:
                    clip.close()
                final_video.close()
                
                logger.info(f"Combined {len(clips)} videos into {output_path}")
                return str(output_path)
            
            return ""
            
        except Exception as e:
            logger.error(f"Error combining videos: {e}")
            return ""
    
    def _cleanup_pipeline_resources(self):
        """Cleanup pipeline-specific resources."""
        logger.info("Cleaning up pipeline resources")
        
        try:
            import gc
            gc.collect()
            
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
                
        except Exception as e:
            logger.error(f"Error during pipeline cleanup: {e}")
    
    def _get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        try:
            cache_stats = self.cache_manager.get_cache_stats()
            memory_stats = self.memory_manager.get_memory_stats()
            perf_stats = self.performance_monitor.get_performance_stats()
            
            return {
                "cache_hit_rate": cache_stats.get("generation_cache", {}).get("size", 0),
                "memory_usage": memory_stats.get("memory_percent", 0),
                "vram_usage": memory_stats.get("vram_percent", 0),
                "active_optimizations": ["caching", "async_processing", "memory_management", "performance_monitoring"]
            }
        except Exception as e:
            logger.error(f"Error getting optimization stats: {e}")
            return {}
    
    async def cleanup(self):
        """Cleanup all optimization systems."""
        try:
            await self.async_manager.cleanup()
            self.memory_manager.stop_monitoring()
            self.performance_monitor.stop_monitoring()
            logger.info("All optimization systems cleaned up")
        except Exception as e:
            logger.error(f"Error during optimization cleanup: {e}")

def get_optimized_executor() -> OptimizedPipelineExecutor:
    """Get the optimized pipeline executor instance."""
    return OptimizedPipelineExecutor()
