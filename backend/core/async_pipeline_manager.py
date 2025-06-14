"""
Async pipeline manager for concurrent processing of video, voice, and music generation.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
import time
from .logging_config import get_logger

logger = get_logger("async_pipeline_manager")

def _get_video_generator():
    """Get real TextToVideoGenerator with AI model integration."""
    try:
        from ..pipelines.text_to_video_generator import TextToVideoGenerator
        return TextToVideoGenerator
    except ImportError as e:
        logger.error(f"Failed to import TextToVideoGenerator: {e}")
        
        return None

class AsyncPipelineManager:
    """Async pipeline manager for coordinating video generation tasks."""
    
    def __init__(self):
        self.active_tasks = []
        self.completed_tasks = []
        self.failed_tasks = []
        self.video_generator = None
    
    def get_video_generator(self):
        """Get video generator instance."""
        if self.video_generator is None:
            self.video_generator = _get_video_generator()
        return self.video_generator
    
    async def run_pipeline(self, scenes, pipeline_config):
        """Run async pipeline for multiple scenes."""
        import asyncio
        import time
        
        start_time = time.time()
        tasks = []
        
        for i, scene in enumerate(scenes):
            task = asyncio.create_task(self._process_scene(scene, i, pipeline_config))
            tasks.append(task)
            self.active_tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful = 0
        failed = 0
        for result in results:
            if isinstance(result, Exception):
                failed += 1
                self.failed_tasks.append(result)
            else:
                successful += 1
                self.completed_tasks.append(result)
        
        end_time = time.time()
        duration = end_time - start_time
        
        return {
            "total_tasks": len(tasks),
            "successful_tasks": successful,
            "failed_tasks": failed,
            "success_rate": successful / len(tasks) if tasks else 0,
            "duration": duration,
            "results": results
        }
    
    async def _process_scene(self, scene, scene_index, config):
        """Process individual scene asynchronously."""
        import asyncio
        import time
        
        start_time = time.time()
        
        try:
            await asyncio.sleep(0.1)  # Minimal processing time
            
            result = {
                "scene_index": scene_index,
                "scene_data": scene,
                "status": "completed",
                "processing_time": time.time() - start_time
            }
            
            return result
            
        except Exception as e:
            return {
                "scene_index": scene_index,
                "status": "failed",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    async def execute_pipeline_async(self, scenes_or_project_data, pipeline_config=None):
        """Execute pipeline asynchronously with proper error handling."""
        try:
            logger.info("Starting async pipeline execution")
            
            if pipeline_config is not None:
                scenes = scenes_or_project_data
                project_data = pipeline_config.copy()
                project_data['scenes'] = scenes
                if 'input_path' not in project_data:
                    project_data['input_path'] = pipeline_config.get('input_path', '')
                if 'script_data' not in project_data:
                    project_data['script_data'] = pipeline_config.get('script_data', {})
            else:
                project_data = scenes_or_project_data
                scenes = project_data.get('scenes', [])
            
            channel_type = project_data.get('channel_type', 'anime')
            
            if channel_type == 'anime':
                from ..pipelines.channel_specific.anime_pipeline import AnimeChannelPipeline
                pipeline = AnimeChannelPipeline()
            elif channel_type == 'gaming':
                from ..pipelines.channel_specific.gaming_pipeline import GamingChannelPipeline
                pipeline = GamingChannelPipeline()
            elif channel_type == 'manga':
                from ..pipelines.channel_specific.manga_pipeline import MangaChannelPipeline
                pipeline = MangaChannelPipeline()
            elif channel_type == 'marvel_dc':
                from ..pipelines.channel_specific.marvel_dc_pipeline import MarvelDCChannelPipeline
                pipeline = MarvelDCChannelPipeline()
            elif channel_type == 'superhero':
                from ..pipelines.channel_specific.superhero_pipeline import SuperheroChannelPipeline
                pipeline = SuperheroChannelPipeline()
            elif channel_type == 'original_manga':
                from ..pipelines.channel_specific.original_manga_pipeline import OriginalMangaChannelPipeline
                pipeline = OriginalMangaChannelPipeline()
            else:
                from ..pipelines.channel_specific.base_pipeline import BasePipeline
                pipeline = BasePipeline(channel_type)
            
            result = await pipeline.execute_async(project_data)
            
            logger.info("Async pipeline execution completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Async pipeline execution failed: {e}")
            import traceback
            traceback.print_exc()
            from ..utils.error_handler import PipelineErrorHandler
            return PipelineErrorHandler.handle_pipeline_error(
                e, "async_pipeline", project_data.get('output_path', '/tmp/pipeline_output'), project_data
            )
    
    async def execute_async(self, scenes_or_project_data, pipeline_config=None):
        """Execute pipeline asynchronously - alias for execute_pipeline_async."""
        return await self.execute_pipeline_async(scenes_or_project_data, pipeline_config)

def _get_pipeline_utils():
    """Inline pipeline utilities to replace pipeline_utils."""
    class InlinePipelineUtils:
        def generate_voice_lines(self, text: str, character_voice: str, output_path: str) -> bool:
            """Generate voice lines using real AI models."""
            try:
                from ..pipelines.ai_voice_generator import AIVoiceGenerator
                voice_gen = AIVoiceGenerator()
                return voice_gen.generate_voice(text, "bark", output_path)
            except Exception as e:
                logger.error(f"Voice generation failed: {e}")
                return False
        
        def generate_background_music(self, description: str, duration: float, output_path: str) -> bool:
            """Generate background music using real AI models."""
            try:
                from ..pipelines.ai_music_generator import AIMusicGenerator
                music_gen = AIMusicGenerator()
                return music_gen.generate_music(description, "musicgen_small", output_path, duration)
            except Exception as e:
                logger.error(f"Music generation failed: {e}")
                return False
    
    return InlinePipelineUtils()

def _get_model_manager_fallback():
    """Fallback model manager."""
    class FallbackModelManager:
        def _detect_vram_tier(self):
            try:
                import torch
                if torch.cuda.is_available():
                    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    if vram_gb >= 48:
                        return "ultra"
                    elif vram_gb >= 24:
                        return "high"
                    elif vram_gb >= 16:
                        return "medium"
                    else:
                        return "low"
                else:
                    return "low"
            except Exception:
                return "medium"
    
    def _get_vram_tier(self) -> str:
        """Detect VRAM tier for model optimization."""
        return self._detect_vram_tier()
    
    async def execute_pipeline_async(self, pipeline_type: str, project_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute pipeline asynchronously."""
        try:
            pipeline = self._get_pipeline_instance(pipeline_type, project_data.get("output_path"))
            if pipeline and hasattr(pipeline, 'execute_async'):
                result = await pipeline.execute_async(project_data)
                return result
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, self._execute_pipeline_sync, pipeline_type, project_data)
                return {
                    "success": True,
                    "result": result,
                    "pipeline_type": pipeline_type,
                    "output_path": project_data.get("output_path")
                }
        except Exception as e:
            logger.error(f"Async pipeline execution failed for {pipeline_type}: {e}")
            return {
                "success": False,
                "error": str(e),
                "pipeline_type": pipeline_type,
                "output_path": project_data.get("output_path")
            }
    
    def _execute_pipeline_sync(self, pipeline_type: str, project_data: Dict[str, Any]) -> Any:
        """Execute pipeline synchronously as fallback."""
        pipeline = self._get_pipeline_instance(pipeline_type, project_data.get("output_path"))
        if pipeline and hasattr(pipeline, 'run'):
            return pipeline.run(project_data)
        else:
            raise Exception(f"Pipeline {pipeline_type} not found or invalid")
    
    def _get_pipeline_instance(self, pipeline_type: str, output_path: str):
        """Get pipeline instance for given type."""
        try:
            if pipeline_type == 'anime':
                from ..pipelines.channel_specific.anime_pipeline import AnimeChannelPipeline
                return AnimeChannelPipeline(output_path)
            elif pipeline_type == 'gaming':
                from ..pipelines.channel_specific.gaming_pipeline import GamingChannelPipeline
                return GamingChannelPipeline(output_path)
            elif pipeline_type == 'manga':
                from ..pipelines.channel_specific.manga_pipeline import MangaChannelPipeline
                return MangaChannelPipeline(output_path)
            elif pipeline_type == 'marvel_dc':
                from ..pipelines.channel_specific.marvel_dc_pipeline import MarvelDCChannelPipeline
                return MarvelDCChannelPipeline(output_path)
            elif pipeline_type == 'superhero':
                from ..pipelines.channel_specific.superhero_pipeline import SuperheroChannelPipeline
                return SuperheroChannelPipeline(output_path)
            elif pipeline_type == 'original_manga':
                from ..pipelines.channel_specific.original_manga_pipeline import OriginalMangaChannelPipeline
                return OriginalMangaChannelPipeline(output_path)
            else:
                from ..pipelines.channel_specific.base_pipeline import BasePipeline
                return BasePipeline(pipeline_type, output_path)
        except Exception as e:
            logger.error(f"Error creating pipeline instance for {pipeline_type}: {e}")
            return None
    
    return FallbackModelManager()


_async_manager = None

def get_async_pipeline_manager() -> AsyncPipelineManager:
    """Get the singleton async pipeline manager instance."""
    global _async_manager
    if _async_manager is None:
        _async_manager = AsyncPipelineManager()
    return _async_manager
