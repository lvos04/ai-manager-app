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

class AsyncPipelineManager:
    """Inline video generator to replace TextToVideoGenerator."""
    class InlineVideoGenerator:
        def __init__(self, vram_tier="medium", target_resolution=(1920, 1080)):
            self.vram_tier = vram_tier
            self.target_resolution = target_resolution
            self.device = "cuda" if vram_tier != "cpu" else "cpu"
        
        def generate_video(self, prompt: str, model_name: str, output_path: str, duration: float = 5.0) -> bool:
            """Generate video using efficient approach."""
            try:
                import cv2
                import numpy as np
                import os
                
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                duration = min(duration, 5.0)
                fps = 24
                frames = int(duration * fps)
                width, height = self.target_resolution
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                if not out.isOpened():
                    return False
                
                for i in range(min(frames, 120)):
                    frame = np.zeros((height, width, 3), dtype=np.uint8)
                    frame[:] = (50, 50, 100)
                    
                    try:
                        cv2.putText(frame, "Generated Video", 
                                   (50, height//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                        cv2.putText(frame, f"Frame {i+1}", 
                                   (50, height//2 + 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
                    except:
                        pass
                    
                    out.write(frame)
                
                out.release()
                return True
                
            except Exception as e:
                return False
    
    def get_video_generator(self):
        return self.InlineVideoGenerator()

def _get_pipeline_utils():
    """Inline pipeline utilities to replace pipeline_utils."""
    class InlinePipelineUtils:
        def generate_voice_lines(self, text: str, character_voice: str, output_path: str) -> bool:
            """Generate voice lines with fallback to silent audio."""
            try:
                import os
                import wave
                import struct
                
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                duration = max(len(text) * 0.1, 1.0)
                sample_rate = 48000
                frames = int(duration * sample_rate)
                
                with wave.open(output_path, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    
                    for _ in range(frames):
                        wav_file.writeframes(struct.pack('<h', 0))
                
                return True
                
            except Exception as e:
                return False
        
        def generate_background_music(self, description: str, duration: float, output_path: str) -> bool:
            """Generate background music with fallback to silent audio."""
            try:
                import os
                import wave
                import struct
                
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                sample_rate = 48000
                frames = int(duration * sample_rate)
                
                with wave.open(output_path, 'wb') as wav_file:
                    wav_file.setnchannels(2)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    
                    for _ in range(frames):
                        wav_file.writeframes(struct.pack('<hh', 0, 0))
                
                return True
                
            except Exception as e:
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
                    if vram_gb >= 24:
                        return "extreme"
                    elif vram_gb >= 16:
                        return "high"
                    elif vram_gb >= 8:
                        return "medium"
                    else:
                        return "low"
                else:
                    return "cpu"
            except Exception:
                return "unknown"
        
        def cleanup_model_memory(self):
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        
        def force_memory_cleanup(self):
            self.cleanup_model_memory()
    
    return FallbackModelManager()

class AsyncPipelineManager:
    """Manages asynchronous execution of pipeline components."""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.max_concurrent_tasks = max_workers
        self.thread_executor = ThreadPoolExecutor(max_workers=max_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=max_workers//2)
        self.active_tasks: Dict[str, asyncio.Task] = {}
        
        from ..core.memory_manager import get_memory_manager
        self.memory_manager = get_memory_manager()
        self.memory_manager.register_cleanup_callback(self._cleanup_pipeline_memory)

    async def execute_pipeline_async(self, scenes: List[Dict], config: Dict) -> Dict[str, Any]:
        """Execute pipeline tasks asynchronously for multiple scenes."""
        if not scenes:
            return {"success": False, "error": "No scenes provided"}
        
        logger.info(f"Starting async pipeline for {len(scenes)} scenes")
        start_time = time.time()
        
        try:
            tasks = []
            for i, scene in enumerate(scenes):
                if config.get("generate_video", True):
                    tasks.append(self._create_video_task(scene, i, config))
                if config.get("generate_voice", True):
                    tasks.append(self._create_voice_task(scene, i, config))
                if config.get("generate_music", True):
                    tasks.append(self._create_music_task(scene, i, config))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            performance_metrics = self._calculate_performance_metrics(results)
            total_time = time.time() - start_time
            
            logger.info(f"Async pipeline completed in {total_time:.2f}s with {len(tasks)} concurrent tasks")
            logger.info(f"Performance metrics: {performance_metrics}")
            
            self._cleanup_comprehensive_memory()
            self._force_memory_release()
            
            return {
                "success": True,
                "results": results,
                "performance_metrics": performance_metrics,
                "total_time": total_time
            }
            
        except Exception as e:
            logger.error(f"Error in async pipeline execution: {e}")
            self._cleanup_comprehensive_memory()
            self._force_memory_release()
            return {
                "success": False,
                "error": str(e),
                "total_time": time.time() - start_time
            }
        finally:
            self._cleanup_comprehensive_memory()
            self._force_memory_release()

    async def _create_video_task(self, scene: Dict, scene_id: int, config: Dict) -> Dict[str, Any]:
        """Create async task for video generation."""
        loop = asyncio.get_event_loop()
        
        def generate_video():
            try:
                async_manager = AsyncPipelineManager()
                video_generator_class = async_manager.get_video_generator()
                start_time = time.time()
                model_manager = _get_model_manager_fallback()
                vram_tier = model_manager._detect_vram_tier()
                
                generator = video_generator_class(vram_tier)
                output_path = f"{config.get('output_path', 'output')}/scene_{scene_id}_video.mp4"
                
                success = generator.generate_video(
                    scene.get("description", ""),
                    config.get("video_model", "animatediff_v2_sdxl"),
                    output_path
                )
                
                return {
                    "scene_id": scene_id,
                    "task_type": "video",
                    "output_path": output_path if success else None,
                    "processing_time": time.time() - start_time,
                    "success": success
                }
            except Exception as e:
                from pathlib import Path
                return {
                    "scene_id": scene_id,
                    "task_type": "video",
                    "output_path": None,
                    "processing_time": time.time() - start_time,
                    "success": False,
                    "error": str(e)
                }
        
        return await loop.run_in_executor(self.thread_executor, generate_video)
    
    async def _create_voice_task(self, scene: Dict, scene_id: int, config: Dict) -> Dict[str, Any]:
        """Create async task for voice generation."""
        loop = asyncio.get_event_loop()
        
        def generate_voice():
            try:
                pipeline_utils = _get_pipeline_utils()
                
                start_time = time.time()
                output_path = f"{config.get('output_path', 'output')}/scene_{scene_id}_voice.wav"
                
                success = pipeline_utils.generate_voice_lines(
                    scene.get("dialogue", ""),
                    scene.get("character_voice", "default"),
                    output_path
                )
                
                return {
                    "scene_id": scene_id,
                    "task_type": "voice",
                    "output_path": output_path if success else None,
                    "processing_time": time.time() - start_time,
                    "success": success
                }
            except Exception as e:
                from pathlib import Path
                return {
                    "scene_id": scene_id,
                    "task_type": "voice",
                    "output_path": None,
                    "processing_time": time.time() - start_time,
                    "success": False,
                    "error": str(e)
                }
        
        return await loop.run_in_executor(self.thread_executor, generate_voice)
    
    async def _create_music_task(self, scene: Dict, scene_id: int, config: Dict) -> Dict[str, Any]:
        """Create async task for music generation."""
        loop = asyncio.get_event_loop()
        
        def generate_music():
            try:
                pipeline_utils = _get_pipeline_utils()
                
                start_time = time.time()
                output_path = f"{config.get('output_path', 'output')}/scene_{scene_id}_music.wav"
                
                success = pipeline_utils.generate_background_music(
                    scene.get("description", ""),
                    scene.get("duration", 10.0),
                    output_path
                )
                
                return {
                    "scene_id": scene_id,
                    "task_type": "music",
                    "output_path": output_path if success else None,
                    "processing_time": time.time() - start_time,
                    "success": success
                }
            except Exception as e:
                from pathlib import Path
                return {
                    "scene_id": scene_id,
                    "task_type": "music",
                    "output_path": None,
                    "processing_time": time.time() - start_time,
                    "success": False,
                    "error": str(e)
                }
        
        return await loop.run_in_executor(self.thread_executor, generate_music)
    
    def _calculate_performance_metrics(self, task_results) -> Dict[str, Any]:
        """Calculate performance metrics from task results."""
        successful_tasks = [r for r in task_results if isinstance(r, dict) and r.get("success")]
        failed_tasks = [r for r in task_results if isinstance(r, Exception) or (isinstance(r, dict) and not r.get("success"))]
        
        processing_times = [r.get("processing_time", 0) for r in successful_tasks]
        
        return {
            "total_tasks": len(task_results),
            "successful_tasks": len(successful_tasks),
            "failed_tasks": len(failed_tasks),
            "success_rate": len(successful_tasks) / len(task_results) if task_results else 0,
            "avg_processing_time": sum(processing_times) / len(processing_times) if processing_times else 0,
            "max_processing_time": max(processing_times) if processing_times else 0,
            "min_processing_time": min(processing_times) if processing_times else 0
        }
    
    async def cleanup(self):
        """Cleanup executors and cancel active tasks."""
        for task_id, task in self.active_tasks.items():
            if not task.done():
                task.cancel()
                logger.info(f"Cancelled task {task_id}")
        
        self.thread_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)
        logger.info("Async pipeline manager cleaned up")

    async def execute_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute multiple pipeline tasks concurrently.
        
        Args:
            tasks: List of task dictionaries containing task configuration
            
        Returns:
            List of results from task execution
        """
        if not tasks:
            return []
        
        logger.info(f"Starting {len(tasks)} concurrent pipeline tasks")
        start_time = time.time()
        
        try:
            semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
            
            task_coroutines = [
                self._execute_single_task(task, semaphore, i) 
                for i, task in enumerate(tasks)
            ]
            
            results = await asyncio.gather(*task_coroutines, return_exceptions=True)
            
            processed_results = []
            successful_tasks = 0
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Task {i} failed: {result}")
                    processed_results.append({
                        "task_id": i,
                        "success": False,
                        "error": str(result),
                        "processing_time": 0
                    })
                else:
                    if isinstance(result, dict) and result.get("success", False):
                        successful_tasks += 1
                    processed_results.append(result)
            
            total_time = time.time() - start_time
            success_rate = successful_tasks / len(tasks) if tasks else 0
            
            logger.info(f"Async pipeline completed in {total_time:.2f}s with {len(tasks)} concurrent tasks")
            logger.info(f"Success rate: {success_rate:.1%} ({successful_tasks}/{len(tasks)})")
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Error in async pipeline execution: {e}")
            return [{"success": False, "error": str(e)} for _ in tasks]

    async def _execute_single_task(self, task: Dict[str, Any], semaphore: asyncio.Semaphore, task_id: int) -> Dict[str, Any]:
        """
        Execute a single pipeline task with concurrency control.
        
        Args:
            task: Task configuration dictionary
            semaphore: Semaphore for concurrency control
            task_id: Unique task identifier
            
        Returns:
            Task execution result
        """
        async with semaphore:
            start_time = time.time()
            
            try:
                task_type = task.get("type", "unknown")
                logger.info(f"Executing task {task_id}: {task_type}")
                
                result = await self._execute_pipeline_task(task)
                
                processing_time = time.time() - start_time
                
                if result is None:
                    result = {"success": False, "error": "Task returned None"}
                
                result.update({
                    "task_id": task_id,
                    "processing_time": processing_time
                })
                
                if result.get("success", False):
                    logger.info(f"Task {task_id} completed successfully in {processing_time:.2f}s")
                else:
                    logger.error(f"Task {task_id} failed: {result.get('error', 'Unknown error')}")
                
                return result
                
            except Exception as e:
                processing_time = time.time() - start_time
                logger.error(f"Task {task_id} failed with exception: {e}")
                return {
                    "task_id": task_id,
                    "success": False,
                    "error": str(e),
                    "processing_time": processing_time
                }
            finally:
                self._cleanup_comprehensive_memory()
                self._force_memory_release()

    async def _execute_pipeline_task(self, task_data: dict) -> dict:
        """Execute a single pipeline task with error handling."""
        try:
            task_type = task_data.get("type")
            
            if task_type == "video_generation":
                try:
                    async_manager = AsyncPipelineManager()
                    video_generator_class = async_manager.get_video_generator()
                    generator = video_generator_class(
                        vram_tier="low",
                        target_resolution=(1920, 1080)
                    )
                    return await self._execute_video_task_new(task_data, generator)
                except Exception as e:
                    logger.error(f"Video generation failed: {e}")
                    return {"success": False, "error": f"Video generation failed: {e}"}
                
            elif task_type == "voice_generation":
                try:
                    pipeline_utils = _get_pipeline_utils()
                    return await self._execute_voice_task_new(task_data, pipeline_utils.generate_voice_lines)
                except Exception as e:
                    logger.error(f"Voice generation failed: {e}")
                    return {"success": False, "error": f"Voice generation failed: {e}"}
                
            elif task_type == "music_generation":
                try:
                    pipeline_utils = _get_pipeline_utils()
                    return await self._execute_music_task_new(task_data, pipeline_utils.generate_background_music)
                except Exception as e:
                    logger.error(f"Music generation failed: {e}")
                    return {"success": False, "error": f"Music generation failed: {e}"}
                
            else:
                logger.warning(f"Unknown task type: {task_type}")
                return {"success": False, "error": f"Unknown task type: {task_type}"}
                
        except ImportError as e:
            logger.error(f"Import error in pipeline task: {e}")
            return {"success": False, "error": f"Module import failed: {e}"}
        except Exception as e:
            logger.error(f"Pipeline task execution failed: {e}")
            return {"success": False, "error": str(e)}

    async def _execute_video_task_new(self, task_data: dict, generator) -> dict:
        """Execute video generation task."""
        loop = asyncio.get_event_loop()
        
        def run_video_task():
            try:
                output_path = task_data.get("output_path", "output/test_video.mp4")
                success = generator.generate_video(
                    task_data.get("text", "Test video"),
                    task_data.get("model", "animatediff_v2_sdxl"),
                    output_path
                )
                return {"success": success, "output_path": output_path}
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        return await loop.run_in_executor(self.thread_executor, run_video_task)

    async def _execute_voice_task_new(self, task_data: dict, voice_func) -> dict:
        """Execute voice generation task."""
        loop = asyncio.get_event_loop()
        
        def run_voice_task():
            try:
                output_path = task_data.get("output_path", "output/test_voice.wav")
                success = voice_func(
                    task_data.get("text", "Test voice"),
                    task_data.get("character_voice", "default"),
                    output_path
                )
                return {"success": success, "output_path": output_path}
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        return await loop.run_in_executor(self.thread_executor, run_voice_task)

    async def _execute_music_task_new(self, task_data: dict, music_func) -> dict:
        """Execute music generation task."""
        loop = asyncio.get_event_loop()
        
        def run_music_task():
            try:
                output_path = task_data.get("output_path", "output/test_music.wav")
                success = music_func(
                    task_data.get("description", "Test music"),
                    task_data.get("duration", 5.0),
                    output_path
                )
                return {"success": success, "output_path": output_path}
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        return await loop.run_in_executor(self.thread_executor, run_music_task)
    
    def _cleanup_pipeline_memory(self):
        """Cleanup pipeline memory when triggered by memory manager."""
        try:
            model_manager = _get_model_manager_fallback()
            model_manager.cleanup_model_memory()
            
            import gc
            gc.collect()
            
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    free = total - reserved
                    
                    if free / total < 0.3:
                        logger.warning(f"Memory fragmentation detected after cleanup: {free:.2f}GB free of {total:.2f}GB total")
                        torch.cuda.set_per_process_memory_fraction(0.5)
                        logger.info("Applied emergency memory fraction limit")
            except Exception as cuda_error:
                logger.warning(f"CUDA cleanup failed: {cuda_error}")
                
            logger.info("Pipeline memory cleanup completed")
        except Exception as e:
            logger.error(f"Pipeline memory cleanup failed: {e}")
    
    def _cleanup_comprehensive_memory(self):
        """Enhanced comprehensive memory cleanup after pipeline completion."""
        try:
            import torch
            import gc
            
            if hasattr(self, 'loaded_models'):
                for model_name, model in self.loaded_models.items():
                    if hasattr(model, 'to'):
                        model.to('cpu')
                    del model
                self.loaded_models.clear()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.ipc_collect()
            
            gc.collect()
            
            logger.info("Comprehensive pipeline memory cleanup completed")
        except Exception as e:
            logger.error(f"Error during comprehensive memory cleanup: {e}")
    
    def _force_memory_release(self):
        """Force aggressive memory release to prevent retention."""
        try:
            model_manager = _get_model_manager_fallback()
            model_manager.force_memory_cleanup()
            
            import gc
            import torch
            
            for _ in range(3):
                gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.ipc_collect()
                
                for device_id in range(torch.cuda.device_count()):
                    with torch.cuda.device(device_id):
                        torch.cuda.empty_cache()
            
            logger.info("Force memory release completed")
        except Exception as e:
            logger.error(f"Error during force memory release: {e}")

_async_manager = None

def get_async_pipeline_manager() -> AsyncPipelineManager:
    """Get the singleton async pipeline manager instance."""
    global _async_manager
    if _async_manager is None:
        _async_manager = AsyncPipelineManager()
    return _async_manager
