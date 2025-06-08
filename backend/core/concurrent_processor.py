"""
Concurrent processing utilities for parallel operations.
"""

import asyncio
import concurrent.futures
import logging
from typing import List, Callable, Any, Tuple
import threading
import time

logger = logging.getLogger(__name__)

class ConcurrentPipelineProcessor:
    """Process independent pipeline operations concurrently."""
    
    def __init__(self, max_workers=4):
        self.max_workers = max_workers
        self.executor = None
    
    def __enter__(self):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.executor:
            self.executor.shutdown(wait=True)
    
    async def run_concurrent_operations(self, operations: List[Tuple[Callable, tuple, dict]]):
        """Run independent operations concurrently."""
        if not operations:
            return []
        
        loop = asyncio.get_event_loop()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            tasks = []
            
            for func, args, kwargs in operations:
                task = loop.run_in_executor(
                    executor, 
                    lambda f=func, a=args, k=kwargs: f(*a, **k)
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Operation {i} failed: {result}")
                    processed_results.append(None)
                else:
                    processed_results.append(result)
            
            return processed_results
    
    def run_parallel_media_generation(self, scene_data, voice_func, music_func, video_func):
        """Run voice, music, and video generation in parallel."""
        operations = [
            (voice_func, (scene_data,), {}),
            (music_func, (scene_data,), {}),
            (video_func, (scene_data,), {})
        ]
        
        return asyncio.run(self.run_concurrent_operations(operations))
    
    def run_parallel_batch_processing(self, batch_data, processing_func, **kwargs):
        """Process multiple batches in parallel."""
        operations = [
            (processing_func, (batch,), kwargs) 
            for batch in batch_data
        ]
        
        return asyncio.run(self.run_concurrent_operations(operations))

class ThreadSafeProgress:
    """Thread-safe progress tracking for concurrent operations."""
    
    def __init__(self, total_operations=0):
        self.total_operations = total_operations
        self.completed_operations = 0
        self.lock = threading.Lock()
        self.callbacks = []
    
    def add_callback(self, callback):
        """Add progress callback."""
        self.callbacks.append(callback)
    
    def update_progress(self, increment=1):
        """Update progress in thread-safe manner."""
        with self.lock:
            self.completed_operations += increment
            progress = self.completed_operations / max(self.total_operations, 1)
            
            for callback in self.callbacks:
                try:
                    callback(progress, self.completed_operations, self.total_operations)
                except Exception as e:
                    logger.error(f"Progress callback failed: {e}")
    
    def get_progress(self):
        """Get current progress."""
        with self.lock:
            return {
                'completed': self.completed_operations,
                'total': self.total_operations,
                'percentage': (self.completed_operations / max(self.total_operations, 1)) * 100
            }

class AsyncPipelineRunner:
    """Async pipeline runner with cancellation support."""
    
    def __init__(self):
        self.current_task = None
        self.cancellation_token = threading.Event()
        self.progress_tracker = None
    
    async def run_pipeline_async(self, pipeline_func, *args, **kwargs):
        """Run pipeline asynchronously with cancellation support."""
        self.cancellation_token.clear()
        
        try:
            kwargs['cancellation_token'] = self.cancellation_token
            
            result = await asyncio.create_task(
                self._run_with_timeout(pipeline_func, *args, **kwargs)
            )
            
            return result
            
        except asyncio.CancelledError:
            logger.info("Pipeline execution cancelled")
            raise
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise
    
    async def _run_with_timeout(self, pipeline_func, *args, **kwargs):
        """Run pipeline with timeout and cancellation checking."""
        loop = asyncio.get_event_loop()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(pipeline_func, *args, **kwargs)
            
            while not future.done():
                if self.cancellation_token.is_set():
                    future.cancel()
                    raise asyncio.CancelledError("Pipeline cancelled by user")
                
                await asyncio.sleep(0.1)  # Check every 100ms
            
            return future.result()
    
    def cancel_pipeline(self):
        """Cancel running pipeline."""
        self.cancellation_token.set()
        
        if self.current_task and not self.current_task.done():
            self.current_task.cancel()
    
    def is_running(self):
        """Check if pipeline is currently running."""
        return self.current_task and not self.current_task.done()

def run_concurrent_functions(functions_with_args, max_workers=4):
    """Simple utility to run functions concurrently."""
    with ConcurrentPipelineProcessor(max_workers) as processor:
        return asyncio.run(processor.run_concurrent_operations(functions_with_args))

def create_progress_tracker(total_operations):
    """Create a thread-safe progress tracker."""
    return ThreadSafeProgress(total_operations)
