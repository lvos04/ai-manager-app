"""
Batch processing utilities for optimized GPU utilization.
"""

import logging
from typing import List, Any, Optional
import torch

logger = logging.getLogger(__name__)

class BatchImageGenerator:
    """Optimized batch image generation for better GPU utilization."""
    
    def __init__(self, model, batch_size=4, max_batch_size=8):
        self.model = model
        self.batch_size = batch_size
        self.max_batch_size = max_batch_size
        self.device = next(model.parameters()).device if hasattr(model, 'parameters') else 'cpu'
    
    def generate_batch(self, prompts: List[str], **kwargs) -> List[Any]:
        """Generate multiple images in batches for better GPU utilization."""
        if not prompts:
            return []
        
        effective_batch_size = self._get_optimal_batch_size(len(prompts))
        
        results = []
        try:
            for i in range(0, len(prompts), effective_batch_size):
                batch_prompts = prompts[i:i + effective_batch_size]
                
                logger.info(f"Processing batch {i//effective_batch_size + 1}: {len(batch_prompts)} prompts")
                
                batch_results = self._generate_single_batch(batch_prompts, **kwargs)
                
                if isinstance(batch_results, list):
                    results.extend(batch_results)
                else:
                    results.append(batch_results)
                
                self._clear_cache()
            
            return results
            
        except Exception as e:
            logger.error(f"Batch generation failed: {e}")
            return self._fallback_single_generation(prompts, **kwargs)
    
    def _generate_single_batch(self, batch_prompts: List[str], **kwargs) -> Any:
        """Generate a single batch of images."""
        try:
            if hasattr(self.model, '__call__'):
                return self.model(batch_prompts, **kwargs)
            elif hasattr(self.model, 'generate'):
                return self.model.generate(batch_prompts, **kwargs)
            else:
                return [self.model(prompt, **kwargs) for prompt in batch_prompts]
                
        except Exception as e:
            logger.error(f"Single batch generation failed: {e}")
            raise
    
    def _get_optimal_batch_size(self, total_prompts: int) -> int:
        """Determine optimal batch size based on available memory."""
        try:
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                allocated_memory = torch.cuda.memory_allocated()
                available_memory = gpu_memory - allocated_memory
                
                if available_memory > 8 * (1024**3):  # 8GB+
                    optimal_batch_size = min(self.max_batch_size, total_prompts)
                elif available_memory > 4 * (1024**3):  # 4-8GB
                    optimal_batch_size = min(self.batch_size, total_prompts)
                else:  # <4GB
                    optimal_batch_size = min(2, total_prompts)
                
                logger.info(f"Optimal batch size: {optimal_batch_size} (available memory: {available_memory/(1024**3):.1f}GB)")
                return optimal_batch_size
            
        except Exception as e:
            logger.warning(f"Could not determine optimal batch size: {e}")
        
        return min(self.batch_size, total_prompts)
    
    def _clear_cache(self):
        """Clear GPU cache after batch processing."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"Could not clear cache: {e}")
    
    def _fallback_single_generation(self, prompts: List[str], **kwargs) -> List[Any]:
        """Fallback to single image generation if batch fails."""
        logger.warning("Falling back to single image generation")
        results = []
        
        for prompt in prompts:
            try:
                if hasattr(self.model, '__call__'):
                    result = self.model([prompt], **kwargs)
                    if hasattr(result, 'images'):
                        results.extend(result.images)
                    else:
                        results.append(result)
                else:
                    results.append(self.model(prompt, **kwargs))
                    
            except Exception as e:
                logger.error(f"Single generation failed for prompt: {e}")
                results.append(None)
        
        return results

class PipelineModelCache:
    """Cache models to avoid reloading in loops."""
    
    def __init__(self, max_cache_size=5):
        self.cached_models = {}
        self.access_order = []
        self.max_cache_size = max_cache_size
    
    def get_or_load_model(self, model_type: str, model_name: str, loader_func, **kwargs):
        """Cache models to avoid reloading in loops."""
        cache_key = f"{model_type}:{model_name}"
        
        if cache_key in self.cached_models:
            self._update_access_order(cache_key)
            logger.info(f"Using cached model: {cache_key}")
            return self.cached_models[cache_key]
        
        logger.info(f"Loading new model: {cache_key}")
        model = loader_func(model_name, **kwargs)
        
        self._add_to_cache(cache_key, model)
        
        return model
    
    def _add_to_cache(self, key: str, model: Any):
        """Add model to cache with LRU eviction."""
        if len(self.cached_models) >= self.max_cache_size:
            lru_key = self.access_order.pop(0)
            if lru_key in self.cached_models:
                del self.cached_models[lru_key]
                logger.info(f"Evicted model from cache: {lru_key}")
        
        self.cached_models[key] = model
        self.access_order.append(key)
        logger.info(f"Added model to cache: {key}")
    
    def _update_access_order(self, key: str):
        """Update access order for LRU."""
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
    
    def clear_cache(self):
        """Clear all cached models."""
        self.cached_models.clear()
        self.access_order.clear()
        logger.info("Model cache cleared")
    
    def get_cache_info(self):
        """Get cache information."""
        return {
            'cached_models': list(self.cached_models.keys()),
            'cache_size': len(self.cached_models),
            'max_size': self.max_cache_size,
            'access_order': self.access_order.copy()
        }

global_model_cache = PipelineModelCache()

def get_batch_generator(model, batch_size=4):
    """Get a batch generator for the given model."""
    return BatchImageGenerator(model, batch_size)

def get_model_cache():
    """Get the global model cache instance."""
    return global_model_cache
