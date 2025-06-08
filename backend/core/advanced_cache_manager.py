"""
Advanced caching system for AI models and expensive operations.
"""

import logging
import pickle
import hashlib
import time
from pathlib import Path
from typing import Any, Optional, Dict, Callable
from functools import wraps
import threading
import weakref

logger = logging.getLogger(__name__)

class LRUCache:
    """Thread-safe LRU cache with persistence support."""
    
    def __init__(self, max_size=10, persist_path=None):
        self.max_size = max_size
        self.persist_path = Path(persist_path) if persist_path else None
        self.cache = {}
        self.access_order = []
        self.lock = threading.RLock()
        self.hit_count = 0
        self.miss_count = 0
        
        self._load_persisted_cache()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self.lock:
            if key in self.cache:
                self._update_access_order(key)
                self.hit_count += 1
                logger.debug(f"Cache hit for key: {key}")
                return self.cache[key]['value']
            
            self.miss_count += 1
            logger.debug(f"Cache miss for key: {key}")
            return None
    
    def put(self, key: str, value: Any, metadata: Dict = None):
        """Put item in cache with LRU eviction."""
        with self.lock:
            if key in self.cache:
                self.access_order.remove(key)
            
            while len(self.cache) >= self.max_size:
                lru_key = self.access_order.pop(0)
                del self.cache[lru_key]
                logger.debug(f"Evicted from cache: {lru_key}")
            
            self.cache[key] = {
                'value': value,
                'timestamp': time.time(),
                'metadata': metadata or {}
            }
            self.access_order.append(key)
            
            logger.debug(f"Added to cache: {key}")
            
            self._persist_cache()
    
    def _update_access_order(self, key: str):
        """Update access order for LRU."""
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
    
    def _load_persisted_cache(self):
        """Load cache from persistent storage."""
        if not self.persist_path or not self.persist_path.exists():
            return
        
        try:
            with open(self.persist_path, 'rb') as f:
                data = pickle.load(f)
                self.cache = data.get('cache', {})
                self.access_order = data.get('access_order', [])
                logger.info(f"Loaded {len(self.cache)} items from persistent cache")
        except Exception as e:
            logger.warning(f"Could not load persistent cache: {e}")
    
    def _persist_cache(self):
        """Persist cache to storage."""
        if not self.persist_path:
            return
        
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_path, 'wb') as f:
                pickle.dump({
                    'cache': self.cache,
                    'access_order': self.access_order
                }, f)
        except Exception as e:
            logger.warning(f"Could not persist cache: {e}")
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.hit_count = 0
            self.miss_count = 0
    
    def get_stats(self):
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hit_count + self.miss_count
            hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_count': self.hit_count,
                'miss_count': self.miss_count,
                'hit_rate': hit_rate,
                'keys': list(self.cache.keys())
            }
    
    def get_model(self, key: str) -> Optional[Any]:
        """Get model from cache (alias for get method)."""
        return self.get(key)
    
    def cache_model(self, key: str, model: Any, metadata: Dict = None):
        """Cache model (alias for put method)."""
        self.put(key, model, metadata)

class GlobalModelCache:
    """Global cache for AI models with intelligent management."""
    
    def __init__(self, max_size=5, cache_dir="cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_cache = LRUCache(
            max_size=max_size,
            persist_path=self.cache_dir / "model_cache.pkl"
        )
        
        self.operation_cache = LRUCache(
            max_size=50,
            persist_path=self.cache_dir / "operation_cache.pkl"
        )
        
        self.model_refs = weakref.WeakValueDictionary()
    
    def get_or_load_model(self, model_type: str, model_name: str, loader_func: Callable, **kwargs):
        """Get cached model or load if not cached."""
        cache_key = self._generate_model_key(model_type, model_name, **kwargs)
        
        cached_model = self.model_cache.get(cache_key)
        if cached_model is not None:
            return cached_model
        
        logger.info(f"Loading model: {model_type}:{model_name}")
        start_time = time.time()
        
        try:
            model = loader_func(model_name, **kwargs)
            load_time = time.time() - start_time
            
            metadata = {
                'model_type': model_type,
                'model_name': model_name,
                'load_time': load_time,
                'kwargs': kwargs
            }
            
            self.model_cache.put(cache_key, model, metadata)
            
            self.model_refs[cache_key] = model
            
            logger.info(f"Model loaded and cached in {load_time:.2f}s: {cache_key}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {cache_key}: {e}")
            raise
    
    def cache_operation_result(self, operation_name: str, inputs: Any, result: Any):
        """Cache expensive operation results."""
        cache_key = self._generate_operation_key(operation_name, inputs)
        
        metadata = {
            'operation': operation_name,
            'timestamp': time.time()
        }
        
        self.operation_cache.put(cache_key, result, metadata)
    
    def get_cached_operation_result(self, operation_name: str, inputs: Any) -> Optional[Any]:
        """Get cached operation result."""
        cache_key = self._generate_operation_key(operation_name, inputs)
        return self.operation_cache.get(cache_key)
    
    def _generate_model_key(self, model_type: str, model_name: str, **kwargs) -> str:
        """Generate cache key for model."""
        key_data = f"{model_type}:{model_name}:{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _generate_operation_key(self, operation_name: str, inputs: Any) -> str:
        """Generate cache key for operation."""
        key_data = f"{operation_name}:{str(inputs)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get_cache_stats(self):
        """Get comprehensive cache statistics."""
        return {
            'model_cache': self.model_cache.get_stats(),
            'operation_cache': self.operation_cache.get_stats(),
            'active_model_refs': len(self.model_refs)
        }
    
    def clear_all_caches(self):
        """Clear all caches."""
        self.model_cache.clear()
        self.operation_cache.clear()
        self.model_refs.clear()
    
    def get_cached_content(self, content_hash: str, content_type: str) -> Optional[str]:
        """Get cached content by hash and type."""
        cache_key = f"{content_type}:{content_hash}"
        return self.get_cached_operation_result(f"content_{content_type}", cache_key)
    
    def cache_generated_content(self, content_hash: str, file_path: str, content_type: str):
        """Cache generated content file path."""
        cache_key = f"{content_type}:{content_hash}"
        self.cache_operation_result(f"content_{content_type}", cache_key, file_path)

global_cache = GlobalModelCache()

def cached_operation(operation_name: str):
    """Decorator for caching expensive operations."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            inputs = (args, tuple(sorted(kwargs.items())))
            cached_result = global_cache.get_cached_operation_result(operation_name, inputs)
            
            if cached_result is not None:
                logger.debug(f"Using cached result for {operation_name}")
                return cached_result
            
            result = func(*args, **kwargs)
            global_cache.cache_operation_result(operation_name, inputs, result)
            
            return result
        return wrapper
    return decorator

def get_cache_manager():
    """Get cache manager instance."""
    return global_cache

def get_global_cache():
    """Get the global cache instance."""
    return global_cache
