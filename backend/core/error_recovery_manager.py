"""
Advanced Error Recovery & Resilience for AI Project Manager.
"""

import logging
import asyncio
import time
import traceback
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
import functools

logger = logging.getLogger(__name__)

class RetryStrategy(Enum):
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    IMMEDIATE = "immediate"

@dataclass
class RetryConfig:
    max_attempts: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True

class ErrorRecoveryManager:
    """Robust error handling and recovery."""
    
    def __init__(self):
        self.retry_strategies = RetryStrategies()
        self.fallback_manager = FallbackManager()
        self.error_history = {}
        self.recovery_stats = {
            "total_errors": 0,
            "recovered_errors": 0,
            "failed_recoveries": 0
        }
        
    async def execute_with_recovery(self, func: Callable, *args, 
                                  retry_config: RetryConfig = None,
                                  fallback_func: Callable = None,
                                  context: Dict[str, Any] = None,
                                  **kwargs) -> Any:
        """Execute function with comprehensive error recovery."""
        if retry_config is None:
            retry_config = RetryConfig()
            
        if context is None:
            context = {}
            
        func_name = getattr(func, '__name__', str(func))
        
        for attempt in range(retry_config.max_attempts):
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                    
                if attempt > 0:
                    self.recovery_stats["recovered_errors"] += 1
                    logger.info(f"Function {func_name} succeeded on attempt {attempt + 1}")
                    
                return result
                
            except Exception as e:
                self.recovery_stats["total_errors"] += 1
                self._record_error(func_name, e, attempt, context)
                
                if attempt == retry_config.max_attempts - 1:
                    if fallback_func:
                        logger.warning(f"All retries failed for {func_name}, trying fallback")
                        return await self._execute_fallback(fallback_func, *args, **kwargs)
                    else:
                        self.recovery_stats["failed_recoveries"] += 1
                        logger.error(f"All retries failed for {func_name}: {e}")
                        raise
                        
                delay = self.retry_strategies.calculate_delay(retry_config, attempt)
                logger.warning(f"Attempt {attempt + 1} failed for {func_name}: {e}. Retrying in {delay:.2f}s")
                await asyncio.sleep(delay)
                
    async def _execute_fallback(self, fallback_func: Callable, *args, **kwargs) -> Any:
        """Execute fallback function."""
        try:
            if asyncio.iscoroutinefunction(fallback_func):
                return await fallback_func(*args, **kwargs)
            else:
                return fallback_func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Fallback function also failed: {e}")
            raise
            
    def _record_error(self, func_name: str, error: Exception, attempt: int, context: Dict[str, Any]):
        """Record error for analysis."""
        error_key = f"{func_name}_{type(error).__name__}"
        
        if error_key not in self.error_history:
            self.error_history[error_key] = []
            
        self.error_history[error_key].append({
            "timestamp": time.time(),
            "attempt": attempt,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "context": context
        })
        
        if len(self.error_history[error_key]) > 100:
            self.error_history[error_key] = self.error_history[error_key][-50:]
            
    def implement_smart_retry(self, retry_config: RetryConfig = None):
        """Decorator for implementing smart retry mechanisms."""
        if retry_config is None:
            retry_config = RetryConfig()
            
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                return await self.execute_with_recovery(func, *args, retry_config=retry_config, **kwargs)
            return wrapper
        return decorator
        
    def create_fallback_pipelines(self, primary_pipeline: Callable, 
                                fallback_pipelines: List[Callable]) -> Callable:
        """Create fallback pipelines for reliability."""
        async def pipeline_with_fallbacks(*args, **kwargs):
            try:
                return await self.execute_with_recovery(primary_pipeline, *args, **kwargs)
            except Exception as e:
                logger.warning(f"Primary pipeline failed: {e}")
                
                for i, fallback in enumerate(fallback_pipelines):
                    try:
                        logger.info(f"Trying fallback pipeline {i + 1}")
                        if asyncio.iscoroutinefunction(fallback):
                            return await fallback(*args, **kwargs)
                        else:
                            return fallback(*args, **kwargs)
                    except Exception as fallback_error:
                        logger.warning(f"Fallback pipeline {i + 1} failed: {fallback_error}")
                        continue
                        
                raise Exception("All pipelines failed")
                
        return pipeline_with_fallbacks
        
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error recovery statistics."""
        total_errors = self.recovery_stats["total_errors"]
        recovery_rate = 0
        
        if total_errors > 0:
            recovery_rate = (self.recovery_stats["recovered_errors"] / total_errors) * 100
            
        return {
            "total_errors": total_errors,
            "recovered_errors": self.recovery_stats["recovered_errors"],
            "failed_recoveries": self.recovery_stats["failed_recoveries"],
            "recovery_rate_percent": recovery_rate,
            "error_types": list(self.error_history.keys())
        }
        
    def get_error_history(self, func_name: str = None) -> Dict[str, List[Dict]]:
        """Get error history for analysis."""
        if func_name:
            return {k: v for k, v in self.error_history.items() if func_name in k}
        return self.error_history.copy()

class RetryStrategies:
    """Different retry strategies implementation."""
    
    def calculate_delay(self, config: RetryConfig, attempt: int) -> float:
        """Calculate delay based on retry strategy."""
        if config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = config.base_delay * (config.backoff_multiplier ** attempt)
        elif config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = config.base_delay * (attempt + 1)
        elif config.strategy == RetryStrategy.FIXED_DELAY:
            delay = config.base_delay
        else:  # IMMEDIATE
            delay = 0
            
        delay = min(delay, config.max_delay)
        
        if config.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)
            
        return delay

class FallbackManager:
    """Manage fallback strategies."""
    
    def __init__(self):
        self.fallback_registry = {}
        
    def register_fallback(self, operation_name: str, fallback_func: Callable):
        """Register a fallback function for an operation."""
        self.fallback_registry[operation_name] = fallback_func
        
    def get_fallback(self, operation_name: str) -> Optional[Callable]:
        """Get registered fallback for an operation."""
        return self.fallback_registry.get(operation_name)
        
    def create_degraded_quality_fallback(self, original_func: Callable) -> Callable:
        """Create a degraded quality fallback."""
        async def degraded_fallback(*args, **kwargs):
            kwargs['quality'] = 'low'
            kwargs['steps'] = kwargs.get('steps', 20) // 2
            kwargs['resolution'] = kwargs.get('resolution', 512) // 2
            
            if asyncio.iscoroutinefunction(original_func):
                return await original_func(*args, **kwargs)
            else:
                return original_func(*args, **kwargs)
                
        return degraded_fallback

_error_recovery_manager = None

def get_error_recovery_manager():
    """Get global error recovery manager instance."""
    global _error_recovery_manager
    if _error_recovery_manager is None:
        _error_recovery_manager = ErrorRecoveryManager()
    return _error_recovery_manager

def with_retry(retry_config: RetryConfig = None, fallback_func: Callable = None):
    """Decorator for adding retry functionality to functions."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            manager = get_error_recovery_manager()
            return await manager.execute_with_recovery(
                func, *args, 
                retry_config=retry_config,
                fallback_func=fallback_func,
                **kwargs
            )
        return wrapper
    return decorator
