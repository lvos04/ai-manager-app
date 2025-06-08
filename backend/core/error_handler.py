"""
Standardized error handling for AI Project Manager.
"""

import logging
import functools
from typing import Any, Callable, Optional
from .exceptions import PipelineError, ModelLoadError, GenerationError

logger = logging.getLogger(__name__)

def handle_pipeline_error(fallback_result: Any = None):
    """Decorator for standardized pipeline error handling."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except PipelineError as e:
                logger.error(f"Pipeline error in {func.__name__}: {e}")
                return create_fallback_result(e, fallback_result)
            except Exception as e:
                logger.error(f"Unexpected error in {func.__name__}: {e}")
                return create_fallback_result(e, fallback_result)
        return wrapper
    return decorator

def create_fallback_result(error: Exception, fallback_result: Any = None) -> Any:
    """Create a fallback result when an error occurs."""
    if fallback_result is not None:
        return fallback_result
    
    if isinstance(error, ModelLoadError):
        return None
    elif isinstance(error, GenerationError):
        return False
    else:
        return None

def log_and_continue(func: Callable) -> Callable:
    """Decorator that logs errors but continues execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Non-critical error in {func.__name__}: {e}")
            return None
    return wrapper

def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator that retries function execution on failure."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import time
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Function {func.__name__} failed after {max_retries} attempts: {e}")
                        raise
                    else:
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying...")
                        time.sleep(delay)
            
        return wrapper
    return decorator
