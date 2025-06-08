"""
Custom exceptions for AI Project Manager.
"""

class PipelineError(Exception):
    """Base exception for pipeline-related errors."""
    pass

class ModelLoadError(PipelineError):
    """Exception raised when model loading fails."""
    pass

class GenerationError(PipelineError):
    """Exception raised when content generation fails."""
    pass

class ConfigurationError(PipelineError):
    """Exception raised when configuration is invalid."""
    pass

class ResourceError(PipelineError):
    """Exception raised when system resources are insufficient."""
    pass

class ValidationError(PipelineError):
    """Exception raised when input validation fails."""
    pass
