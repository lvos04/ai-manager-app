"""
Centralized logging configuration for AI Project Manager.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logging(log_level: str = "INFO", log_file: str = None) -> logging.Logger:
    """
    Setup centralized logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        
    Returns:
        Configured logger instance
    """
    
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        handlers.append(file_handler)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_log_file = log_dir / f"ai_project_manager_{timestamp}.log"
        file_handler = logging.FileHandler(default_log_file)
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
        force=True
    )
    
    logger = logging.getLogger("ai_project_manager")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("diffusers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    
    logger.info(f"Logging initialized with level {log_level}")
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f"ai_project_manager.{name}")
