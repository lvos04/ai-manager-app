"""
Pipeline Error Handler for comprehensive error logging.
"""

import os
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class PipelineErrorHandler:
    """Centralized error handling for pipeline operations."""
    
    def __init__(self):
        self.error_count = 0
    
    def log_error(self, error_type: str, error_message: str, output_dir: str, 
                  context: Optional[Dict[str, Any]] = None) -> str:
        """
        Log detailed error information to the output directory.
        
        Args:
            error_type: Type/category of the error
            error_message: Detailed error message
            output_dir: Directory to save error logs
            context: Additional context information
            
        Returns:
            str: Path to the error log file
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            error_log_path = output_path / f"error_log_{timestamp}.txt"
            error_json_path = output_path / f"error_log_{timestamp}.json"
            
            error_data = {
                "timestamp": datetime.now().isoformat(),
                "error_type": error_type,
                "error_message": error_message,
                "traceback": traceback.format_exc(),
                "context": context or {}
            }
            
            with open(error_log_path, 'w', encoding='utf-8') as f:
                f.write(f"ERROR LOG - {timestamp}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Error Type: {error_type}\n")
                f.write(f"Error Message: {error_message}\n")
                f.write(f"Timestamp: {error_data['timestamp']}\n\n")
                
                if context:
                    f.write("Context Information:\n")
                    f.write("-" * 20 + "\n")
                    for key, value in context.items():
                        f.write(f"{key}: {value}\n")
                    f.write("\n")
                
                f.write("Full Traceback:\n")
                f.write("-" * 15 + "\n")
                f.write(error_data['traceback'])
            
            with open(error_json_path, 'w', encoding='utf-8') as f:
                json.dump(error_data, f, indent=2, ensure_ascii=False)
            
            logger.error(f"Error logged to: {error_log_path}")
            return str(error_log_path)
            
        except Exception as e:
            logger.error(f"Failed to log error: {e}")
            return ""
    
    def log_error_to_output(self, error: Exception, output_path: str, 
                           context: Optional[Dict[str, Any]] = None) -> str:
        """
        Log error to output directory with exception details.
        
        Args:
            error: Exception object
            output_path: Output directory path
            context: Additional context information
            
        Returns:
            str: Path to error log file
        """
        error_type = type(error).__name__
        error_message = str(error)
        
        return self.log_error(
            error_type=error_type,
            error_message=error_message,
            output_dir=output_path,
            context=context
        )
    
    def handle_pipeline_error(self, error: Exception, output_dir: str, 
                             pipeline_stage: str = "unknown") -> Dict[str, Any]:
        """
        Handle pipeline error with comprehensive logging.
        
        Args:
            error: Exception that occurred
            output_dir: Directory to save error logs
            pipeline_stage: Stage where error occurred
            
        Returns:
            Dict containing error response
        """
        try:
            error_log_path = self.log_error_to_output(
                error=error,
                output_path=output_dir,
                context={
                    "pipeline_stage": pipeline_stage,
                    "error_count": self.error_count
                }
            )
            
            self.error_count += 1
            
            return {
                "success": False,
                "error": str(error),
                "error_type": type(error).__name__,
                "error_log_path": error_log_path,
                "pipeline_stage": pipeline_stage
            }
            
        except Exception as e:
            logger.error(f"Critical error in error handler: {e}")
            return {
                "success": False,
                "error": "Critical error in error handling system",
                "error_type": "ErrorHandlerFailure"
            }
