"""
Centralized error handling for pipeline operations.
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
    """Centralized error handler for pipeline operations."""
    
    def __init__(self):
        self.error_count = 0
    
    def log_error(self, error_type: str, error_message: str, output_dir: str, 
                  context: Optional[Dict[str, Any]] = None) -> str:
        """
        Log error to output directory with comprehensive details.
        
        Args:
            error_type: Type of error (e.g., "MODEL_LOADING_FAILURE")
            error_message: Detailed error message
            output_dir: Directory to write error log
            context: Additional context information
            
        Returns:
            Path to created error log file
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            error_filename = f"error_log_{timestamp}.txt"
            error_path = os.path.join(output_dir, error_filename)
            
            error_details = {
                "timestamp": datetime.now().isoformat(),
                "error_type": error_type,
                "error_message": error_message,
                "context": context or {},
                "traceback": traceback.format_exc() if traceback.format_exc().strip() != "NoneType: None" else None
            }
            
            with open(error_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write(f"PIPELINE ERROR LOG - {error_details['timestamp']}\n")
                f.write("=" * 80 + "\n\n")
                
                f.write(f"ERROR TYPE: {error_details['error_type']}\n\n")
                f.write(f"ERROR MESSAGE:\n{error_details['error_message']}\n\n")
                
                if error_details['context']:
                    f.write("CONTEXT:\n")
                    for key, value in error_details['context'].items():
                        f.write(f"  {key}: {value}\n")
                    f.write("\n")
                
                if error_details['traceback']:
                    f.write("TRACEBACK:\n")
                    f.write(error_details['traceback'])
                    f.write("\n")
                
                f.write("=" * 80 + "\n")
                f.write("END OF ERROR LOG\n")
                f.write("=" * 80 + "\n")
            
            self.error_count += 1
            logger.error(f"Error logged to: {error_path}")
            
            return error_path
            
        except Exception as e:
            logger.error(f"Failed to log error to output directory: {e}")
            try:
                temp_path = os.path.join("/tmp", f"pipeline_error_{timestamp}.txt")
                with open(temp_path, 'w') as f:
                    f.write(f"PIPELINE ERROR: {error_message}\n")
                    f.write(f"Original output dir: {output_dir}\n")
                    f.write(f"Logging error: {e}\n")
                return temp_path
            except:
                logger.critical(f"Complete failure to log error: {error_message}")
                return ""
