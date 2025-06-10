import logging
import traceback
from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class PipelineErrorHandler:
    """Centralized error handling for pipeline operations."""
    
    @staticmethod
    def log_error_to_output(error: Exception, output_path: str, context: Dict[str, Any] = None) -> None:
        """Log complete error information to output folder."""
        try:
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            error_file = output_dir / "error_log.txt"
            
            error_info = {
                "timestamp": datetime.now().isoformat(),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
                "context": context or {}
            }
            
            with open(error_file, 'w', encoding='utf-8') as f:
                f.write(f"Pipeline Error Report\n")
                f.write(f"Generated: {error_info['timestamp']}\n")
                f.write(f"{'='*50}\n\n")
                f.write(f"Error Type: {error_info['error_type']}\n")
                f.write(f"Error Message: {error_info['error_message']}\n\n")
                f.write(f"Context Information:\n")
                for key, value in error_info['context'].items():
                    f.write(f"  {key}: {value}\n")
                f.write(f"\nFull Traceback:\n")
                f.write(error_info['traceback'])
            
            error_json = output_dir / "error_details.json"
            with open(error_json, 'w', encoding='utf-8') as f:
                json.dump(error_info, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Error details saved to {error_file}")
            
        except Exception as e:
            logger.error(f"Failed to save error log: {e}")
    
    @staticmethod
    def handle_pipeline_error(error: Exception, pipeline_type: str, output_path: str, 
                            project_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle pipeline errors with comprehensive logging."""
        context = {
            "pipeline_type": pipeline_type,
            "project_data": project_data or {},
            "output_path": output_path
        }
        
        PipelineErrorHandler.log_error_to_output(error, output_path, context)
        
        return {
            "success": False,
            "error": str(error),
            "error_type": type(error).__name__,
            "message": f"Pipeline {pipeline_type} failed: {str(error)}",
            "error_log_path": str(Path(output_path) / "error_log.txt")
        }
