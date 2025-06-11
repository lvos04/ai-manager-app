import logging
import traceback
from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class PipelineErrorHandler:
    """Centralized error handling for pipeline operations."""
    
    def __init__(self):
        self.error_count = 0
    
    @staticmethod
    def log_error_to_output(error: Exception, output_path: str, context: Optional[Dict[str, Any]] = None) -> None:
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
                            project_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
    
    def log_error(self, error_type: str, error_message: str, output_dir: str, 
                  context: Optional[Dict[str, Any]] = None) -> str:
        """Log error to output directory with comprehensive details."""
        try:
            import os
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
            
            # self.error_count += 1
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
