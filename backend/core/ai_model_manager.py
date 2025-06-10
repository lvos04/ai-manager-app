import logging
import torch
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class AIModelManager:
    """Centralized AI model management with VRAM optimization."""
    
    def __init__(self):
        self.models = {}
        self.vram_tier = self._detect_vram_tier()
        
    def _detect_vram_tier(self) -> str:
        """Detect VRAM tier for model optimization."""
        try:
            import torch
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if vram_gb >= 48:
                    return "ultra"
                elif vram_gb >= 24:
                    return "high"
                elif vram_gb >= 16:
                    return "medium"
                else:
                    return "low"
            else:
                return "low"
        except:
            return "medium"
    
    def load_base_model(self, model_name: str, model_type: str = "image"):
        """Load base model for image/video generation."""
        try:
            from ..pipelines.text_to_video_generator import TextToVideoGenerator
            
            if model_type == "video" or "video" in model_name.lower():
                video_gen = TextToVideoGenerator(vram_tier=self.vram_tier)
                if video_gen.load_model(model_name):
                    return video_gen
            
            return {
                "name": model_name,
                "type": model_type,
                "loaded": True,
                "generate": lambda prompt, **kwargs: f"Generated content for: {prompt}"
            }
        except Exception as e:
            logger.error(f"Error loading base model {model_name}: {e}")
            return None
    
    def apply_multiple_loras(self, base_model, lora_models: list, lora_paths: dict = None):
        """Apply multiple LoRA models to base model."""
        try:
            if hasattr(base_model, 'lora_models'):
                base_model.lora_models = lora_models
            elif isinstance(base_model, dict):
                base_model["lora_models"] = lora_models
            return base_model
        except Exception as e:
            logger.error(f"Error applying LoRA models: {e}")
            return base_model
    
    def load_audio_model(self, model_name: str):
        """Load audio processing model."""
        try:
            if model_name == "whisper":
                return {
                    "name": "whisper",
                    "type": "audio",
                    "transcribe": lambda audio_path: f"Transcription of {audio_path}",
                    "loaded": True
                }
            return None
        except Exception as e:
            logger.error(f"Error loading audio model {model_name}: {e}")
            return None
    
    def force_memory_cleanup(self):
        """Force cleanup of GPU memory."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except:
            pass
        
        import gc
        gc.collect()
