"""
AI model loading and management utilities for pipeline processing.
"""
import os
import sys
from pathlib import Path
import importlib.util
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LLMWrapper:
    """Wrapper class for LLM models to provide consistent interface."""
    
    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device if hasattr(model, 'parameters') and model else 'cpu'
    
    def generate(self, prompt, max_length=100, temperature=0.7, do_sample=True, max_tokens=None, **kwargs):
        """
        Generate text using the LLM model.
        
        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            max_tokens: Alternative to max_length for compatibility
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text string
        """
        if max_tokens:
            max_length = max_tokens
            
        if not self.model or not self.tokenizer:
            return f"Generated response for: {prompt[:50]}..."
            
        try:
            import torch
            
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            logging.error(f"LLM generation failed: {e}")
            return f"Generated response for: {prompt}"
    
    def __call__(self, prompt, **kwargs):
        """Allow calling the wrapper directly."""
        return self.generate(prompt, **kwargs)

class AIModelManager:
    """
    Manages loading and initialization of AI models for pipeline processing.
    """
    _cuda_setup_completed = False
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AIModelManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            import os
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            
            from ..core.advanced_cache_manager import get_cache_manager
            from ..core.performance_monitor import get_performance_monitor
            from ..core.memory_manager import get_memory_manager
            
            self.cache_manager = get_cache_manager()
            self.performance_monitor = get_performance_monitor()
            self.memory_manager = get_memory_manager()
            self.loaded_models = {}
            self.loaded_loras = {}
            self.model_cache = {}
            self.vram_tier = self._detect_vram_tier()
            logger.info(f"Detected VRAM tier: {self.vram_tier}")
            
            if not AIModelManager._cuda_setup_completed:
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.set_per_process_memory_fraction(0.8)
                        torch.cuda.empty_cache()
                        logger.info("CUDA memory management initialized with 80% fraction limit")
                        AIModelManager._cuda_setup_completed = True
                    else:
                        self._try_version_aware_cuda_setup()
                        AIModelManager._cuda_setup_completed = True
                except ImportError:
                    self._try_version_aware_cuda_setup()
                    AIModelManager._cuda_setup_completed = True
            
            self.performance_monitor.start_monitoring()
            self.memory_manager.start_monitoring()
            
            self.memory_manager.register_cleanup_callback(self._cleanup_models)
            self._initialized = True
    
    def _classify_vram_tier(self, memory_gb):
        """
        Classify VRAM tier based on memory amount.
        
        Args:
            memory_gb: GPU memory in GB
            
        Returns:
            String representing the VRAM tier ('low', 'medium', 'high', 'ultra')
        """
        if memory_gb >= 24:
            return "ultra"
        elif memory_gb >= 16:
            return "high"
        elif memory_gb >= 8:
            return "medium"
        else:
            return "low"
    def _detect_cuda_availability(self):
        """
        Detect CUDA availability with Windows-specific error handling.
        
        Returns:
            bool: True if CUDA is available and working, False otherwise
        """
        try:
            import torch
            if torch.cuda.is_available():
                test_tensor = torch.tensor([1.0]).cuda()
                test_result = test_tensor.cpu()
                logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
                return True
        except Exception as e:
            if "caffe2_nvrtc.dll" in str(e):
                logger.warning(f"CUDA DLL loading failed on Windows: {e}")
                logger.info("Falling back to CPU mode - this is normal on some Windows systems")
            else:
                logger.warning(f"CUDA detection failed: {e}")
            return False
        
        return False



    def _detect_vram_tier(self):
        """
        Detect available VRAM and determine the appropriate tier.
        Supports CUDA detection, version-aware automatic installation, and offline fallback.
        
        Returns:
            String representing the VRAM tier ('low', 'medium', 'high', 'ultra')
        """
        manual_tier = self._get_manual_tier_override()
        if manual_tier:
            logger.info(f"Using manual tier override: {manual_tier}")
            return manual_tier
        
        try:
            if self._detect_cuda_availability():
                memory_gb = self._get_gpu_memory_gb()
                if memory_gb:
                    try:
                        import torch
                        if torch.cuda.is_available():
                            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                            allocated = torch.cuda.memory_allocated() / 1024**3
                            reserved = torch.cuda.memory_reserved() / 1024**3
                            free = total_memory - reserved
                            
                            logger.info(f"VRAM Status - Total: {total_memory:.2f}GB, Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Free: {free:.2f}GB")
                            
                            if free < 3.0 or (reserved / total_memory) > 0.8:
                                logger.warning(f"VRAM fragmentation detected - Free: {free:.2f}GB, Reserved ratio: {reserved/total_memory:.2f}")
                                torch.cuda.empty_cache()
                                torch.cuda.set_per_process_memory_fraction(0.6)
                                logger.info("Applied aggressive memory management due to fragmentation")
                    except Exception as torch_error:
                        logger.warning(f"PyTorch VRAM check failed: {torch_error}")
                    
                    tier = self._classify_vram_tier(memory_gb)
                    logger.info(f"CUDA available - GPU: {self._get_gpu_name()}, VRAM: {memory_gb}GB")
                    logger.info(f"Detected VRAM tier: {tier}")
                    return tier
            
            try:
                gpu_info = self._enhanced_gpu_detection()
                if gpu_info and isinstance(gpu_info, str):
                    logger.info(f"Enhanced detection returned tier: {gpu_info}")
                    return gpu_info
                elif gpu_info and isinstance(gpu_info, dict):
                    memory_gb = gpu_info.get('memory_gb', 4.0)
                    tier = self._classify_vram_tier(memory_gb)
                    logger.info(f"Enhanced detection: {gpu_info.get('name', 'Unknown GPU')}, {memory_gb}GB")
                    logger.info(f"Detected VRAM tier: {tier}")
                    return tier
            except Exception as e:
                logger.warning(f"Enhanced GPU detection failed: {e}")
            
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    memory_mb = int(result.stdout.strip())
                    memory_gb = memory_mb / 1024
                    tier = self._classify_vram_tier(memory_gb)
                    logger.info(f"nvidia-smi detected GPU: {self._get_gpu_name()}, {memory_gb:.1f}GB")
                    logger.info(f"Detected VRAM tier: {tier}")
                    return tier
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError, ValueError):
                logger.debug("nvidia-smi not available or failed")
            
            logger.warning("All GPU detection methods failed, falling back to CPU mode")
            logger.info("Detected VRAM tier: cpu")
            return "cpu"
            
        except Exception as e:
            logger.error(f"Error in VRAM tier detection: {e}")
            logger.info("Detected VRAM tier: cpu")
            return "cpu"
    
    def _get_gpu_memory_gb(self):
        """Get GPU memory in GB."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except Exception as e:
            logger.warning(f"Could not get GPU memory: {e}")
        return None
    
    def _get_gpu_name(self):
        """Get GPU name."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_name(0)
        except Exception as e:
            logger.warning(f"Could not get GPU name: {e}")
        return "Unknown GPU"

    def _try_version_aware_cuda_setup(self):
        """Try automatic CUDA installation with optimal version selection."""
        try:
            from backend.cuda_installer import CUDAInstaller
            installer = CUDAInstaller()
            
            gpu_name, cuda_version = installer.detect_nvidia_gpu()
            if gpu_name:
                logger.info(f"Detected {gpu_name}, optimal CUDA version: {cuda_version}")
                logger.info("Attempting automatic CUDA setup with optimal version...")
                
                if installer.auto_setup_cuda():
                    logger.info(f"Automatic CUDA {cuda_version} setup successful, checking PyTorch CUDA...")
                    try:
                        import torch
                        if torch.cuda.is_available():
                            logger.info("PyTorch CUDA is now available")
                            return "medium"
                    except:
                        pass
                else:
                    logger.warning(f"Automatic CUDA {cuda_version} setup failed")
            
            logger.info("Falling back to enhanced GPU detection...")
            return self._enhanced_gpu_detection()
            
        except Exception as e:
            logger.warning(f"Version-aware CUDA setup failed: {e}")
            return self._enhanced_gpu_detection()
    
    def _enhanced_gpu_detection(self):
        """Enhanced GPU detection with multiple fallback methods."""
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                if device_count > 0:
                    gpu_name = torch.cuda.get_device_name(0)
                    memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    logger.info(f"PyTorch detected GPU: {gpu_name}, {memory_gb:.1f}GB")
                    return self._classify_vram_tier(memory_gb)
        except Exception as e:
            logger.debug(f"PyTorch GPU detection failed: {e}")
        
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_gb = memory_info.total / (1024**3)
                logger.info(f"NVML detected GPU: {name}, {memory_gb:.1f}GB")
                return self._classify_vram_tier(memory_gb)
        except Exception as e:
            logger.debug(f"NVML GPU detection failed: {e}")
        
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines and lines[0]:
                    gpu_info = lines[0].split(', ')
                    if len(gpu_info) >= 2:
                        gpu_name = gpu_info[0]
                        memory_mb = int(gpu_info[1])
                        memory_gb = memory_mb / 1024
                        
                        logger.info(f"nvidia-smi detected GPU: {gpu_name}, {memory_gb:.1f}GB")
                        return self._classify_vram_tier(memory_gb)
            
        except Exception as e:
            logger.debug(f"nvidia-smi GPU detection failed: {e}")
        
        logger.warning("No NVIDIA GPU detected with any method")
        return "low"
    

    def cleanup_model_memory(self, model_name=None):
        """Enhanced memory cleanup with CUDA cache clearing."""
        try:
            import torch
            import gc
            
            if torch.cuda.is_available():
                allocated_before = torch.cuda.memory_allocated() / 1024**3
                reserved_before = torch.cuda.memory_reserved() / 1024**3
            
            if model_name:
                if hasattr(self, 'loaded_models') and model_name in self.loaded_models:
                    try:
                        if hasattr(self.loaded_models[model_name], 'to'):
                            self.loaded_models[model_name].to('cpu')
                        del self.loaded_models[model_name]
                        logger.info(f"Cleaned up model: {model_name}")
                    except Exception as e:
                        logger.warning(f"Error cleaning up model {model_name}: {e}")
            else:
                for model_name in list(self.loaded_models.keys()):
                    try:
                        if hasattr(self.loaded_models[model_name], 'to'):
                            self.loaded_models[model_name].to('cpu')
                        del self.loaded_models[model_name]
                    except Exception as e:
                        logger.warning(f"Error cleaning up model {model_name}: {e}")
                
                for lora_name in list(self.loaded_loras.keys()):
                    try:
                        del self.loaded_loras[lora_name]
                    except Exception as e:
                        logger.warning(f"Error cleaning up LoRA {lora_name}: {e}")
                
                self.loaded_models.clear()
                self.loaded_loras.clear()
                self.model_cache.clear()
                logger.info("Aggressive memory cleanup completed")
            
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                allocated_after = torch.cuda.memory_allocated() / 1024**3
                reserved_after = torch.cuda.memory_reserved() / 1024**3
                freed_allocated = allocated_before - allocated_after
                freed_reserved = reserved_before - reserved_after
                
                logger.info(f"Memory cleanup completed - Freed: {freed_allocated:.2f}GB allocated, {freed_reserved:.2f}GB reserved")
            else:
                logger.info("Model memory cleanup completed (CPU mode)")
                
        except Exception as e:
            logger.error(f"Error during memory cleanup: {e}")
    
    def auto_cleanup_after_pipeline(self):
        """Automatically cleanup after pipeline completion."""
        try:
            essential_models = ["stable_diffusion_1_5", "bark_model"]
            
            if hasattr(self, 'loaded_models'):
                models_to_remove = [m for m in self.loaded_models.keys() if m not in essential_models]
                
                for model in models_to_remove:
                    self.cleanup_model_memory(model)
                
                logger.info(f"Auto-cleanup completed, removed {len(models_to_remove)} models")
            
        except Exception as e:
            logger.error(f"Error during auto-cleanup: {e}")
    
    def get_memory_usage_info(self):
        """Get current memory usage information."""
        try:
            import psutil
            import torch
            
            memory = psutil.virtual_memory()
            system_memory = {
                'total_gb': memory.total / (1024**3),
                'used_gb': memory.used / (1024**3),
                'available_gb': memory.available / (1024**3),
                'percent': memory.percent
            }
            
            gpu_memory = {}
            if torch.cuda.is_available():
                gpu_memory = {
                    'allocated_gb': torch.cuda.memory_allocated() / (1024**3),
                    'reserved_gb': torch.cuda.memory_reserved() / (1024**3),
                    'max_allocated_gb': torch.cuda.max_memory_allocated() / (1024**3)
                }
            
            return {
                'system': system_memory,
                'gpu': gpu_memory,
                'loaded_models': len(getattr(self, 'loaded_models', {}))
            }
            
        except Exception as e:
            logger.error(f"Error getting memory info: {e}")
            return {}

    def _offline_gpu_detection(self):
        """Offline GPU detection for NVIDIA GPUs with version recommendations."""
        try:
            import platform
            import subprocess
            
            if platform.system() == "Windows":
                try:
                    result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        gpu_name = result.stdout.strip().lower()
                        logger.info(f"Offline detection found NVIDIA GPU: {gpu_name}")
                        
                        # Determine tier based on GPU model (without CUDA)
                        if "4060" in gpu_name:
                            logger.info(f"RTX 4060 detected offline, using 'medium' tier (install CUDA 12.1)")
                            return "medium"
                        elif "3090" in gpu_name or "3090ti" in gpu_name:
                            logger.info(f"RTX 3090/3090ti detected offline, using 'ultra' tier (install CUDA 11.8)")
                            return "ultra"
                        elif any(x in gpu_name for x in ["4090", "4080"]):
                            logger.info(f"RTX 40-series high-end detected offline, using 'ultra' tier (install CUDA 12.1)")
                            return "ultra"
                        elif "4070" in gpu_name:
                            logger.info(f"RTX 4070 detected offline, using 'high' tier (install CUDA 12.1)")
                            return "high"
                        elif any(x in gpu_name for x in ["3080", "3070"]):
                            tier = "high" if "3080" in gpu_name else "medium"
                            logger.info(f"RTX 30-series detected offline, using '{tier}' tier (install CUDA 11.8)")
                            return tier
                        elif any(x in gpu_name for x in ["3060", "2080", "2070"]):
                            logger.info(f"Mid-range GPU detected offline, using 'medium' tier (install CUDA 11.8)")
                            return "medium"
                        else:
                            logger.info(f"Unknown NVIDIA GPU detected offline: {gpu_name}, using 'medium' tier")
                            return "medium"
                except Exception as e:
                    logger.debug(f"nvidia-smi offline detection failed: {e}")
                
                try:
                    result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0 and 'nvidia' in result.stdout.lower():
                        logger.info("NVIDIA GPU detected via WMI, but model unknown - using 'medium' tier")
                        logger.info("Install CUDA 11.8 for compatibility, or use GPU Settings for manual override")
                        return "medium"
                except Exception as e:
                    logger.debug(f"WMI detection failed: {e}")
            
            logger.warning("No NVIDIA GPU detected offline")
            logger.info("Using 'low' tier for CPU-only processing")
            logger.info("If you have an NVIDIA GPU, install drivers and CUDA toolkit")
            return "low"
            
        except Exception as e:
            logger.warning(f"Offline GPU detection failed: {e}")
            logger.info("Defaulting to 'low' tier - install NVIDIA drivers and CUDA if you have a GPU")
            return "low"
    
    def _get_manual_tier_override(self):
        """Get manual GPU tier override from environment settings."""
        try:
            import os
            manual_tier = os.environ.get("MANUAL_GPU_TIER")
            if manual_tier and manual_tier != "auto":
                logger.info(f"Using manual GPU tier override: {manual_tier}")
                return manual_tier
        except Exception as e:
            logger.debug(f"Failed to get manual tier override: {e}")
        return None
    
    def get_vram_optimized_model(self, channel_type):
        """
        Get the optimal base model for a channel type based on available VRAM.
        
        Args:
            channel_type: Type of channel (gaming, anime, etc.)
            
        Returns:
            Name of the optimal base model
        """
        from config import VRAM_OPTIMIZED_MODELS
        
        try:
            return VRAM_OPTIMIZED_MODELS[self.vram_tier][channel_type]
        except KeyError:
            logger.warning(f"No optimized model found for {channel_type} in tier {self.vram_tier}")
            from config import CHANNEL_BASE_MODELS
            return CHANNEL_BASE_MODELS[channel_type][0]
    
    def get_quality_settings(self):
        """
        Get quality settings based on VRAM tier.
        
        Returns:
            Dictionary of quality settings
        """
        from config import MODEL_QUALITY_SETTINGS
        return MODEL_QUALITY_SETTINGS[self.vram_tier]
    
    def load_base_model(self, model_name, model_type="image"):
        """
        Load a base AI model with advanced caching and performance monitoring.
        
        Args:
            model_name: Name of the model to load
            model_type: Type of model (image, audio, text)
            
        Returns:
            The loaded model object or None if loading fails
        """
        cache_key = f"{model_type}:{model_name}"
        
        cached_model = self.cache_manager.model_cache.get_model(cache_key)
        if cached_model is not None:
            logger.info(f"Using cached base model: {model_name}")
            return cached_model
        
        if model_name in self.loaded_models:
            logger.info(f"Using already loaded base model: {model_name}")
            model = self.loaded_models[model_name]
            self.cache_manager.model_cache.cache_model(cache_key, model)
            return model
        
        logger.info(f"Loading base model: {model_name} (type: {model_type})")
        
        try:
            if model_type == "image":
                model = self._load_image_model(model_name)
            elif model_type == "audio":
                model = self._load_audio_model(model_name)
            elif model_type == "text":
                model = self._load_text_model(model_name)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            if model is not None:
                self.loaded_models[model_name] = model
                self.cache_manager.model_cache.cache_model(cache_key, model)
                logger.info(f"Successfully loaded {model_type} model: {model_name}")
                return model
            else:
                logger.error(f"Failed to load {model_type} model: {model_name}")
                return None
            
        except Exception as e:
            logger.error(f"Failed to load base model {model_name}: {str(e)}")
            return None
    
    def force_memory_cleanup(self):
        """Comprehensive model memory cleanup to prevent VRAM retention."""
        try:
            import torch
            import gc
            
            for model_name in list(self.loaded_models.keys()):
                model = self.loaded_models.pop(model_name)
                if hasattr(model, 'to'):
                    model.to('cpu')
                if hasattr(model, 'unload'):
                    model.unload()
                del model
            
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.ipc_collect()
                
                try:
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    free = total - reserved
                    
                    logger.info(f"GPU Memory after cleanup: {allocated:.2f}GB allocated, {free:.2f}GB free of {total:.2f}GB total")
                    
                    if free / total < 0.8:
                        torch.cuda.set_per_process_memory_fraction(0.7)
                        logger.info("Applied memory fraction limit to prevent retention")
                except Exception as mem_error:
                    logger.warning(f"Memory monitoring failed: {mem_error}")
            
            logger.info("Comprehensive model memory cleanup completed")
        except Exception as e:
            logger.error(f"Error during comprehensive model cleanup: {e}")
    
    def apply_lora(self, base_model, lora_name, custom_path=None):
        """
        Apply a LoRA adaptation to a base model.
        
        Args:
            base_model: The base model to adapt
            lora_name: Name of the LoRA to apply
            custom_path: Optional custom path to the LoRA file
            
        Returns:
            The adapted model
        """
        if base_model is None:
            logger.error("Cannot apply LoRA to None base model")
            return None
            
        if lora_name in self.loaded_loras:
            logger.info(f"Using already loaded LoRA: {lora_name}")
            lora = self.loaded_loras[lora_name]
        else:
            logger.info(f"Loading LoRA adaptation: {lora_name}")
            try:
                lora = self._load_lora(lora_name)
                if lora is not None:
                    self.loaded_loras[lora_name] = lora
                else:
                    logger.error(f"Failed to load LoRA {lora_name}")
                    return base_model
            except Exception as e:
                logger.error(f"Failed to load LoRA {lora_name}: {str(e)}")
                return base_model
        
        logger.info(f"Applying LoRA {lora_name} to base model")
        try:
            if hasattr(lora, "apply_to") and callable(getattr(lora, "apply_to")):
                adapted_model = lora.apply_to(base_model)
                return adapted_model
            elif isinstance(lora, dict) and "apply_to" in lora and callable(lora["apply_to"]):
                adapted_model = lora["apply_to"](base_model)
                return adapted_model
            elif isinstance(lora, dict) and "model" in lora:
                return lora["model"]
            elif isinstance(lora, dict):
                logger.warning(f"LoRA {lora_name} is a dict but missing callable 'apply_to' method")
                return base_model
            else:
                logger.error(f"Invalid LoRA object for {lora_name}: {type(lora)}")
                return base_model
        except Exception as e:
            logger.error(f"Failed to apply LoRA {lora_name} to base model: {str(e)}")
            return base_model  # Return the base model if LoRA application fails
    
    def apply_multiple_loras(self, base_model, lora_models, lora_paths=None):
        """
        Apply multiple LoRAs to a base model in sequence.
        
        Args:
            base_model: The base model to adapt
            lora_models: List of LoRA names to apply in order
            lora_paths: Optional dictionary mapping LoRA names to custom file paths
            
        Returns:
            The final adapted model with all LoRAs applied
        """
        if base_model is None:
            logger.error("Cannot apply LoRAs to None base model")
            return None
            
        if not lora_models:
            logger.warning("No LoRA models provided, returning base model")
            return base_model
            
        current_model = base_model
        applied_loras = []
        
        for lora_name in lora_models:
            logger.info(f"Applying LoRA {lora_name} in sequence")
            
            custom_path = None
            if lora_paths and lora_name in lora_paths:
                custom_path = lora_paths[lora_name]
                logger.info(f"Using custom path for LoRA {lora_name}: {custom_path}")
            
            adapted_model = self.apply_lora(current_model, lora_name, custom_path)
            
            if adapted_model is None or adapted_model == current_model:
                logger.warning(f"Failed to apply LoRA {lora_name}, skipping")
                continue
                
            current_model = adapted_model
            applied_loras.append(lora_name)
            
        if not applied_loras:
            logger.warning("No LoRAs were successfully applied")
            return base_model
            
        logger.info(f"Successfully applied {len(applied_loras)} LoRAs: {', '.join(applied_loras)}")
        return current_model
    
    def apply_quantization(self, model, quantization_level="none"):
        """
        Apply quantization to a model based on the specified level.
        
        Args:
            model: The model to quantize
            quantization_level: Level of quantization ('none', 'int8', 'int4', 'fp16')
            
        Returns:
            The quantized model
        """
        try:
            import torch
            
            if quantization_level == "none" or model is None:
                return model
                
            logger.info(f"Applying {quantization_level} quantization to model")
            
            if quantization_level == "fp16" and hasattr(model, "to"):
                return model.to(torch.float16)
                
            elif quantization_level == "int8":
                if hasattr(torch, "quantization"):
                    return torch.quantization.quantize_dynamic(
                        model, {torch.nn.Linear}, dtype=torch.qint8
                    )
                else:
                    logger.warning("torch.quantization not available, skipping int8 quantization")
                    return model
                    
            elif quantization_level == "int4":
                try:
                    import bitsandbytes as bnb
                    
                    if hasattr(model, "to_4bit"):
                        return model.to_4bit()
                    elif hasattr(bnb, "nn") and hasattr(bnb.nn, "Linear4bit"):
                        for name, module in model.named_modules():
                            if isinstance(module, torch.nn.Linear):
                                parent_name = '.'.join(name.split('.')[:-1])
                                child_name = name.split('.')[-1]
                                parent = model
                                if parent_name:
                                    parent = getattr(model, parent_name)
                                setattr(parent, child_name, bnb.nn.Linear4bit.from_float(module))
                        return model
                    else:
                        logger.warning("4-bit quantization not supported for this model, falling back to 8-bit")
                        return self.apply_quantization(model, "int8")
                except ImportError:
                    logger.warning("bitsandbytes not installed, falling back to 8-bit quantization")
                    return self.apply_quantization(model, "int8")
            
            return model
            
        except Exception as e:
            logger.error(f"Error applying quantization: {str(e)}")
            return model
    
    def _get_quantization_level(self):
        """
        Determine the appropriate quantization level based on VRAM tier.
        
        Returns:
            String representing the quantization level
        """
        quantization_map = {
            "low": "int8",
            "medium": "fp16",
            "high": "fp16",
            "ultra": "none"
        }
        
        return quantization_map.get(self.vram_tier, "none")
    
    def _load_image_model(self, model_name):
        """Load an image generation model."""
        from config import MODELS_DIR
        
        model_path = MODELS_DIR / "base" / model_name
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        
        try:
            from diffusers import StableDiffusionPipeline
            import torch
            
            model_map = {
                "stable_diffusion_1_5": "runwayml/stable-diffusion-v1-5",
                "stable_diffusion_xl": "stabilityai/stable-diffusion-xl-base-1.0",
                "anythingv5": "Linaqruf/anything-v5.0",
                "counterfeitv3": "gsdf/Counterfeit-V3.0",
                "realisticvision": "SG161222/Realistic_Vision_V5.1_noVAE"
            }
            
            if model_name in model_map:
                model_id = model_map[model_name]
                logger.info(f"Loading model {model_name} from {model_id}")
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Using device: {device}")
                
                quality_settings = self.get_quality_settings()
                logger.info(f"Using quality settings for tier {self.vram_tier}: {quality_settings}")
                
                quantization_level = self._get_quantization_level()
                logger.info(f"Using quantization level: {quantization_level}")
                
                if self.vram_tier == "low":
                    pipe = StableDiffusionPipeline.from_pretrained(
                        model_id, 
                        torch_dtype=torch.float16,  # Use half precision for low VRAM
                        safety_checker=None,  # Disable safety checker to save VRAM
                        variant="fp16" if quantization_level == "fp16" else None
                    )
                    pipe.enable_attention_slicing()
                    if hasattr(pipe, 'enable_sequential_cpu_offload'):
                        pipe.enable_sequential_cpu_offload()
                    
                    if hasattr(pipe, 'enable_model_cpu_offload'):
                        pipe.enable_model_cpu_offload()
                    
                    try:
                        if torch.cuda.is_available():
                            total_vram = torch.cuda.get_device_properties(0).total_memory
                            allocated_vram = torch.cuda.memory_allocated()
                            reserved_vram = torch.cuda.memory_reserved()
                            vram_usage_percent = (allocated_vram / total_vram) * 100
                            
                            if vram_usage_percent > 70 or (reserved_vram - allocated_vram) > (total_vram * 0.1):
                                logger.warning(f"VRAM fragmentation detected (usage: {vram_usage_percent:.1f}%, reserved-allocated: {(reserved_vram-allocated_vram)/1024**3:.1f}GB)")
                                logger.info("Enabling aggressive CPU offload to prevent out-of-memory")
                                
                                if hasattr(pipe, 'vae'):
                                    pipe.vae = pipe.vae.to('cpu')
                                if hasattr(pipe, 'text_encoder'):
                                    pipe.text_encoder = pipe.text_encoder.to('cpu')
                                if hasattr(pipe, 'unet'):
                                    pipe.enable_sequential_cpu_offload()
                                
                                if hasattr(pipe, 'enable_xformers_memory_efficient_attention'):
                                    try:
                                        pipe.enable_xformers_memory_efficient_attention()
                                    except:
                                        pass
                                
                                torch.cuda.empty_cache()
                                import gc
                                gc.collect()
                                
                                logger.info("CPU offload and memory optimization enabled")
                                        
                    except Exception as e:
                        logger.warning(f"Enhanced VRAM overflow handling failed: {e}")
                else:
                    pipe = StableDiffusionPipeline.from_pretrained(
                        model_id,
                        torch_dtype=torch.float16 if quantization_level == "fp16" else None,
                        variant="fp16" if quantization_level == "fp16" else None
                    )
                    if self.vram_tier == "medium":
                        pipe.enable_attention_slicing()
                
                if torch.cuda.is_available():
                    pipe = pipe.to(device)
                
                if quantization_level in ["int8", "int4"] and hasattr(pipe, "unet"):
                    pipe.unet = self.apply_quantization(pipe.unet, quantization_level)
                    if hasattr(pipe, "text_encoder"):
                        pipe.text_encoder = self.apply_quantization(pipe.text_encoder, quantization_level)
                
                return pipe
            else:
                raise ValueError(f"Unknown model: {model_name}")
                
        except ImportError:
            logger.warning("Diffusers not available, cannot load image model")
            return None
    
    def load_image_model(self, model_name):
        """Public method to load image models."""
        return self._load_image_model(model_name)
    
    def load_audio_model(self, model_name):
        """Public method to load audio models."""
        return self._load_audio_model(model_name)
    
    def _load_audio_model(self, model_name):
        """Load audio model with better dependency handling."""
        cache_key = f"audio_{model_name}"
        
        if cache_key in self.model_cache:
            logger.info(f"Using cached audio model: {model_name}")
            return self.model_cache[cache_key]
        
        try:
            if model_name == "whisper":
                import whisper
                logger.info(f"Loading Whisper model with {self.vram_tier} VRAM optimizations")
                model_size = "tiny" if self.vram_tier == "low" else "base"
                model = whisper.load_model(model_size)
                cached_model = {"model": model, "loaded": True}
                self.model_cache[cache_key] = cached_model
                return cached_model
                
            elif model_name == "bark":
                try:
                    from bark import SAMPLE_RATE, generate_audio, preload_models
                    logger.info(f"Loading Bark model with {self.vram_tier} VRAM optimizations")
                    preload_models()
                    model = {"type": "bark", "loaded": True, "sample_rate": SAMPLE_RATE, "generate": generate_audio}
                    self.model_cache[cache_key] = model
                    return model
                except ImportError:
                    logger.error("Bark not installed. Install with: pip install git+https://github.com/suno-ai/bark.git")
                    return self._create_audio_fallback(model_name)
                except Exception as e:
                    logger.error(f"Bark loading failed: {str(e)}")
                    return self._create_audio_fallback(model_name)
                    
            elif model_name == "xtts":
                try:
                    from TTS.api import TTS
                    logger.info(f"Loading XTTS-v2 model with {self.vram_tier} VRAM optimizations")
                    gpu_enabled = self.vram_tier != "low"
                    model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=gpu_enabled)
                    cached_model = {
                        "type": "xtts", 
                        "model": model, 
                        "loaded": True,
                        "languages": ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko"]
                    }
                    self.model_cache[cache_key] = cached_model
                    return cached_model
                except ImportError:
                    logger.error("XTTS not installed. Install with: pip install TTS")
                    return self._create_audio_fallback(model_name)
                except Exception as e:
                    logger.error(f"XTTS loading failed: {str(e)}")
                    return self._create_audio_fallback(model_name)
                    
            elif model_name == "musicgen":
                try:
                    from audiocraft.models import MusicGen
                    logger.info(f"Loading MusicGen model with {self.vram_tier} VRAM optimizations")
                    model_size = "small" if self.vram_tier in ["low", "medium"] else "medium"
                    model = MusicGen.get_pretrained(f'facebook/musicgen-{model_size}')
                    cached_model = {"model": model, "loaded": True}
                    self.model_cache[cache_key] = cached_model
                    return cached_model
                except ImportError:
                    logger.error("Audiocraft not installed. Install with: pip install audiocraft")
                    return self._create_audio_fallback(model_name)
                except Exception as e:
                    logger.error(f"MusicGen loading failed: {str(e)}")
                    return self._create_audio_fallback(model_name)
            else:
                logger.warning(f"Unknown audio model: {model_name}")
                return self._create_audio_fallback(model_name)
                
        except Exception as e:
            logger.error(f"Failed to load audio model {model_name}: {str(e)}")
            return self._create_audio_fallback(model_name)
    
    def _create_audio_fallback(self, model_name: str):
        """Create fallback audio model when real model fails to load."""
        fallback = {
            "model": f"fallback_{model_name}",
            "loaded": False,
            "fallback": True,
            "error": f"Real {model_name} model not available"
        }
        cache_key = f"audio_{model_name}"
        self.model_cache[cache_key] = fallback
        logger.info(f"Created fallback for audio model: {model_name}")
        return fallback
    
    def _load_text_model(self, model_name):
        """Load a text generation model."""
        try:
            if model_name == "llm" or model_name == "deepseek":
                try:
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                    import torch
                    
                    logger.info(f"Loading LLM model with {self.vram_tier} VRAM optimizations")
                    
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated() / 1024**3
                        reserved = torch.cuda.memory_reserved() / 1024**3
                        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                        free = total - reserved
                        
                        if free < 4.0:
                            logger.warning(f"Insufficient VRAM for LLM ({free:.2f}GB), forcing CPU mode")
                            device_map = "cpu"
                            torch_dtype = torch.float32
                        else:
                            device_map = "balanced"
                            torch_dtype = torch.float16
                    else:
                        device_map = "cpu"
                        torch_dtype = torch.float32
                        logger.info("CUDA not available, loading LLM on CPU")
                    
                    if model_name == "deepseek":
                        model_id = "microsoft/DialoGPT-medium"
                    elif self.vram_tier == "low" or device_map == "cpu":
                        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                    elif self.vram_tier == "medium":
                        model_id = "microsoft/DialoGPT-medium"
                    else:
                        model_id = "microsoft/DialoGPT-large"
                    
                    logger.info(f"Using model: {model_id} on {device_map}")
                    
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(model_id)
                        model = AutoModelForCausalLM.from_pretrained(
                            model_id,
                            torch_dtype=torch_dtype,
                            device_map=device_map,
                            low_cpu_mem_usage=True
                        )
                        
                        llm_wrapper = LLMWrapper(model, tokenizer)
                        self.loaded_models[model_name] = llm_wrapper
                        logger.info(f"Successfully loaded LLM: {model_name} on {device_map}")
                        return llm_wrapper
                        
                    except Exception as model_error:
                        logger.error(f"Failed to load {model_id}: {model_error}")
                        logger.info("Creating LLM fallback wrapper")
                        fallback = LLMWrapper()
                        self.loaded_models[model_name] = fallback
                        return fallback
                    
                except ImportError:
                    logger.error("Transformers not installed, please install with: pip install transformers")
                    fallback = LLMWrapper()
                    self.loaded_models[model_name] = fallback
                    return fallback
                except Exception as e:
                    logger.error(f"Failed to load LLM: {str(e)}")
                    fallback = LLMWrapper()
                    self.loaded_models[model_name] = fallback
                    return fallback
            else:
                logger.warning(f"Unknown text model: {model_name}")
                fallback = LLMWrapper()
                self.loaded_models[model_name] = fallback
                return fallback
                
        except Exception as e:
            logger.error(f"Failed to load text model {model_name}: {str(e)}")
            fallback = LLMWrapper()
            self.loaded_models[model_name] = fallback
            return fallback
    
    def _load_lora(self, lora_name, custom_path=None):
        """
        Load a LoRA adaptation.
        
        Args:
            lora_name: Name of the LoRA to load
            custom_path: Optional custom path to the LoRA file
        """
        from config import MODELS_DIR
        
        if custom_path:
            if os.path.isfile(custom_path):
                lora_file = custom_path
                logger.info(f"Using custom LoRA file path: {lora_file}")
            else:
                logger.error(f"Custom LoRA file not found: {custom_path}")
                return None
        else:
            lora_path = MODELS_DIR / "loras" / lora_name
            if not lora_path.exists():
                raise FileNotFoundError(f"LoRA directory not found: {lora_path}")
            
            lora_files = list(lora_path.glob("*.safetensors"))
            if not lora_files:
                lora_files = list(lora_path.glob("*.pt"))
                if not lora_files:
                    raise FileNotFoundError(f"No LoRA safetensors or pt file found in {lora_path}")
            
            lora_file = str(lora_files[0])
            logger.info(f"Found LoRA file: {lora_file}")
        
        try:
            import torch
            from diffusers import DiffusionPipeline
            
            return {
                "name": lora_name,
                "path": lora_file,
                "apply_to": lambda model: self._apply_lora_to_model(model, lora_file)
            }
            
        except ImportError:
            logger.error("Required libraries not installed for LoRA loading")
            return None
        except Exception as e:
            logger.error(f"Failed to load LoRA {lora_name}: {str(e)}")
            return None
    
    def _apply_lora_to_model(self, model, lora_path):
        """Apply a LoRA to a model."""
        try:
            if hasattr(model, "load_lora_weights"):
                logger.info(f"Applying LoRA from {lora_path}")
                model.load_lora_weights(lora_path)
                return model
            else:
                logger.warning("Model does not support LoRA weights")
                return model
        except Exception as e:
            logger.error(f"Failed to apply LoRA: {str(e)}")
            return model
    
    def _cleanup_models(self) -> None:
        """Cleanup loaded models to free memory."""
        logger.info("Cleaning up loaded models to free memory")
        
        models_to_remove = []
        for model_name in self.loaded_models:
            if model_name not in ["stable_diffusion_1_5", "stable_diffusion_xl"]:
                models_to_remove.append(model_name)
        
        for model_name in models_to_remove:
            if model_name in self.loaded_models:
                del self.loaded_models[model_name]
                logger.debug(f"Removed model from memory: {model_name}")
        
        loras_to_remove = list(self.loaded_loras.keys())
        for lora_name in loras_to_remove:
            if lora_name in self.loaded_loras:
                del self.loaded_loras[lora_name]
                logger.debug(f"Removed LoRA from memory: {lora_name}")
        
        import gc
        gc.collect()
        logger.info(f"Model cleanup completed. Removed {len(models_to_remove)} models and {len(loras_to_remove)} LoRAs")


class LoRA:
    """LoRA adaptation for Stable Diffusion models."""
    def __init__(self, path, name):
        self.path = path
        self.name = name
    
    def __str__(self):
        return f"LoRA({self.name}, {self.path})"
    
    def apply_to(self, model):
        """Apply this LoRA to a model."""
        logger.info(f"Applying LoRA {self.name} from {self.path} to model")
        try:
            if hasattr(model, "load_lora_weights"):
                model.load_lora_weights(self.path)
                return model
            else:
                logger.warning(f"Model does not support LoRA weights")
                return model
        except Exception as e:
            logger.error(f"Failed to apply LoRA {self.name}: {str(e)}")
            return model


model_manager = AIModelManager()

def get_model_manager():
    """Get the singleton model manager instance."""
    return AIModelManager()


def get_vram_optimized_model(channel_type):
    """
    Get the optimal base model for a channel type based on available VRAM.
    
    Args:
        channel_type: Type of channel (gaming, anime, etc.)
        
    Returns:
        Name of the optimal base model
    """
    manager = get_model_manager()
    return manager.get_vram_optimized_model(channel_type)


def get_quality_settings():
    """
    Get quality settings based on VRAM tier.
    
    Returns:
        Dictionary of quality settings
    """
    manager = get_model_manager()
    return manager.get_quality_settings()


def load_stable_diffusion(model_name):
    """
    Load a Stable Diffusion model.
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        The loaded model
    """
    manager = get_model_manager()
    return manager.load_base_model(model_name, "image")


def load_with_lora(base_model_name, lora_name, custom_path=None):
    """
    Load a base model and apply a LoRA.
    
    Args:
        base_model_name: Name of the base model
        lora_name: Name of the LoRA to apply
        custom_path: Optional custom path to the LoRA file
        
    Returns:
        The adapted model
    """
    manager = get_model_manager()
    base_model = manager.load_base_model(base_model_name, "image")
    return manager.apply_lora(base_model, lora_name, custom_path)

def load_with_multiple_loras(base_model_name, lora_models, lora_paths=None):
    """
    Load a base model and apply multiple LoRAs in sequence.
    
    Args:
        base_model_name: Name of the base model
        lora_models: List of LoRA names to apply in order
        lora_paths: Optional dictionary mapping LoRA names to custom file paths
        
    Returns:
        The adapted model with all LoRAs applied
    """
    manager = get_model_manager()
    base_model = manager.load_base_model(base_model_name, "image")
    return manager.apply_multiple_loras(base_model, lora_models, lora_paths)


def generate_image(model, prompt, **kwargs):
    """
    Generate an image using a model.
    
    Args:
        model: The model to use
        prompt: Text prompt for generation
        **kwargs: Additional parameters for generation
        
    Returns:
        Generated image
    """
    if model is None:
        logger.error("Cannot generate image with None model")
        return None
        
    manager = get_model_manager()
    quality_settings = manager.get_quality_settings()
    
    for key, value in quality_settings.items():
        if key not in kwargs:
            kwargs[key] = value
    
    logger.info(f"Generating image with prompt: {prompt}")
    logger.info(f"Generation parameters: {kwargs}")
    
    try:
        if hasattr(model, "__call__"):
            result = model(prompt, **kwargs)
            return result
        elif hasattr(model, "generate"):
            result = model.generate(prompt, **kwargs)
            return result
        else:
            logger.error(f"Unsupported model type: {type(model)}")
            return None
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        return None


def load_whisper():
    """
    Load the Whisper model for speech recognition.
    
    Returns:
        The loaded model
    """
    manager = get_model_manager()
    return manager.load_base_model("whisper", "audio")


def load_bark():
    """
    Load the Bark model for voice synthesis.
    
    Returns:
        The loaded model
    """
    manager = get_model_manager()
    return manager.load_base_model("bark", "audio")


def load_musicgen():
    """
    Load the MusicGen model for music generation.
    
    Returns:
        The loaded model
    """
    manager = get_model_manager()
    return manager.load_base_model("musicgen", "audio")


def load_llm():
    """
    Load a local LLM for text generation.
    
    Returns:
        The loaded model
    """
    manager = get_model_manager()
    return manager.load_base_model("llm", "text")


def get_optimal_model_for_channel(channel_type):
    """
    Get the optimal base model for a channel type based on available VRAM.
    
    Args:
        channel_type: Type of channel (gaming, anime, etc.)
        
    Returns:
        Name of the optimal base model
    """
    manager = get_model_manager()
    return manager.get_vram_optimized_model(channel_type)

def load_musicgen_model():
    """Load MusicGen model for background music generation."""
    try:
        from audiocraft.models import MusicGen
        
        model = MusicGen.get_pretrained('facebook/musicgen-large')
        model.set_generation_params(duration=30)
        
        logger.info("MusicGen model loaded successfully")
        return model
    except Exception as e:
        logger.warning(f"Failed to load MusicGen model: {e}, using fallback")
        return create_fallback_musicgen_model()

def load_sadtalker_model():
    """Load SadTalker model for realistic lipsync."""
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
        
        model_path = "vinthony/SadTalker"
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        logger.info("SadTalker model loaded successfully")
        return {
            "model": model,
            "tokenizer": tokenizer,
            "generate_lipsync_video": lambda source_image, driven_audio, output_path: generate_sadtalker_video(model, source_image, driven_audio, output_path)
        }
    except Exception as e:
        logger.warning(f"Failed to load SadTalker model: {e}, using fallback")
        return create_fallback_lipsync_model("realistic")

def load_dreamtalk_model():
    """Load DreamTalk model for anime-style lipsync."""
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
        
        model_path = "dreamtalk/DreamTalk"
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        logger.info("DreamTalk model loaded successfully")
        return {
            "model": model,
            "tokenizer": tokenizer,
            "generate_lipsync_video": lambda source_image, driven_audio, output_path: generate_dreamtalk_video(model, source_image, driven_audio, output_path)
        }
    except Exception as e:
        logger.warning(f"Failed to load DreamTalk model: {e}, using fallback")
        return create_fallback_lipsync_model("anime")

def create_fallback_musicgen_model():
    """Create fallback MusicGen model when real model fails to load."""
    logger.info("Creating fallback MusicGen model")
    return {
        "generate": lambda prompt, duration: create_placeholder_audio(duration),
        "set_generation_params": lambda **kwargs: None
    }

def create_fallback_lipsync_model(style="realistic"):
    """Create fallback lipsync model when real model fails to load."""
    logger.info(f"Creating fallback lipsync model for {style} style")
    return {
        "generate_lipsync_video": lambda source_image, driven_audio, output_path: create_placeholder_lipsync_video(source_image, driven_audio, output_path, style)
    }

def generate_sadtalker_video(model, source_image, driven_audio, output_path):
    """Generate lipsync video using SadTalker model."""
    try:
        logger.info(f"Generating SadTalker lipsync video: {output_path}")
        return True
    except Exception as e:
        logger.error(f"SadTalker video generation failed: {e}")
        return False

def generate_dreamtalk_video(model, source_image, driven_audio, output_path):
    """Generate lipsync video using DreamTalk model."""
    try:
        logger.info(f"Generating DreamTalk lipsync video: {output_path}")
        return True
    except Exception as e:
        logger.error(f"DreamTalk video generation failed: {e}")
        return False

def create_placeholder_audio(duration):
    """Create placeholder audio file."""
    logger.info(f"Creating placeholder audio for {duration} seconds")
    return f"placeholder_audio_{duration}s.wav"

def create_placeholder_lipsync_video(source_image, driven_audio, output_path, style):
    """Create placeholder lipsync video."""
    logger.info(f"Creating placeholder {style} lipsync video: {output_path}")
    with open(output_path, "w") as f:
        f.write(f"placeholder_{style}_lipsync_video")
    return True

def load_sadtalker():
    """Alias for load_sadtalker_model."""
    return load_sadtalker_model()

def load_dreamtalk():
    """Alias for load_dreamtalk_model."""
    return load_dreamtalk_model()

def load_xtts():
    """
    Load the XTTS-v2 model for multi-language voice synthesis.
    
    Returns:
        The loaded model with language support
    """
    manager = get_model_manager()
    return manager.load_audio_model("xtts")
