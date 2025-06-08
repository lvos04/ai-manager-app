"""
Centralized configuration management for AI Project Manager.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from .logging_config import get_logger

logger = get_logger("config_manager")

class ConfigManager:
    """Singleton configuration manager for centralized settings."""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self.load_config()
    
    def load_config(self):
        """Load configuration from YAML files."""
        try:
            config_dir = Path("config")
            self._config = {}
            
            models_config = config_dir / "models.yaml"
            if models_config.exists():
                with open(models_config, 'r', encoding='utf-8') as f:
                    self._config['models'] = yaml.safe_load(f)
                logger.info("Loaded models configuration")
            
            if not self._config:
                self._config = self._get_default_config()
                logger.info("Using default configuration")
                
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self._config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'models': {
                'vram_tiers': {
                    'low': '4-8GB',
                    'medium': '8-16GB', 
                    'high': '16-24GB',
                    'ultra_high': '24GB+'
                },
                'aspect_ratio_enforcement': {
                    'enabled': True,
                    'target_ratio': '16:9',
                    'crop_method': 'center',
                    'pad_method': 'letterbox'
                }
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_vram_tier_models(self, tier: str) -> Dict[str, Any]:
        """Get models for specific VRAM tier."""
        models = self.get('models', {})
        tier_models = {}
        
        for category, model_dict in models.items():
            if category == 'vram_tiers' or category == 'aspect_ratio_enforcement':
                continue
                
            for model_name, model_config in model_dict.items():
                if model_config.get('vram_requirement') == tier:
                    tier_models[model_name] = model_config
        
        return tier_models
    
    def is_aspect_ratio_enforcement_enabled(self) -> bool:
        """Check if 16:9 aspect ratio enforcement is enabled."""
        return self.get('models.aspect_ratio_enforcement.enabled', True)
    
    def get_target_aspect_ratio(self) -> str:
        """Get target aspect ratio."""
        return self.get('models.aspect_ratio_enforcement.target_ratio', '16:9')

_config_manager = None

def get_config_manager() -> ConfigManager:
    """Get the singleton configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager
