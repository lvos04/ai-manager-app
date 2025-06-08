"""
Automatic model version update system for AI Project Manager.
"""

import json
import requests
import time
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from packaging import version
from .logging_config import get_logger

try:
    from .config_manager import get_config_manager
except ImportError:
    def get_config_manager():
        class MockConfigManager:
            def get(self, key, default=None):
                return default
        return MockConfigManager()

logger = get_logger("model_version_updater")

class ModelVersionUpdater:
    """Manages automatic checking and updating of AI model versions."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelVersionUpdater, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self.config_manager = get_config_manager()
            self.cache_file = Path("cache/model_versions.json")
            self.cache_file.parent.mkdir(exist_ok=True)
            self.version_cache = self._load_version_cache()
            self.last_check_time = 0
            self.check_interval = 3600  # 1 hour
            self._initialized = True
    
    def _load_version_cache(self) -> Dict[str, Any]:
        """Load cached version information."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading version cache: {e}")
        return {}
    
    def _save_version_cache(self) -> None:
        """Save version information to cache."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.version_cache, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving version cache: {e}")
    
    def check_for_updates(self, force: bool = False) -> Dict[str, List[Dict]]:
        """
        Check for model updates across all categories.
        
        Args:
            force: Force check even if within cache interval
            
        Returns:
            Dictionary of available updates by category
        """
        current_time = time.time()
        if not force and (current_time - self.last_check_time) < self.check_interval:
            logger.debug("Skipping update check - within cache interval")
            return self._get_cached_updates()
        
        logger.info("Checking for model updates...")
        updates = {
            "base_models": [],
            "video_models": [],
            "voice_models": [],
            "music_models": [],
            "lipsync_models": [],
            "llm_models": []
        }
        
        models_config = self.config_manager.get('models', {})
        
        for category, models in models_config.items():
            if category in ['vram_tiers', 'aspect_ratio_enforcement']:
                continue
                
            category_key = f"{category}"
            if category_key not in updates:
                continue
                
            for model_name, model_config in models.items():
                if isinstance(model_config, dict) and 'repository' in model_config:
                    update_info = self._check_model_update(model_name, model_config)
                    if update_info:
                        updates[category_key].append(update_info)
        
        self.last_check_time = current_time
        self.version_cache['last_check'] = current_time
        self.version_cache['updates'] = updates
        self._save_version_cache()
        
        total_updates = sum(len(category_updates) for category_updates in updates.values())
        logger.info(f"Found {total_updates} available model updates")
        
        return updates
    
    def _check_model_update(self, model_name: str, model_config: Dict) -> Optional[Dict]:
        """Check for updates for a specific model."""
        try:
            repository = model_config.get('repository', '')
            if not repository or not repository.startswith(('huggingface.co/', 'hf.co/')):
                return None
            
            repo_id = repository.replace('huggingface.co/', '').replace('hf.co/', '')
            current_version = model_config.get('version', 'latest')
            
            latest_info = self._get_latest_version_info(repo_id)
            if not latest_info:
                return None
            
            latest_version = latest_info.get('version', 'latest')
            
            if self._is_newer_version(current_version, latest_version):
                return {
                    "model_name": model_name,
                    "repository": repository,
                    "current_version": current_version,
                    "latest_version": latest_version,
                    "update_available": True,
                    "breaking_changes": self._check_breaking_changes(current_version, latest_version),
                    "release_notes": latest_info.get('release_notes', ''),
                    "size_mb": latest_info.get('size_mb', 0)
                }
        
        except Exception as e:
            logger.error(f"Error checking updates for {model_name}: {e}")
        
        return None
    
    def _get_latest_version_info(self, repo_id: str) -> Optional[Dict]:
        """Get latest version information from HuggingFace API."""
        try:
            api_url = f"https://huggingface.co/api/models/{repo_id}"
            response = requests.get(api_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                tags = data.get('tags', [])
                latest_tag = None
                
                for tag in tags:
                    if tag.startswith('v') and self._is_valid_version(tag[1:]):
                        if latest_tag is None or self._is_newer_version(latest_tag, tag):
                            latest_tag = tag
                
                return {
                    "version": latest_tag or "latest",
                    "release_notes": data.get('description', ''),
                    "size_mb": self._estimate_model_size(data),
                    "last_modified": data.get('lastModified', '')
                }
        
        except Exception as e:
            logger.debug(f"Error fetching version info for {repo_id}: {e}")
        
        return None
    
    def _is_valid_version(self, version_str: str) -> bool:
        """Check if version string is valid semver."""
        try:
            version.parse(version_str)
            return True
        except:
            return False
    
    def _is_newer_version(self, current: str, latest: str) -> bool:
        """Compare version strings to determine if update is available."""
        if current == "latest" or latest == "latest":
            return False
        
        try:
            current_clean = current.lstrip('v')
            latest_clean = latest.lstrip('v')
            
            if not self._is_valid_version(current_clean) or not self._is_valid_version(latest_clean):
                return False
            
            return version.parse(latest_clean) > version.parse(current_clean)
        except:
            return False
    
    def _check_breaking_changes(self, current: str, latest: str) -> bool:
        """Check if update contains breaking changes based on semver."""
        try:
            current_clean = current.lstrip('v')
            latest_clean = latest.lstrip('v')
            
            if not self._is_valid_version(current_clean) or not self._is_valid_version(latest_clean):
                return False
            
            current_ver = version.parse(current_clean)
            latest_ver = version.parse(latest_clean)
            
            return latest_ver.major > current_ver.major
        except:
            return False
    
    def _estimate_model_size(self, model_data: Dict) -> int:
        """Estimate model size in MB from HuggingFace API data."""
        try:
            siblings = model_data.get('siblings', [])
            total_size = 0
            
            for file_info in siblings:
                if file_info.get('rfilename', '').endswith(('.bin', '.safetensors', '.ckpt')):
                    total_size += file_info.get('size', 0)
            
            return total_size // (1024 * 1024)  # Convert to MB
        except:
            return 0
    
    def _get_cached_updates(self) -> Dict[str, List[Dict]]:
        """Get cached update information."""
        return self.version_cache.get('updates', {
            "base_models": [],
            "video_models": [],
            "voice_models": [],
            "music_models": [],
            "lipsync_models": [],
            "llm_models": []
        })
    
    def apply_update(self, model_name: str, category: str) -> bool:
        """
        Apply an available update for a model.
        
        Args:
            model_name: Name of the model to update
            category: Category of the model
            
        Returns:
            True if update was successful
        """
        try:
            updates = self._get_cached_updates()
            category_updates = updates.get(category, [])
            
            model_update = None
            for update in category_updates:
                if update['model_name'] == model_name:
                    model_update = update
                    break
            
            if not model_update:
                logger.error(f"No update available for {model_name}")
                return False
            
            if model_update.get('breaking_changes', False):
                logger.warning(f"Update for {model_name} contains breaking changes")
            
            from ..model_manager import download_model
            from ..database import get_db
            
            db = next(get_db())
            try:
                success = download_model(
                    model_name=model_name,
                    db=db,
                    model_id=0,  # Will be resolved by download function
                    hf_token=None
                )
                
                if success:
                    self._update_model_config(model_name, category, model_update['latest_version'])
                    logger.info(f"Successfully updated {model_name} to {model_update['latest_version']}")
                    return True
                else:
                    logger.error(f"Failed to download update for {model_name}")
                    return False
                    
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error applying update for {model_name}: {e}")
            return False
    
    def _update_model_config(self, model_name: str, category: str, new_version: str) -> None:
        """Update model configuration with new version."""
        try:
            config_file = Path("config/models.yaml")
            if config_file.exists():
                import yaml
                
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                if category in config and model_name in config[category]:
                    config[category][model_name]['version'] = new_version
                    
                    with open(config_file, 'w') as f:
                        yaml.dump(config, f, default_flow_style=False)
                    
                    logger.info(f"Updated {model_name} version to {new_version} in config")
        
        except Exception as e:
            logger.error(f"Error updating model config: {e}")
    
    def get_update_summary(self) -> Dict[str, Any]:
        """Get summary of available updates."""
        updates = self._get_cached_updates()
        
        total_updates = sum(len(category_updates) for category_updates in updates.values())
        breaking_changes = sum(
            1 for category_updates in updates.values()
            for update in category_updates
            if update.get('breaking_changes', False)
        )
        
        return {
            "total_updates": total_updates,
            "breaking_changes": breaking_changes,
            "last_check": self.version_cache.get('last_check', 0),
            "categories": {
                category: len(category_updates)
                for category, category_updates in updates.items()
            }
        }

_version_updater = None

def get_model_version_updater() -> ModelVersionUpdater:
    """Get the singleton model version updater instance."""
    global _version_updater
    if _version_updater is None:
        _version_updater = ModelVersionUpdater()
    return _version_updater
