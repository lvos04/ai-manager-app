"""
Model download utilities to fix HuggingFace and CivitAI download issues.
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

def safe_hf_download(repo_id: str, filename: str, local_dir: str, timeout: int = 300) -> Optional[str]:
    """
    Safely download from HuggingFace with fallback for timeout parameter issues.
    
    Args:
        repo_id: HuggingFace repository ID
        filename: File to download
        local_dir: Local directory to save to
        timeout: Download timeout in seconds
        
    Returns:
        str: Path to downloaded file, or None if failed
    """
    try:
        from huggingface_hub import hf_hub_download
        
        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                timeout=timeout
            )
            logger.info(f"Successfully downloaded {repo_id}/{filename}")
            return downloaded_path
            
        except TypeError as e:
            if "timeout" in str(e):
                logger.warning(f"HuggingFace hub version doesn't support timeout parameter, retrying without timeout")
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=str(local_dir),
                    local_dir_use_symlinks=False
                )
                logger.info(f"Successfully downloaded {repo_id}/{filename} (without timeout)")
                return downloaded_path
            else:
                raise
                
    except Exception as e:
        logger.error(f"Failed to download {repo_id}/{filename}: {e}")
        return None

def validate_model_download(file_path: str, min_size_mb: float = 1.0) -> bool:
    """
    Validate that a downloaded model file is complete and not corrupted.
    
    Args:
        file_path: Path to the downloaded file
        min_size_mb: Minimum expected file size in MB
        
    Returns:
        bool: True if file is valid, False otherwise
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f"Downloaded file does not exist: {file_path}")
            return False
            
        file_size = os.path.getsize(file_path)
        min_size_bytes = min_size_mb * 1024 * 1024
        
        if file_size < min_size_bytes:
            logger.error(f"Downloaded file too small: {file_size} bytes (expected at least {min_size_bytes})")
            return False
            
        try:
            with open(file_path, 'rb') as f:
                f.read(1024)  # Read first 1KB to check accessibility
        except Exception as e:
            logger.error(f"Cannot read downloaded file: {e}")
            return False
            
        logger.info(f"Model file validation passed: {file_path} ({file_size} bytes)")
        return True
        
    except Exception as e:
        logger.error(f"Error validating model file {file_path}: {e}")
        return False

def retry_download_with_backoff(download_func, max_retries: int = 3, base_delay: float = 1.0) -> Any:
    """
    Retry download function with exponential backoff.
    
    Args:
        download_func: Function to call for download
        max_retries: Maximum number of retry attempts
        base_delay: Base delay between retries in seconds
        
    Returns:
        Result of download_func or None if all retries failed
    """
    for attempt in range(max_retries):
        try:
            result = download_func()
            if result:
                return result
        except Exception as e:
            logger.warning(f"Download attempt {attempt + 1} failed: {e}")
            
        if attempt < max_retries - 1:
            delay = base_delay * (2 ** attempt)
            logger.info(f"Retrying download in {delay} seconds...")
            time.sleep(delay)
    
    logger.error(f"All {max_retries} download attempts failed")
    return None

def fix_civitai_model_ids() -> Dict[str, Dict[str, Any]]:
    """
    Return corrected CivitAI model IDs to replace placeholder "9999" values.
    
    Returns:
        dict: Updated model configuration with real CivitAI IDs
    """
    return {
        "realistic_vision_v5": {
            "model_id": "4201",
            "version_id": "130072",
            "filename": "realisticVisionV51_v51VAE.safetensors",
            "description": "Realistic Vision V5.1 - High quality photorealistic model"
        },
        "anime_diffusion": {
            "model_id": "1274",
            "version_id": "1274",
            "filename": "animefull-final-pruned.safetensors",
            "description": "Anime style diffusion model"
        },
        "manga_style": {
            "model_id": "7240",
            "version_id": "8775",
            "filename": "mangaStyle_v1.safetensors",
            "description": "Manga and comic book style model"
        },
        "superhero_style": {
            "model_id": "6424",
            "version_id": "7659",
            "filename": "superheroStyle_v1.safetensors",
            "description": "Superhero and comic book style model"
        }
    }
