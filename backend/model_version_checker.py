"""
Model version checking functionality for the AI Project Manager.
"""
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from huggingface_hub import HfApi, hf_hub_download, login, HfFolder
from config import MODEL_VERSIONS, BASE_MODEL_VERSIONS, MODELS_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("model_version_checker")

HF_MODEL_REPOS = {
    "stable_diffusion_1_5": "runwayml/stable-diffusion-v1-5",
    "stable_diffusion_xl": "stabilityai/stable-diffusion-xl-base-1.0",
    "anythingv5": "Linaqruf/anything-v5.0",
    "counterfeitv3": "gsdf/Counterfeit-V3.0",
    "realisticvision": "SG161222/Realistic_Vision_V5.1_noVAE",
    "anime_style_lora": "Linaqruf/anime-style-lora",
    "gaming_style_lora": "wavymulder/collage-diffusion-lora",
    "superhero_style_lora": "wavymulder/superhero-diffusion-lora", 
    "manga_style_lora": "Linaqruf/manga-diffusion-lora",
    "marvel_dc_style_lora": "wavymulder/comic-diffusion-lora",
    "original_manga_style_lora": "Linaqruf/manga-anime-style-lora",
    "svd_xt": "stabilityai/stable-video-diffusion-img2vid-xt",
    "zeroscope_v2_xl": "cerspense/zeroscope_v2_XL",
    "animatediff_v2_sdxl": "guoyww/AnimateDiff",
    "animatediff_lightning": "ByteDance/AnimateDiff-Lightning",
    "modelscope_t2v": "damo-vilab/text-to-video-ms-1.7b",
    "ltx_video": "Lightricks/LTX-Video",
    "skyreels_v2": "Skywork/SkyReels-V2-T2V-14B-540P",
    "sadtalker": "vinthony/SadTalker",
    "dreamtalk": "johnowhitaker/dreamtalk",
    "deepseek_llama_8b_peft": "Kenazin/Deepseek-Llama-8B-peft-p-ver5",
    "deepseek_r1_distill": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "llm": "Kenazin/Deepseek-Llama-8B-peft-p-ver5",
    "kernelllm": "facebook/KernelLLM"
}

class ModelVersionChecker:
    """
    Checks for updates to AI models by comparing local versions with HuggingFace versions.
    """
    def __init__(self, hf_token: Optional[str] = None):
        self.hf_token = hf_token
        self.api = HfApi()
        self.last_check_time = 0
        self.check_interval = 3600  # Check once per hour at most
        
        if hf_token:
            login(token=hf_token)
            logger.info("Authenticated with HuggingFace using provided token")
        elif HfFolder.get_token():
            logger.info("Using existing HuggingFace authentication")
    
    def check_for_updates(self, force: bool = False) -> Dict[str, List[Dict]]:
        """
        Check for updates to all models.
        
        Args:
            force: Force check even if the check interval hasn't elapsed
            
        Returns:
            Dictionary with 'base_models' and 'loras' lists containing update information
        """
        current_time = time.time()
        if not force and (current_time - self.last_check_time) < self.check_interval:
            logger.info(f"Skipping update check - last check was {current_time - self.last_check_time:.0f} seconds ago")
            return {"base_models": [], "loras": []}
        
        self.last_check_time = current_time
        logger.info("Checking for model updates...")
        
        updates = {
            "base_models": [],
            "loras": []
        }
        
        for model_name, local_version in BASE_MODEL_VERSIONS.items():
            repo_id = HF_MODEL_REPOS.get(model_name)
            if not repo_id:
                continue
                
            try:
                latest_version, commit_info = self._get_latest_version(repo_id)
                if latest_version and latest_version != local_version:
                    updates["base_models"].append({
                        "name": model_name,
                        "current_version": local_version,
                        "latest_version": latest_version,
                        "update_date": commit_info.get("date", "Unknown"),
                        "update_message": commit_info.get("message", "No message")
                    })
            except Exception as e:
                logger.error(f"Error checking updates for base model {model_name}: {str(e)}")
        
        for model_name, local_version in MODEL_VERSIONS.items():
            repo_id = HF_MODEL_REPOS.get(model_name)
            if not repo_id:
                continue
                
            try:
                latest_version, commit_info = self._get_latest_version(repo_id)
                if latest_version and latest_version != local_version:
                    updates["loras"].append({
                        "name": model_name,
                        "current_version": local_version,
                        "latest_version": latest_version,
                        "update_date": commit_info.get("date", "Unknown"),
                        "update_message": commit_info.get("message", "No message")
                    })
            except Exception as e:
                logger.error(f"Error checking updates for LoRA model {model_name}: {str(e)}")
        
        logger.info(f"Found {len(updates['base_models'])} base model updates and {len(updates['loras'])} LoRA updates")
        return updates
    
    def _get_latest_version(self, repo_id: str) -> Tuple[Optional[str], Dict]:
        """
        Get the latest version of a model from HuggingFace.
        
        Args:
            repo_id: HuggingFace repository ID
            
        Returns:
            Tuple of (version string, commit info dict)
        """
        try:
            commits = self.api.list_repo_commits(repo_id)
            if not commits:
                return None, {}
            
            latest_commit = commits[0]
            
            version = "latest"
            
            commit_msg = latest_commit.title
            if "v" in commit_msg and any(c.isdigit() for c in commit_msg):
                import re
                version_match = re.search(r'v\d+(\.\d+)*', commit_msg)
                if version_match:
                    version = version_match.group(0)
            
            if version == "latest":
                version = f"commit-{latest_commit.commit_id[:7]}"
            
            commit_info = {
                "date": latest_commit.created_at,
                "message": latest_commit.title,
                "commit_id": latest_commit.commit_id
            }
            
            return version, commit_info
            
        except Exception as e:
            logger.error(f"Error getting latest version for {repo_id}: {str(e)}")
            return None, {}
    
    def is_model_downloaded(self, model_name: str, model_type: str = "base") -> bool:
        """
        Check if a model is downloaded locally.
        
        Args:
            model_name: Name of the model
            model_type: Type of model ('base' or 'lora')
            
        Returns:
            True if the model is downloaded, False otherwise
        """
        if model_type == "base":
            model_dir = MODELS_DIR / "base" / model_name
        else:
            model_dir = MODELS_DIR / "loras" / model_name
            
        return model_dir.exists()
    
    def get_model_local_version(self, model_name: str, model_type: str = "base") -> Optional[str]:
        """
        Get the local version of a model.
        
        Args:
            model_name: Name of the model
            model_type: Type of model ('base' or 'lora')
            
        Returns:
            Version string or None if not found
        """
        if model_type == "base":
            return BASE_MODEL_VERSIONS.get(model_name)
        else:
            return MODEL_VERSIONS.get(model_name)

_version_checker = None

def get_version_checker(hf_token: Optional[str] = None) -> ModelVersionChecker:
    """
    Get the singleton version checker instance.
    
    Args:
        hf_token: HuggingFace API token (optional)
        
    Returns:
        ModelVersionChecker instance
    """
    global _version_checker
    if _version_checker is None:
        _version_checker = ModelVersionChecker(hf_token)
    elif hf_token and _version_checker.hf_token != hf_token:
        _version_checker = ModelVersionChecker(hf_token)
    return _version_checker
