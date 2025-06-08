import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.absolute()
ASSETS_DIR = BASE_DIR / "assets"

POSSIBLE_MODEL_PATHS = [
    Path("/media/leon/NieuwVolume/AI app/models"),  # User's Linux mount
    Path("G:/ai_project_manager_app/models"),       # User's Windows path
    BASE_DIR / "models"                             # Default fallback
]

MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

# API settings
API_HOST = "127.0.0.1"
API_PORT = 8000

__all__ = [
    'BASE_DIR', 'ASSETS_DIR', 'MODELS_DIR', 'OUTPUT_DIR', 'POSSIBLE_MODEL_PATHS',
    'API_HOST', 'API_PORT', 'DATABASE_URL',
    'GPU_ENABLED', 'GPU_DEVICES', 'VRAM_TIERS',
    'BASE_MODEL_VERSIONS', 'BASE_MODEL_VRAM', 'VRAM_OPTIMIZED_MODELS',
    'MODEL_QUALITY_SETTINGS', 'MODEL_VERSIONS', 'AUDIO_MODEL_VERSIONS',
    'VIDEO_MODEL_VERSIONS', 'TEXT_MODEL_VERSIONS', 'EDITING_MODEL_VERSIONS', 'ALL_MODEL_CATEGORIES',
    'CHANNEL_BASE_MODELS', 'CHANNEL_TYPES'
]

# Database settings
DATABASE_URL = f"sqlite:///{BASE_DIR}/database/app.db"

# Hardware settings
GPU_ENABLED = False
GPU_DEVICES = [0]  # List of GPU device IDs to use

VRAM_TIERS = {
    "low": 8,      # 8GB VRAM
    "medium": 16,  # 16GB VRAM
    "high": 24,    # 24GB VRAM
    "ultra": 48    # 48GB VRAM
}

# Model settings
BASE_MODEL_VERSIONS = {
    "stable_diffusion_1_5": "v1.5",
    "stable_diffusion_xl": "v1.0",
    "anythingv5": "v5.0",
    "counterfeitv3": "v3.0",
    "realisticvision": "v5.1",
    "svd_xt": "1.1",
    "deforum": "v0.7"
}

BASE_MODEL_VRAM = {
    "stable_diffusion_1_5": 4,
    "stable_diffusion_xl": 8,
    "anythingv5": 6,
    "counterfeitv3": 6,
    "realisticvision": 8,
    "svd_xt": 16,
    "deforum": 12
}

VRAM_OPTIMIZED_MODELS = {
    "low": {  # 8GB VRAM
        "gaming": "stable_diffusion_1_5",
        "anime": "anythingv5",
        "superhero": "stable_diffusion_1_5",
        "manga": "anythingv5",
        "marvel_dc": "stable_diffusion_1_5",
        "original_manga": "anythingv5"
    },
    "medium": {  # 16GB VRAM
        "gaming": "realisticvision",
        "anime": "anythingv5",
        "superhero": "stable_diffusion_xl",
        "manga": "counterfeitv3",
        "marvel_dc": "stable_diffusion_xl",
        "original_manga": "counterfeitv3"
    },
    "high": {  # 24GB VRAM
        "gaming": "realisticvision",
        "anime": "anythingv5",
        "superhero": "stable_diffusion_xl",
        "manga": "counterfeitv3",
        "marvel_dc": "stable_diffusion_xl",
        "original_manga": "counterfeitv3"
    },
    "ultra": {  # 48GB VRAM
        "gaming": "realisticvision",
        "anime": "anythingv5",
        "superhero": "stable_diffusion_xl",
        "manga": "counterfeitv3",
        "marvel_dc": "stable_diffusion_xl",
        "original_manga": "counterfeitv3"
    }
}

MODEL_QUALITY_SETTINGS = {
    "low": {
        "width": 512,
        "height": 512,
        "steps": 20,
        "batch_size": 1
    },
    "medium": {
        "width": 768,
        "height": 768,
        "steps": 30,
        "batch_size": 1
    },
    "high": {
        "width": 1024,
        "height": 1024,
        "steps": 40,
        "batch_size": 2
    },
    "ultra": {
        "width": 1536,
        "height": 1536,
        "steps": 50,
        "batch_size": 4
    }
}

MODEL_VERSIONS = {
    "anime_style_lora": "v1.0",
    "gaming_style_lora": "v1.0",
    "superhero_style_lora": "v1.0",
    "manga_style_lora": "v1.0",
    "marvel_dc_style_lora": "v1.0",
    "original_manga_style_lora": "v1.0"
}

AUDIO_MODEL_VERSIONS = {
    "whisper": "large-v3",
    "bark": "v0.0.5", 
    "musicgen": "medium",
    "rvc": "v2.0",
    "sadtalker": "v0.0.2",
    "dreamtalk": "v1.0"
}

VIDEO_MODEL_VERSIONS = {
    "animatediff": "v2.0"
}

TEXT_MODEL_VERSIONS = {
    "llm": "phi-2"
}

EDITING_MODEL_VERSIONS = {
    "real_esrgan": "v0.6.0",
    "gfpgan": "v1.3.8",
    "codeformer": "v0.1.0"
}

ALL_MODEL_CATEGORIES = {
    "base": BASE_MODEL_VERSIONS,
    "lora": MODEL_VERSIONS,
    "audio": AUDIO_MODEL_VERSIONS,
    "video": VIDEO_MODEL_VERSIONS,
    "text": TEXT_MODEL_VERSIONS,
    "editing": EDITING_MODEL_VERSIONS
}

CHANNEL_BASE_MODELS = {
    "gaming": ["stable_diffusion_1_5", "realisticvision"],
    "anime": ["anythingv5", "stable_diffusion_1_5"],
    "superhero": ["stable_diffusion_1_5", "stable_diffusion_xl"],
    "manga": ["anythingv5", "counterfeitv3"],
    "marvel_dc": ["stable_diffusion_1_5", "stable_diffusion_xl"],
    "original_manga": ["anythingv5", "counterfeitv3"]
}

# Channel settings
CHANNEL_TYPES = [
    "gaming",
    "anime",
    "superhero",
    "manga",
    "marvel_dc",
    "original_manga"
]

# Ensure directories exist
for directory in [ASSETS_DIR, MODELS_DIR, OUTPUT_DIR]:
    directory.mkdir(exist_ok=True)
