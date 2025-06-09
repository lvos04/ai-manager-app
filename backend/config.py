import os

API_HOST = "localhost"
API_PORT = 8000

DATABASE_URL = "sqlite:///./ai_project_manager.db"

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
BASE_MODELS_DIR = os.path.join(MODELS_DIR, "base")
LORA_MODELS_DIR = os.path.join(MODELS_DIR, "lora")
VIDEO_MODELS_DIR = os.path.join(MODELS_DIR, "video")
AUDIO_MODELS_DIR = os.path.join(MODELS_DIR, "audio")
UPSCALING_MODELS_DIR = os.path.join(MODELS_DIR, "upscaling")

DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
OUTPUT_DIR = DEFAULT_OUTPUT_DIR

CUDA_ENABLED = True
VRAM_OPTIMIZATION = True

MAX_CONCURRENT_TASKS = 4
ENABLE_MEMORY_OPTIMIZATION = True
ENABLE_CACHING = True

LOG_LEVEL = "INFO"
LOG_FILE = "ai_project_manager.log"

SUPPORTED_LANGUAGES = ["en", "es", "fr", "de", "nl", "zh", "ja"]
DEFAULT_LANGUAGE = "en"

DEFAULT_FPS = 24
DEFAULT_RESOLUTION = "1920x1080"
ASPECT_RATIO = "16:9"

AUDIO_SAMPLE_RATE = 48000
AUDIO_CHANNELS = 2

VIDEO_BITRATE = "12000k"
VIDEO_CODEC = "libx264"
VIDEO_PRESET = "veryslow"
VIDEO_CRF = "15"

ENABLE_MAXIMUM_QUALITY = True
PROCESSING_TIME_PRIORITY = False

ENABLE_AI_TRANSLATION = True
TRANSLATION_MODEL = "llm"

SVD_INFERENCE_STEPS = 50
ZEROSCOPE_INFERENCE_STEPS = 100
ANIMATEDIFF_INFERENCE_STEPS = 50

BASE_MODEL_VERSIONS = {
    "anythingv5": "v1.0",
    "realisticvision": "v6.0", 
    "dreamshaper": "v8.0",
    "kenshi": "v1.0",
    "arcane_diffusion": "v3.0",
    "aam_xl_animemix": "v3.1",
    "abyssorangemix3": "v3.0",
    "meina_mix": "v11.0",
    "mistoon": "v2.0",
    "animagine_v3": "v3.0",
    "realesrgan_anime": "v0.3.0",
    "absolutereality": "v1.8.1",
    "deliberate": "v3.0"
}

MODEL_VERSIONS = {
    "animatediff_v2": "v2.0",
    "stable_video_diffusion": "v1.1",
    "zeroscope_v2": "v2.0",
    "bark": "v0.1.5",
    "musicgen": "v1.0"
}

CHANNEL_BASE_MODELS = {
    "anime": ["anythingv5", "dreamshaper", "kenshi", "arcane_diffusion", "aam_xl_animemix", "abyssorangemix3", "meina_mix", "mistoon", "animagine_v3"],
    "manga": ["anythingv5", "dreamshaper", "meina_mix", "mistoon"], 
    "gaming": ["realisticvision", "absolutereality", "deliberate"],
    "superhero": ["realisticvision", "deliberate", "absolutereality"],
    "marvel_dc": ["realisticvision", "deliberate", "absolutereality"],
    "original_manga": ["anythingv5", "dreamshaper", "meina_mix"]
}

AUDIO_MODEL_VERSIONS = {
    "bark": "v0.1.5",
    "xtts": "v2.0",
    "musicgen": "v1.0"
}

VIDEO_MODEL_VERSIONS = {
    "animatediff_v2": "v2.0",
    "stable_video_diffusion": "v1.1", 
    "zeroscope_v2": "v2.0"
}

TEXT_MODEL_VERSIONS = {
    "deepseek_llama": "v1.0",
    "llama2": "v7b",
    "mistral": "v0.1"
}

EDITING_MODEL_VERSIONS = {
    "realesrgan": "v0.3.0",
    "rife": "v4.6"
}

ALL_MODEL_CATEGORIES = ["base", "lora", "video", "audio", "upscaling", "text", "editing"]

POSSIBLE_MODEL_PATHS = [
    "/media/leon/NieuwVolume/AI app/models/",
    "./models/",
    "~/models/",
    "/opt/models/"
]
