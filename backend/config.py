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
