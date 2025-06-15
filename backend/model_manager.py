import os
import time
import requests
from pathlib import Path
from sqlalchemy.orm import Session
from typing import Optional, List, Dict, Any
import logging

from config import (
    MODELS_DIR, MODEL_VERSIONS, BASE_MODEL_VERSIONS, CHANNEL_BASE_MODELS,
    AUDIO_MODEL_VERSIONS, VIDEO_MODEL_VERSIONS, TEXT_MODEL_VERSIONS, EDITING_MODEL_VERSIONS, 
    ALL_MODEL_CATEGORIES, POSSIBLE_MODEL_PATHS
)
from .database import DBModel
from .core.model_version_updater import get_model_version_updater



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_manager")

VIDEO_MODELS = {

    "zeroscope_v2_xl": {
        "name": "Zeroscope v2 XL",
        "description": "Optimized for 16:9 compositions - 1024×576, 24 frames, no watermarks",
        "type": "video",
        "size": "5.4GB",
        "downloaded": False,
        "path": None,
        "model_id": "cerspense/zeroscope_v2_XL",
        "resolution": "1024×576",
        "max_frames": 24,
        "vram_requirement": "medium"
    },
    "animatediff_v2_sdxl": {
        "name": "AnimateDiff v2 SDXL-Beta",
        "description": "High resolution with camera movements - 1024×1024, 16 frames",
        "type": "video",
        "size": "6.8GB",
        "downloaded": False,
        "path": None,
        "model_id": "guoyww/animatediff-motion-adapter-sdxl-beta",
        "resolution": "1024×1024",
        "max_frames": 16,
        "vram_requirement": "medium"
    },
    "animatediff_lightning": {
        "name": "AnimateDiff-Lightning",
        "description": "Fast video generation - 512×512, 16 frames, 10x faster",
        "type": "video",
        "size": "3.2GB",
        "downloaded": False,
        "path": None,
        "model_id": "ByteDance/AnimateDiff-Lightning",
        "resolution": "512×512",
        "max_frames": 16,
        "vram_requirement": "low"
    },
    "modelscope_t2v": {
        "name": "ModelScope T2V",
        "description": "Lightweight text-to-video - 256×256, 16 frames",
        "type": "video",
        "size": "2.1GB",
        "downloaded": False,
        "path": None,
        "model_id": "damo-vilab/text-to-video-ms-1.7b",
        "resolution": "256×256",
        "max_frames": 16,
        "vram_requirement": "low"
    },

    "ltx_video": {
        "name": "LTX-Video",
        "description": "Transformer-based real-time generation - 768×512, 120 frames",
        "type": "video",
        "size": "12.0GB",
        "downloaded": False,
        "path": None,
        "model_id": "Lightricks/LTX-Video",
        "resolution": "768×512",
        "max_frames": 120,
        "vram_requirement": "ultra"
    },
    "skyreels_v2": {
        "name": "SkyReels V2",
        "description": "Infinite length video generation - 540p, unlimited frames",
        "type": "video",
        "size": "15.0GB",
        "downloaded": False,
        "path": None,
        "model_id": "Skywork/SkyReels-V2-T2V-14B-540P",
        "resolution": "960×540",
        "max_frames": "Unlimited",
        "vram_requirement": "ultra"
    },
    "self_forcing": {
        "name": "Self-Forcing Video Generation",
        "description": "Real-time streaming video generation - 480P, ~16 FPS, autoregressive rollout",
        "type": "video",
        "size": "5.68GB",
        "downloaded": False,
        "path": None,
        "model_id": "gdhe17/Self-Forcing",
        "resolution": "480P (scalable)",
        "max_frames": "Unlimited (streaming)",
        "vram_requirement": "high"
    },

}

LLM_MODELS = {
    "deepseek_llama_8b_peft": {
        "name": "Deepseek Llama 8B PEFT",
        "description": "High-quality text generation - 8B parameters, PEFT optimized",
        "type": "llm",
        "size": "16.2GB",
        "downloaded": False,
        "path": None,
        "model_id": "Kenazin/Deepseek-Llama-8B-peft-p-ver5",
        "capabilities": ["text_generation", "conversation", "creative_writing"],
        "vram_requirement": "high"
    },
    "deepseek_r1_distill": {
        "name": "Deepseek R1 Distill",
        "description": "Reasoning-focused LLM - 8B parameters, distilled from R1",
        "type": "llm",
        "size": "14.8GB",
        "downloaded": False,
        "path": None,
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "capabilities": ["reasoning", "problem_solving", "analysis"],
        "vram_requirement": "high"
    },
    "phi_3_5_mini": {
        "name": "Phi 3.5 Mini",
        "description": "Efficient small language model - 3.8B parameters",
        "type": "llm",
        "size": "7.6GB",
        "downloaded": False,
        "path": None,
        "model_id": "microsoft/phi-3.5-mini-instruct",
        "capabilities": ["text_generation", "instruction_following", "efficient"],
        "vram_requirement": "medium"
    },
    "dialogpt_medium": {
        "name": "DialoGPT Medium",
        "description": "Conversational AI model - 345M parameters",
        "type": "llm",
        "size": "1.4GB",
        "downloaded": False,
        "path": None,
        "model_id": "microsoft/DialoGPT-medium",
        "capabilities": ["conversation", "dialogue", "chat"],
        "vram_requirement": "low"
    }
}

EDITING_MODELS = {
    "scene_detection_v2": {
        "name": "Scene Detection v2",
        "description": "AI-powered scene change detection for video editing",
        "type": "editing",
        "size": "2.1GB",
        "downloaded": False,
        "path": None,
        "model_id": "microsoft/DialoGPT-medium",
        "capabilities": ["scene_detection", "video_analysis", "temporal_segmentation"],
        "vram_requirement": "low"
    },
    "highlight_extraction_gaming": {
        "name": "Gaming Highlight Extractor",
        "description": "Specialized model for extracting gaming highlights",
        "type": "editing",
        "size": "3.4GB",
        "downloaded": False,
        "path": None,
        "model_id": "microsoft/DialoGPT-medium",
        "capabilities": ["highlight_detection", "gaming_analysis", "action_recognition"],
        "vram_requirement": "medium"
    },
    "auto_editor_pro": {
        "name": "Auto Editor Pro",
        "description": "Professional automatic video editing with AI",
        "type": "editing",
        "size": "5.2GB",
        "downloaded": False,
        "path": None,
        "model_id": "microsoft/DialoGPT-medium",
        "capabilities": ["auto_editing", "cut_detection", "pacing_optimization"],
        "vram_requirement": "medium"
    },
    "shorts_generator_ai": {
        "name": "AI Shorts Generator",
        "description": "Specialized model for creating viral short-form content",
        "type": "editing",
        "size": "4.1GB",
        "downloaded": False,
        "path": None,
        "model_id": "microsoft/DialoGPT-medium",
        "capabilities": ["shorts_creation", "viral_optimization", "vertical_video"],
        "vram_requirement": "medium"
    },
    "commentary_generator": {
        "name": "Commentary Generator",
        "description": "AI model for generating gaming commentary and narration",
        "type": "editing",
        "size": "2.8GB",
        "downloaded": False,
        "path": None,
        "model_id": "microsoft/DialoGPT-medium",
        "capabilities": ["commentary_generation", "narration", "gaming_analysis"],
        "vram_requirement": "low"
    },
    "upscaler_real_esrgan": {
        "name": "Real-ESRGAN Upscaler",
        "description": "High-quality video upscaling with Real-ESRGAN",
        "type": "editing",
        "size": "67MB",
        "downloaded": False,
        "path": None,
        "model_id": "ai-forever/Real-ESRGAN",
        "capabilities": ["4x_upscaling", "noise_reduction", "detail_enhancement"],
        "vram_requirement": "medium"
    }
}

TEXT_MODELS = {
    "deepseek_llama_8b_peft": {
        "name": "Deepseek Llama 8B PEFT v5",
        "description": "Advanced language model optimized for content generation and script writing",
        "type": "text",
        "size": "8.5GB",
        "downloaded": False,
        "path": None,
        "model_id": "Kenazin/Deepseek-Llama-8B-peft-p-ver5",
        "context_length": 32768,
        "optimal_vram": "16GB+",
        "vram_requirement": "high"
    },
    "deepseek_r1_distill": {
        "name": "Deepseek R1 Distill Llama 8B",
        "description": "Efficient reasoning model for medium VRAM systems with strong logic capabilities",
        "type": "text",
        "size": "4.2GB",
        "downloaded": False,
        "path": None,
        "model_id": "deepseek-ai/deepseek-r1-distill-llama-8b",
        "context_length": 16384,
        "optimal_vram": "8-12GB",
        "vram_requirement": "medium"
    },
    "dialogpt_medium": {
        "name": "DialoGPT Medium",
        "description": "Fallback conversational model for low VRAM systems",
        "type": "text",
        "size": "1.2GB",
        "downloaded": False,
        "path": None,
        "model_id": "microsoft/DialoGPT-medium",
        "context_length": 1024,
        "optimal_vram": "4-6GB",
        "vram_requirement": "low"
    }
}

EDITING_MODELS = {
    "scene_detection_v2": {
        "name": "Advanced Scene Detection v2",
        "description": "AI-powered scene boundary detection with motion analysis for gaming content",
        "type": "editing",
        "size": "450MB",
        "downloaded": False,
        "path": None,
        "model_id": "scene-detection-v2",
        "capabilities": ["motion_analysis", "scene_transitions", "cut_detection"],
        "vram_requirement": "low"
    },
    "highlight_extraction_gaming": {
        "name": "Gaming Highlight Extractor",
        "description": "Specialized AI for detecting exciting moments in gameplay footage",
        "type": "editing",
        "size": "680MB",
        "downloaded": False,
        "path": None,
        "model_id": "highlight-extractor-gaming-v2",
        "capabilities": ["action_detection", "kill_highlights", "epic_moments"],
        "vram_requirement": "low"
    },
    "auto_editor_pro": {
        "name": "Auto Editor Pro",
        "description": "Intelligent video editing with automatic cuts, transitions, and pacing",
        "type": "editing",
        "size": "520MB",
        "downloaded": False,
        "path": None,
        "model_id": "auto-editor-pro-v1",
        "capabilities": ["smart_cuts", "transition_detection", "pacing_optimization"],
        "vram_requirement": "low"
    },
    "shorts_generator_ai": {
        "name": "AI Shorts Generator",
        "description": "Advanced AI for creating viral short-form content with optimal timing",
        "type": "editing",
        "size": "750MB",
        "downloaded": False,
        "path": None,
        "model_id": "shorts-generator-ai-v2",
        "capabilities": ["viral_detection", "optimal_length", "engagement_optimization"],
        "vram_requirement": "low"
    },
    "commentary_generator": {
        "name": "AI Commentary Generator",
        "description": "Generates natural commentary and narration for gaming content",
        "type": "editing",
        "size": "920MB",
        "downloaded": False,
        "path": None,
        "model_id": "commentary-generator-v1",
        "capabilities": ["voice_synthesis", "contextual_commentary", "emotion_detection"],
        "vram_requirement": "medium"
    },
    "upscaler_real_esrgan": {
        "name": "Real-ESRGAN Video Upscaler",
        "description": "AI-powered video upscaling for enhanced resolution and quality",
        "type": "editing",
        "size": "1.2GB",
        "downloaded": False,
        "path": None,
        "model_id": "real-esrgan-video-v1",
        "capabilities": ["4x_upscaling", "noise_reduction", "detail_enhancement"],
        "vram_requirement": "medium"
    }
}

TEXT_MODELS = {
    "deepseek_llama_8b_peft": {
        "name": "Deepseek Llama 8B PEFT v5",
        "description": "Advanced language model optimized for content generation and script writing",
        "type": "text",
        "size": "8.5GB",
        "downloaded": False,
        "path": None,
        "model_id": "Kenazin/Deepseek-Llama-8B-peft-p-ver5",
        "context_length": 32768,
        "optimal_vram": "16GB+",
        "vram_requirement": "high"
    },
    "deepseek_r1_distill": {
        "name": "Deepseek R1 Distill Llama 8B",
        "description": "Efficient reasoning model for medium VRAM systems with strong logic capabilities",
        "type": "text",
        "size": "4.2GB",
        "downloaded": False,
        "path": None,
        "model_id": "deepseek-ai/deepseek-r1-distill-llama-8b",
        "context_length": 16384,
        "optimal_vram": "8-12GB",
        "vram_requirement": "medium"
    },
    "dialogpt_medium": {
        "name": "DialoGPT Medium",
        "description": "Fallback conversational model for low VRAM systems",
        "type": "text",
        "size": "1.2GB",
        "downloaded": False,
        "path": None,
        "model_id": "microsoft/DialoGPT-medium",
        "context_length": 1024,
        "optimal_vram": "4-6GB",
        "vram_requirement": "low"
    }
}

EDITING_MODELS = {
    "scene_detection_v2": {
        "name": "Advanced Scene Detection v2",
        "description": "AI-powered scene boundary detection with motion analysis for gaming content",
        "type": "editing",
        "size": "450MB",
        "downloaded": False,
        "path": None,
        "model_id": "scene-detection-v2",
        "capabilities": ["motion_analysis", "scene_transitions", "cut_detection"],
        "vram_requirement": "low"
    },
    "highlight_extraction_gaming": {
        "name": "Gaming Highlight Extractor",
        "description": "Specialized AI for detecting exciting moments in gameplay footage",
        "type": "editing",
        "size": "680MB",
        "downloaded": False,
        "path": None,
        "model_id": "highlight-extractor-gaming-v2",
        "capabilities": ["action_detection", "kill_highlights", "epic_moments"],
        "vram_requirement": "low"
    },
    "auto_editor_pro": {
        "name": "Auto Editor Pro",
        "description": "Intelligent video editing with automatic cuts, transitions, and pacing",
        "type": "editing",
        "size": "520MB",
        "downloaded": False,
        "path": None,
        "model_id": "auto-editor-pro-v1",
        "capabilities": ["smart_cuts", "transition_detection", "pacing_optimization"],
        "vram_requirement": "low"
    },
    "shorts_generator_ai": {
        "name": "AI Shorts Generator",
        "description": "Advanced AI for creating viral short-form content with optimal timing",
        "type": "editing",
        "size": "750MB",
        "downloaded": False,
        "path": None,
        "model_id": "shorts-generator-ai-v2",
        "capabilities": ["viral_detection", "optimal_length", "engagement_optimization"],
        "vram_requirement": "low"
    },
    "commentary_generator": {
        "name": "AI Commentary Generator",
        "description": "Generates natural commentary and narration for gaming content",
        "type": "editing",
        "size": "920MB",
        "downloaded": False,
        "path": None,
        "model_id": "commentary-generator-v1",
        "capabilities": ["voice_synthesis", "contextual_commentary", "emotion_detection"],
        "vram_requirement": "medium"
    },
    "upscaler_real_esrgan": {
        "name": "Real-ESRGAN Video Upscaler",
        "description": "AI-powered video upscaling for enhanced resolution and quality",
        "type": "editing",
        "size": "1.2GB",
        "downloaded": False,
        "path": None,
        "model_id": "real-esrgan-video-v1",
        "capabilities": ["4x_upscaling", "noise_reduction", "detail_enhancement"],
        "vram_requirement": "medium"
    }
}

BASE_MODELS = {
    "stable_diffusion_1_5": {
        "name": "Stable Diffusion 1.5",
        "description": "Base model for realistic image generation",
        "size": "4.2GB",
        "downloaded": False,
        "vram_required": "medium",
        "repo": "runwayml/stable-diffusion-v1-5",
        "version": "v1.5"
    },
    "stable_diffusion_xl": {
        "name": "Stable Diffusion XL",
        "description": "High-resolution base model",
        "size": "6.9GB", 
        "downloaded": False,
        "vram_required": "high",
        "repo": "stabilityai/stable-diffusion-xl-base-1.0",
        "version": "v1.0"
    },
    "anythingv5": {
        "name": "Anything V5",
        "description": "Anime-focused base model",
        "size": "5.8GB",
        "downloaded": False,
        "vram_required": "medium",
        "repo": "Linaqruf/anything-v3.0",
        "version": "v5.0"
    },
    "counterfeitv3": {
        "name": "Counterfeit V3",
        "description": "High-quality anime base model",
        "size": "5.6GB",
        "downloaded": False,
        "vram_required": "medium", 
        "repo": "gsdf/Counterfeit-V3.0",
        "version": "v3.0"
    },
    "realisticvision": {
        "name": "Realistic Vision",
        "description": "Photorealistic base model",
        "size": "7.2GB",
        "downloaded": False,
        "vram_required": "high",
        "repo": "SG161222/Realistic_Vision_V6.0_B1_noVAE",
        "version": "v5.1"
    },
    "svd_xt": {
        "name": "Stable Video Diffusion XT",
        "description": "Latest SVD from Stability AI - 1024×576, 25 frames, highest quality",
        "size": "7.2GB",
        "downloaded": False,
        "vram_required": "high",
        "repo": "stabilityai/stable-video-diffusion-img2vid-xt",
        "version": "1.1"
    },
    "deforum": {
        "name": "Deforum Stable Diffusion",
        "description": "Advanced interpolation control - Variable resolution and frames",
        "size": "4.5GB",
        "downloaded": False,
        "vram_required": "medium",
        "repo": "deforum/deforum-stable-diffusion",
        "version": "v0.7"
    },
    "dreamshaper": {
        "name": "DreamShaper",
        "description": "Best for: Versatile anime/realistic hybrid content. Style: Dreamy, painting-like aesthetic with soft colors and artistic flair, excellent for fantasy and character portraits.",
        "size": "5.5GB",
        "downloaded": False,
        "vram_required": "medium",
        "repo": "Lykon/DreamShaper",
        "version": "v8.0"
    },
    "kenshi": {
        "name": "Kenshi",
        "description": "Best for: Traditional anime characters with detailed features. Style: Semi-realistic anime aesthetic balanced between anime and realistic art, ideal for character-focused content.",
        "size": "5.2GB", 
        "downloaded": False,
        "vram_required": "medium",
        "repo": "Linaqruf/anything-v3.0",
        "version": "v1.0"
    },
    "arcane_diffusion": {
        "name": "Arcane Diffusion",
        "description": "Best for: Stylized animation and fantasy content. Style: League of Legends Arcane series art style with bold colors and distinctive character designs, perfect for animated storytelling.",
        "size": "4.8GB",
        "downloaded": False,
        "vram_required": "medium",
        "repo": "nitrosocke/Arcane-Diffusion",
        "version": "v3.0"
    },
    "aam_xl_animemix": {
        "name": "AAM XL AnimeMix",
        "description": "Best for: High-resolution anime content with exceptional detail. Style: Premium anime mixing model with crisp details and vibrant colors, ideal for professional-quality anime production.",
        "size": "6.2GB",
        "downloaded": False,
        "vram_required": "high",
        "repo": "Linaqruf/animagine-xl-3.0",
        "version": "v3.1"
    },
    "abyssorangemix3": {
        "name": "AbyssOrangeMix3",
        "description": "Best for: Vibrant anime content with rich color palettes. Style: Popular anime model known for saturated colors and dynamic character expressions, excellent for energetic and colorful scenes.",
        "size": "5.7GB",
        "downloaded": False,
        "vram_required": "medium",
        "repo": "WarriorMama777/OrangeMixs",
        "version": "v3.0"
    },
    "meina_mix": {
        "name": "Meina Series",
        "description": "Best for: Modern anime characters with detailed facial features. Style: Contemporary anime aesthetic with focus on character detail and expression, perfect for character-driven narratives.",
        "size": "5.4GB",
        "downloaded": False,
        "vram_required": "medium",
        "repo": "Meina/MeinaMix",
        "version": "v11"
    },
    "mistoon": {
        "name": "Mistoon",
        "description": "Best for: Classic anime and cartoon-style content. Style: Blend of anime and cartoon aesthetics with clean lines and traditional animation feel, ideal for nostalgic or retro anime projects.",
        "size": "5.1GB",
        "downloaded": False,
        "vram_required": "medium",
        "repo": "nitrosocke/Arcane-Diffusion",
        "version": "v2.0"
    },
    "animagine_v3": {
        "name": "AnimagineV3",
        "description": "Best for: Latest generation anime content with cutting-edge quality. Style: State-of-the-art anime generation with exceptional detail and consistency, perfect for high-quality anime production.",
        "size": "6.0GB",
        "downloaded": False,
        "vram_required": "medium",
        "repo": "Linaqruf/animagine-xl-3.0",
        "version": "v3.0"
    },
    "absolutereality": {
        "name": "Absolute Reality",
        "description": "Best for: Photorealistic content and realistic character generation. Style: Ultra-realistic model with exceptional detail for lifelike characters and environments, ideal for realistic storytelling.",
        "size": "5.8GB",
        "downloaded": False,
        "vram_required": "medium",
        "repo": "Lykon/AbsoluteReality",
        "version": "v1.8.1"
    },
    "deliberate": {
        "name": "Deliberate",
        "description": "Best for: Versatile realistic content with artistic flair. Style: Balanced realistic model with artistic enhancement, excellent for cinematic scenes and dramatic character portraits.",
        "size": "5.6GB",
        "downloaded": False,
        "vram_required": "medium",
        "repo": "XpucT/Deliberate",
        "version": "v3.0"
    }
}

BASE_MODEL_PROMPT_TEMPLATES = {
    "anythingv5": {
        "prefix": "masterpiece, best quality, ultra detailed, 8k resolution, cinematic lighting, smooth animation, professional anime style",
        "negative": "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name"
    },
    "anything_xl": {
        "prefix": "masterpiece, best quality, newest, anime style",
        "structure": "<|quality|>, <|year|>, <|characters|>, <|tags|>",
        "negative": "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name"
    },
    "counterfeitv3": {
        "prefix": "masterpiece, best quality, detailed anime art, vibrant colors, professional illustration",
        "negative": "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    },
    "absolutereality": {
        "prefix": "photorealistic, highly detailed, professional photography, 8k uhd, realistic lighting, sharp focus",
        "negative": "cartoon, anime, painting, drawing, illustration, low quality, blurry, out of focus"
    },
    "deliberate": {
        "prefix": "highly detailed, photorealistic, professional quality, cinematic lighting, sharp focus",
        "negative": "cartoon, anime, low quality, blurry, distorted, deformed"
    },
    "stable_diffusion_1_5": {
        "prefix": "high quality, detailed, professional, sharp focus",
        "negative": "low quality, blurry, distorted, deformed, bad anatomy"
    },
    "stable_diffusion_xl": {
        "prefix": "masterpiece, best quality, ultra detailed, 8k uhd, professional quality, sharp focus",
        "negative": "low quality, blurry, distorted, deformed, bad anatomy, worst quality"
    },
    "dreamshaper": {
        "prefix": "masterpiece, best quality, highly detailed, professional art, vibrant colors, sharp focus",
        "negative": "low quality, blurry, distorted, deformed, bad anatomy, worst quality, jpeg artifacts"
    },
    "kenshi": {
        "prefix": "masterpiece, best quality, ultra detailed, anime style, professional illustration, vibrant colors",
        "negative": "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    },
    "aam_xl_animemix": {
        "prefix": "masterpiece, best quality, newest, detailed anime art, professional illustration, 8k resolution",
        "structure": "<|quality|>, <|year|>, <|characters|>, <|tags|>",
        "negative": "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name"
    },
    "mistoon": {
        "prefix": "masterpiece, best quality, detailed anime art, arcane style, professional illustration, vibrant colors",
        "negative": "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    },
    "meina_mix": {
        "prefix": "masterpiece, best quality, ultra detailed, anime style, professional illustration, cinematic lighting",
        "negative": "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    },
    "orange_mix": {
        "prefix": "masterpiece, best quality, vibrant colors, dynamic lighting, anime style, professional illustration",
        "negative": "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    }
}

CIVITAI_LORA_MODELS = {
    "anything_xl": {
        "model_id": "9409",
        "version_id": "384264",
        "name": "万象熔炉 | Anything XL",
        "description": "Advanced SDXL anime model with structured prompt support. Best for: high-quality anime art with modern styles.",
        "channel_compatibility": ["anime", "manga", "original_manga"]
    },
    "RealisticVisionV5_LoRA": {"model_id": "4201", "version_id": "501240"},  # Realistic Vision V6.0 B1
    "EpicPhotographicLoRA": {"model_id": "29460", "version_id": "98028"},  # Epic Photographic Style
    "Cinematic-Lighting-LoRA": {"model_id": "98374", "version_id": "117829"},  # Cinematic Lighting Style
    "MoodyPortraitLoRA": {"model_id": "46846", "version_id": "167706"},  # Moody Portrait Style
    "ActionHeroLoRA": {"model_id": "89374", "version_id": "106729"},  # Action Hero Style
    "FPSStyleLoRA": {"model_id": "56671", "version_id": "192371"},  # FPS Game Style
    "UltraRealDetailLoRA": {"model_id": "20282", "version_id": "71161"},  # Ultra Real Detail
    "DramaticLightingLoRA": {"model_id": "84729", "version_id": "101394"},  # Dramatic Lighting Style
    "StylizedGameLoRA": {"model_id": "7716", "version_id": "9070"},  # Taiwan Doll Likeness
    "UrbanCombatLoRA": {"model_id": "72847", "version_id": "86394"},  # Urban Combat Style
    "ApocalypseSurvivalLoRA": {"model_id": "74829", "version_id": "89374"},  # Apocalypse Survival Style
    "RainSceneLoRA": {"model_id": "73294", "version_id": "87629"},  # Rain Scene Style
    "MilitaryGearLoRA": {"model_id": "95729", "version_id": "114847"},  # Military Gear Style
    "BattlefieldFocusLoRA": {"model_id": "67394", "version_id": "80147"},  # Battlefield Focus Style
    "RealFaceEnhancerLoRA": {"model_id": "13941", "version_id": "16677"},  # Real Face Enhancer
    
    "Flat2D-AnimeLoRA": {"model_id": "12415", "version_id": "14644"},  # Monable Style (Anime)
    "ShoujoStyleLoRA": {"model_id": "25636", "version_id": "30638"},  # Shoujo Anime Style
    "AnimeBackgroundLoRA": {"model_id": "15916", "version_id": "18771"},  # Anime Background Style
    "MoeExpressionLoRA": {"model_id": "23906", "version_id": "28608"},  # Moe Expression Style
    "KaelHeroLoRA": {"model_id": "19229", "version_id": "22840"},  # Kael Hero Style
    "StudioGhibliLoRA": {"model_id": "6526", "version_id": "7657"},  # Studio Ghibli Style
    "ShonenFightLoRA": {"model_id": "20063", "version_id": "23833"},  # Female Miqo'te - FFXIV
    "SadAnimeLoRA": {"model_id": "31284", "version_id": "37329"},  # Sad Anime Style
    "ClassicAnimeEyesLoRA": {"model_id": "27659", "version_id": "33012"},  # Classic Anime Eyes
    "SliceOfLifeLoRA": {"model_id": "18394", "version_id": "21847"},  # Slice of Life Style
    "HighSchoolUniformLoRA": {"model_id": "22847", "version_id": "27294"},  # High School Uniform
    "AnimeNightSceneLoRA": {"model_id": "29384", "version_id": "35029"},  # Anime Night Scene
    "HairDetailLoRA": {"model_id": "16729", "version_id": "19847"},  # Hair Detail Style
    "AnimeFireMagicLoRA": {"model_id": "97384", "version_id": "116729"},  # Anime Fire Magic Style
    "RetroAnimeStyleLoRA": {"model_id": "24758", "version_id": "29584"},  # Retro Anime Style
    
    "ComicBookLineArtLoRA": {"model_id": "7716", "version_id": "9070"},  # Taiwan Doll Likeness (Comic Style)
    "WesternComicLoRA": {"model_id": "33847", "version_id": "40293"},  # Western Comic Style
    "HeroicPoseLoRA": {"model_id": "38472", "version_id": "45829"},  # Heroic Pose Style
    "PanelTextureLoRA": {"model_id": "42859", "version_id": "51037"},  # Panel Texture Style
    "SpeechBubbleLoRA": {"model_id": "35729", "version_id": "42584"},  # Speech Bubble Style
    "ClassicMarvelStyleLoRA": {"model_id": "28394", "version_id": "33847"},  # Classic Marvel Style
    "VillainShadowLoRA": {"model_id": "44729", "version_id": "53184"},  # Villain Shadow Style
    "ColorComicLoRA": {"model_id": "47293", "version_id": "56384"},  # Color Comic Style
    "ExplosionEffectLoRA": {"model_id": "84729", "version_id": "101394"},  # Explosion Effect Style
    "InkedActionLoRA": {"model_id": "39485", "version_id": "47029"},  # Inked Action Style
    "ComicGlowLoRA": {"model_id": "52847", "version_id": "62948"},  # Comic Glow Style
    "MangaCrossoverLoRA": {"model_id": "11376", "version_id": "13466"},  # Mizore Manga Style
    "ComicSplashPageLoRA": {"model_id": "41829", "version_id": "49738"},  # Comic Splash Page Style
    "BoldOutlineLoRA": {"model_id": "36294", "version_id": "43185"},  # Bold Outline Style
    "DarkComicLoRA": {"model_id": "51847", "version_id": "61829"},  # Dark Comic Style
    
    "MangaLineArtLoRA": {"model_id": "48392", "version_id": "57483"},  # Manga Line Art Style
    "ClassicShonenLoRA": {"model_id": "20063", "version_id": "23833"},  # Female Miqo'te - FFXIV
    "InkSketchLoRA": {"model_id": "53847", "version_id": "64029"},  # Ink Sketch Style
    "MangaGrayToneLoRA": {"model_id": "58394", "version_id": "69472"},  # Manga Gray Tone Style
    "PanelDepthLoRA": {"model_id": "61847", "version_id": "73529"},  # Panel Depth Style
    "MangaScreenToneLoRA": {"model_id": "55729", "version_id": "66384"},  # Manga Screen Tone Style
    "ShojoSoftLoRA": {"model_id": "49283", "version_id": "58647"},  # Shoujo Soft Style
    "GrittyMangaLoRA": {"model_id": "64729", "version_id": "76384"},  # Gritty Manga Style
    "SimpleInkLoRA": {"model_id": "57293", "version_id": "68147"},  # Simple Ink Style
    "MangaSpeedEffectLoRA": {"model_id": "73847", "version_id": "88294"},  # Manga Speed Effect Style
    "MangaVillainLoRA": {"model_id": "69384", "version_id": "82947"},  # Manga Villain Style
    "ChibiAccentLoRA": {"model_id": "52847", "version_id": "62948"},  # Chibi Accent Style
    "FantasyMangaPanelLoRA": {"model_id": "81394", "version_id": "97284"},  # Fantasy Manga Panel Style
    "RealisticMangaBlendLoRA": {"model_id": "25995", "version_id": "31020"},  # Realistic Manga Blend
    "MysteryShadowLoRA": {"model_id": "71829", "version_id": "85394"},  # Mystery Shadow Style
    
    "CustomArmorLoRA": {"model_id": "94729", "version_id": "113847"},  # Custom Armor Style
    "MaskedHeroLoRA": {"model_id": "83947", "version_id": "100294"},  # Masked Hero Style
    "UrbanFightSceneLoRA": {"model_id": "77384", "version_id": "92147"},  # Urban Fight Scene Style
    "HeroineCapeLoRA": {"model_id": "88729", "version_id": "105847"},  # Heroine Cape Style
    "ComicColorLoRA": {"model_id": "7716", "version_id": "9070"},  # Taiwan Doll Likeness (Comic Style)
    "EnergyBlastLoRA": {"model_id": "91847", "version_id": "109573"},  # Energy Blast Style
    "SuperVillainLoRA": {"model_id": "82947", "version_id": "98573"},  # Super Villain Style
    "GlowingEyesLoRA": {"model_id": "96384", "version_id": "115729"},  # Glowing Eyes Style
    "SciFiSuitLoRA": {"model_id": "87394", "version_id": "104729"},  # Sci-Fi Suit Style
    "DarkKnightLoRA": {"model_id": "75384", "version_id": "89627"},  # Dark Knight Style
    "ElementControlLoRA": {"model_id": "68394", "version_id": "81729"},  # Element Control Style
    "SuperTeamPoseLoRA": {"model_id": "92384", "version_id": "110729"},  # Super Team Pose Style
    "RivalShowdownLoRA": {"model_id": "86947", "version_id": "103584"},  # Rival Showdown Style
    "SpaceHeroLoRA": {"model_id": "76847", "version_id": "91394"},  # Space Hero Style
    "HeroOriginFlashbackLoRA": {"model_id": "79847", "version_id": "95294"},  # Hero Origin Flashback Style
    
    "FantasyMangaLoRA": {"model_id": "93284", "version_id": "111729"},  # Fantasy Manga Style
    "MysticVillageLoRA": {"model_id": "91847", "version_id": "109384"},  # Mystic Village Style
    "ProtagonistFocusLoRA": {"model_id": "74293", "version_id": "88647"},  # Protagonist Focus Style
    "MangaPageLayoutLoRA": {"model_id": "11376", "version_id": "13466"},  # Mizore Manga Style
    "DarkFantasyLoRA": {"model_id": "86294", "version_id": "102847"},  # Dark Fantasy Style
    "CreatureDesignLoRA": {"model_id": "85729", "version_id": "102394"},  # Creature Design Style
    "SymbolMagicLoRA": {"model_id": "76294", "version_id": "91047"},  # Symbol Magic Style
    "PostApocMangaLoRA": {"model_id": "94738", "version_id": "112947"},  # Post Apocalyptic Manga Style
    "SpiritualAuraLoRA": {"model_id": "67394", "version_id": "80147"},  # Spiritual Aura Style
    "MythicalBeastsLoRA": {"model_id": "78294", "version_id": "93847"},  # Mythical Beasts Style
    "HistoricalJapanLoRA": {"model_id": "82394", "version_id": "98729"},  # Historical Japan Style
    "MonasterySceneLoRA": {"model_id": "78394", "version_id": "93284"},  # Monastery Scene Style
    "EmotionalFlashbackLoRA": {"model_id": "75847", "version_id": "90294"},  # Emotional Flashback Style
    "MagicDuelLoRA": {"model_id": "18776", "version_id": "22290"},  # Panel Gag
    "WorldMapLoRA": {"model_id": "89729", "version_id": "107384"},  # World Map Style
    
    "anime_style_lora": {"model_id": "88394", "version_id": "105729"},  # Anime Style LoRA
    "gaming_style_lora": {"model_id": "122359", "version_id": "135867"},  # Working Gaming Style LoRA
    "superhero_style_lora": {"model_id": "97384", "version_id": "116729"},  # Superhero Style LoRA
    "manga_style_lora": {"model_id": "84729", "version_id": "101394"},  # Manga Style LoRA
    "marvel_dc_style_lora": {"model_id": "73847", "version_id": "88294"},  # Marvel DC Style LoRA
    "original_manga_style_lora": {"model_id": "91847", "version_id": "109573"}  # Original Manga Style LoRA
}

HF_MODEL_REPOS = {
    # Base models
    "stable_diffusion_1_5": "runwayml/stable-diffusion-v1-5",
    "stable_diffusion_xl": "stabilityai/stable-diffusion-xl-base-1.0",
    "anythingv5": "Linaqruf/anything-v3.0",
    "counterfeitv3": "gsdf/Counterfeit-V3.0",
    "realisticvision": "SG161222/Realistic_Vision_V6.0_B1_noVAE",
    "dreamshaper": "Lykon/DreamShaper",
    "kenshi": "Linaqruf/anything-v3.0",
    "arcane_diffusion": "nitrosocke/Arcane-Diffusion",
    "aam_xl_animemix": "Linaqruf/animagine-xl-3.0",
    "abyssorangemix3": "WarriorMama777/OrangeMixs",
    "meina_mix": "Meina/MeinaMix",
    "mistoon": "nitrosocke/Arcane-Diffusion",
    # Video models
    "zeroscope_v2_xl": "cerspense/zeroscope_v2_XL",
    "animatediff_v2_sdxl": "guoyww/animatediff-motion-adapter-sdxl-beta",
    "animatediff_lightning": "ByteDance/AnimateDiff-Lightning",
    "modelscope_t2v": "damo-vilab/text-to-video-ms-1.7b",
    "ltx_video": "Lightricks/LTX-Video",
    "skyreels_v2": "Skywork/SkyReels-V2-T2V-14B-540P",
    "self_forcing": "gdhe17/Self-Forcing",
    "animagine_v3": "Linaqruf/animagine-xl-3.0",
    "absolutereality": "Lykon/AbsoluteReality",
    "deliberate": "XpucT/Deliberate",
    # Audio models
    "whisper": "openai/whisper-large-v3",
    "bark": "suno/bark",
    "musicgen": "facebook/musicgen-large",
    "rvc": "coqui/XTTS-v2",
    # Video models
    "svd_xt": "stabilityai/stable-video-diffusion-img2vid-xt",
    "zeroscope_v2_xl": "cerspense/zeroscope_v2_XL",
    "animatediff_v2_sdxl": "guoyww/animatediff-motion-adapter-v1-5-2",
    "animatediff_lightning": "ByteDance/AnimateDiff-Lightning",
    "modelscope_t2v": "damo-vilab/text-to-video-ms-1.7b",
    "ltx_video": "Lightricks/LTX-Video",
    "skyreels_v2": "Skywork/SkyReels-V2-T2V-14B-540P",
    # LLM models
    "deepseek_llama_8b_peft": "Kenazin/Deepseek-Llama-8B-peft-p-ver5",
    "deepseek_r1_distill": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "llm": "Kenazin/Deepseek-Llama-8B-peft-p-ver5",
    "kernelllm": "facebook/KernelLLM",
    "dialogpt_medium": "microsoft/DialoGPT-medium",
    # Other models
    "sadtalker": "vinthony/SadTalker",
    "dreamtalk": "vinthony/SadTalker",
    "scene_detection": "facebook/detr-resnet-50",
    "scene_detection_v2": "microsoft/DialoGPT-medium",
    "upscaler_real_esrgan": "ai-forever/Real-ESRGAN",
    "realesrgan": "ai-forever/Real-ESRGAN",
    "realesrgan_anime": "ai-forever/Real-ESRGAN",
    # Additional models
    "Deepseek Llama 8B PEFT v5": "Kenazin/Deepseek-Llama-8B-peft-p-ver5",
    "Deepseek R1": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "animatediff": "guoyww/animatediff",
    "animatediff_v2_sdxl": "guoyww/AnimateDiff",
    "auto_editor_pro": "openai/whisper-large-v3",
    "commentary_generator": "microsoft/DialoGPT-medium",
    "deepseek_llama": "Kenazin/Deepseek-Llama-8B-peft-p-ver5",
    "deepseek_llama_8b": "Kenazin/Deepseek-Llama-8B-peft-p-ver5",
    "deepseek_r1": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deforum": "stabilityai/stable-diffusion-2-1",
    "highlight_extraction_gaming": "openai/whisper-large-v3",
    "shorts_generator_ai": "stabilityai/stable-diffusion-2-1",
    "auto_editor": "openai/whisper-large-v3",
    "highlight_extraction": "openai/whisper-large-v3", 
    "shorts_generator": "stabilityai/stable-diffusion-2-1"
}

VOICE_MODELS = {
    "bark_small": {
        "name": "Bark Small",
        "description": "Lightweight voice generation model",
        "size": "1.2GB",
        "downloaded": False,
        "vram_required": "4GB",
        "repo": "suno/bark-small"
    },
    "bark_large": {
        "name": "Bark Large", 
        "description": "High-quality voice generation model",
        "size": "3.8GB",
        "downloaded": False,
        "vram_required": "8GB",
        "repo": "suno/bark",
        "version": "1.0"
    },
    "xtts_v2": {
        "name": "XTTS v2 Multi-Language",
        "description": "Advanced text-to-speech with voice cloning and 17 language support",
        "size": "2.1GB", 
        "downloaded": False,
        "vram_required": "6GB",
        "repo": "coqui/XTTS-v2",
        "version": "2.0",
        "languages": ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko"]
    }
}

TRANSLATION_MODELS = {
    "helsinki_opus_mt": {
        "name": "Helsinki OPUS-MT Translation",
        "description": "Multi-language translation models for 10 major languages",
        "size": "1.2GB",
        "downloaded": False,
        "vram_required": "4GB",
        "repo": "Helsinki-NLP/opus-mt-en-*",
        "version": "1.0",
        "languages": ["es", "fr", "de", "it", "pt", "ru", "ja", "zh-cn", "ar", "hi"]
    }
}

MUSIC_MODELS = {
    "musicgen_small": {
        "name": "MusicGen Small",
        "description": "Lightweight music generation model",
        "size": "1.5GB",
        "downloaded": False,
        "vram_required": "4GB",
        "repo": "facebook/musicgen-small"
    },
    "musicgen_medium": {
        "name": "MusicGen Medium",
        "description": "Balanced music generation model",
        "size": "3.3GB",
        "downloaded": False,
        "vram_required": "8GB", 
        "repo": "facebook/musicgen-medium"
    },
    "musicgen_large": {
        "name": "MusicGen Large",
        "description": "High-quality music generation model",
        "size": "6.7GB",
        "downloaded": False,
        "vram_required": "16GB",
        "repo": "facebook/musicgen-large"
    }
}

LIPSYNC_MODELS = {
    "sadtalker": {
        "name": "SadTalker",
        "description": "Realistic lipsync for human characters",
        "size": "2.8GB",
        "downloaded": False,
        "vram_required": "8GB",
        "repo": "vinthony/SadTalker"
    },
    "dreamtalk": {
        "name": "DreamTalk",
        "description": "Anime-style lipsync for animated characters",
        "size": "1.9GB",
        "downloaded": False,
        "vram_required": "6GB",
        "repo": "dreamtalk/dreamtalk-v1"
    }
}
TRANSLATION_MODELS = {
    "helsinki_opus_mt": {
        "name": "Helsinki OPUS-MT Translation",
        "description": "Multi-language translation models for 10 major languages",
        "size": "500MB",
        "downloaded": False,
        "vram_required": "2GB",
        "repo": "Helsinki-NLP/opus-mt-en-mul",
        "version": "1.0",
        "languages": ["es", "fr", "de", "it", "pt", "ru", "ja", "zh-cn", "ar", "hi"]
    }
}



def get_available_models():
    """
    Get all available models organized by category.
    
    Returns:
        List containing all available models
    """
    models = []
    
    # Add base models from BASE_MODELS
    for model_id, model_info in BASE_MODELS.items():
        model_dir = MODELS_DIR / "base" / model_id
        downloaded = check_model_downloaded(model_id, "base")
        
        models.append({
            "name": model_id,
            "display_name": model_info["name"],
            "description": model_info["description"],
            "size": model_info["size"],
            "downloaded": downloaded,
            "vram_required": model_info["vram_required"],
            "repo": model_info.get("repo", ""),
            "version": model_info.get("version", "1.0"),
            "model_type": "base",
            "category": "image",
            "channel_compatibility": [channel for channel, models in CHANNEL_BASE_MODELS.items() if model_id in models] or ["all"],
            "size_mb": int(float(model_info["size"].replace("GB", "").replace("MB", "")) * (1024 if "GB" in model_info["size"] else 1)),
            "download_path": str(model_dir) if downloaded else None
        })
    
    for model_name, model_info in VIDEO_MODELS.items():
        model_dir = MODELS_DIR / "video" / model_name
        downloaded = check_model_downloaded(model_name, "video")
        
        size_str = model_info.get("size", "1.0GB")
        try:
            size_mb = float(size_str.replace("GB", "").replace("MB", "")) * (1024 if "GB" in size_str else 1)
        except (ValueError, AttributeError):
            size_mb = 1024.0
        
        models.append({
            "name": model_name,
            "display_name": model_info["name"],
            "description": model_info["description"],
            "version": "1.0",
            "model_type": "video",
            "category": "video",
            "channel_compatibility": ["anime", "gaming", "superhero", "manga", "marvel_dc", "original_manga"],
            "size_mb": size_mb,
            "downloaded": downloaded,
            "download_path": str(model_dir) if downloaded else None,
            "model_id": model_info.get("model_id", ""),
            "vram_requirement": model_info.get("vram_requirement", "medium")
        })
    
    for model_name, model_info in TEXT_MODELS.items():
        model_dir = MODELS_DIR / "text" / model_name
        downloaded = check_model_downloaded(model_name, "text")
        
        size_str = model_info.get("size", "1.0GB")
        try:
            size_mb = float(size_str.replace("GB", "").replace("MB", "")) * (1024 if "GB" in size_str else 1)
        except (ValueError, AttributeError):
            size_mb = 1024.0  # Default 1GB
        
        models.append({
            "name": model_name,
            "display_name": model_info["name"],
            "description": model_info["description"],
            "version": "1.0",
            "model_type": "text",
            "category": "text",
            "channel_compatibility": ["anime", "gaming", "superhero", "manga", "marvel_dc", "original_manga"],
            "size_mb": size_mb,
            "downloaded": downloaded,
            "download_path": str(model_dir) if downloaded else None,
            "model_id": model_info.get("model_id", ""),
            "vram_requirement": model_info.get("vram_requirement", "medium")
        })
    
    editing_models = {
        "scene_detection": {
            "name": "Scene Detection AI",
            "description": "Automatic scene detection and segmentation",
            "type": "editing",
            "size": "2.1GB",
            "model_id": "microsoft/DialoGPT-medium",
            "vram_requirement": "low"
        },
        "highlight_extraction": {
            "name": "Highlight Extractor",
            "description": "AI for extracting gaming highlights",
            "type": "editing",
            "size": "3.4GB",
            "model_id": "openai/clip-vit-base-patch32",
            "vram_requirement": "medium"
        },
        "auto_editor": {
            "name": "Auto Video Editor",
            "description": "AI-powered automatic video editing",
            "type": "editing",
            "size": "4.8GB",
            "model_id": "facebook/detr-resnet-50",
            "vram_requirement": "high"
        },
        "shorts_generator": {
            "name": "Shorts Generator AI",
            "description": "AI for creating engaging short-form content",
            "type": "editing",
            "size": "2.9GB",
            "model_id": "google/flan-t5-base",
            "vram_requirement": "medium"
        },
        "realesrgan_anime": {
            "name": "RealESRGAN Anime Upscaler",
            "description": "AI-powered 4x upscaling optimized for anime content",
            "type": "editing",
            "size": "67MB",
            "model_id": "ai-forever/Real-ESRGAN",
            "vram_requirement": "medium"
        }
    }
    
    for model_name, model_info in editing_models.items():
        model_dir = MODELS_DIR / "editing" / model_name
        downloaded = check_model_downloaded(model_name, "editing")
        
        size_str = model_info.get("size", "1.0GB")
        try:
            size_mb = float(size_str.replace("GB", "").replace("MB", "")) * (1024 if "GB" in size_str else 1)
        except (ValueError, AttributeError):
            size_mb = 1024.0
        
        models.append({
            "name": model_name,
            "display_name": model_info["name"],
            "description": model_info["description"],
            "version": "1.0",
            "model_type": "editing",
            "category": "editing",
            "channel_compatibility": ["gaming", "anime", "superhero", "manga", "marvel_dc"],
            "size_mb": size_mb,
            "downloaded": downloaded,
            "download_path": str(model_dir) if downloaded else None,
            "model_id": model_info.get("model_id", ""),
            "vram_requirement": model_info.get("vram_requirement", "medium")
        })
    
    try:
        from lora_config import CHANNEL_LORAS, LORA_DESCRIPTIONS
    except ImportError:
        CHANNEL_LORAS = {}
        LORA_DESCRIPTIONS = {}
    
    for model_name, model_info in CIVITAI_LORA_MODELS.items():
        model_dir = MODELS_DIR / "loras" / model_name
        downloaded = check_model_downloaded(model_name, "loras")
        
        compatible_channels = []
        for channel, loras in CHANNEL_LORAS.items():
            if model_name in loras:
                compatible_channels.append(channel)
        
        if not compatible_channels:
            compatible_channels = ["all"]
        
        models.append({
            "name": model_name,
            "display_name": model_name.replace("_", " ").replace("LoRA", "").title(),
            "description": LORA_DESCRIPTIONS.get(model_name, f"LoRA adaptation: {model_name.replace('_', ' ').title()}"),
            "version": "v1.0",
            "model_type": "lora",
            "category": "image",
            "channel_compatibility": compatible_channels,
            "size_mb": 150.0 if downloaded else 0.0,
            "downloaded": downloaded,
            "download_path": str(model_dir) if downloaded else None,
            "model_id": model_info["model_id"],
            "version_id": model_info["version_id"]
        })
    
    for model_name, model_info in VOICE_MODELS.items():
        model_dir = MODELS_DIR / "voice" / model_name
        downloaded = check_model_downloaded(model_name, "voice")
        
        size_str = model_info.get("size", "1.0GB")
        try:
            size_mb = float(size_str.replace("GB", "").replace("MB", "")) * (1024 if "GB" in size_str else 1)
        except (ValueError, AttributeError):
            size_mb = 1024.0
        
        models.append({
            "name": model_name,
            "display_name": model_info["name"],
            "description": model_info["description"],
            "version": model_info.get("version", "1.0"),
            "model_type": "voice",
            "category": "audio",
            "channel_compatibility": ["anime", "gaming", "superhero", "manga", "marvel_dc", "original_manga"],
            "size_mb": size_mb,
            "downloaded": downloaded,
            "download_path": str(model_dir) if downloaded else None,
            "model_id": model_info.get("repo", ""),
            "vram_requirement": model_info.get("vram_required", "medium"),
            "languages": model_info.get("languages", ["en"])
        })
    
    for model_name, model_info in MUSIC_MODELS.items():
        model_dir = MODELS_DIR / "music" / model_name
        downloaded = check_model_downloaded(model_name, "music")
        
        size_str = model_info.get("size", "1.0GB")
        try:
            size_mb = float(size_str.replace("GB", "").replace("MB", "")) * (1024 if "GB" in size_str else 1)
        except (ValueError, AttributeError):
            size_mb = 1024.0
        
        models.append({
            "name": model_name,
            "display_name": model_info["name"],
            "description": model_info["description"],
            "version": "1.0",
            "model_type": "music",
            "category": "audio",
            "channel_compatibility": ["anime", "gaming", "superhero", "manga", "marvel_dc", "original_manga"],
            "size_mb": size_mb,
            "downloaded": downloaded,
            "download_path": str(model_dir) if downloaded else None,
            "model_id": model_info.get("repo", ""),
            "vram_requirement": model_info.get("vram_required", "medium")
        })
    
    for model_name, model_info in LIPSYNC_MODELS.items():
        model_dir = MODELS_DIR / "lipsync" / model_name
        downloaded = check_model_downloaded(model_name, "lipsync")
        
        size_str = model_info.get("size", "1.0GB")
        try:
            size_mb = float(size_str.replace("GB", "").replace("MB", "")) * (1024 if "GB" in size_str else 1)
        except (ValueError, AttributeError):
            size_mb = 1024.0
        
        models.append({
            "name": model_name,
            "display_name": model_info["name"],
            "description": model_info["description"],
            "version": "1.0",
            "model_type": "lipsync",
            "category": "audio",
            "channel_compatibility": ["anime", "manga", "marvel_dc", "original_manga"],
            "size_mb": size_mb,
            "downloaded": downloaded,
            "download_path": str(model_dir) if downloaded else None,
            "model_id": model_info.get("repo", ""),
            "vram_requirement": model_info.get("vram_required", "medium")
        })

    audio_models = {
        "whisper": {"version": "large-v3", "size": 1500.0, "desc": "Speech recognition and transcription model"},
        "rvc": {"version": "v2.0", "size": 600.0, "desc": "Voice conversion and cloning model"}
    }
    
    for model_name, info in audio_models.items():
        model_dir = MODELS_DIR / "audio" / model_name
        downloaded = check_model_downloaded(model_name, "audio")
        
        models.append({
            "name": model_name,
            "version": info["version"],
            "model_type": "audio",
            "category": "audio",
            "channel_compatibility": list(CHANNEL_BASE_MODELS.keys()),
            "size_mb": info["size"] if downloaded else 0.0,
            "downloaded": downloaded,
            "download_path": str(model_dir) if downloaded else None,
            "description": info["desc"]
        })
    
    video_models = {
        "animatediff": {"version": "v2.0", "size": 1500.0, "desc": "Image animation model"}
    }
    
    for model_name, info in video_models.items():
        model_dir = MODELS_DIR / "video" / model_name
        downloaded = check_model_downloaded(model_name, "video")
        
        models.append({
            "name": model_name,
            "version": info["version"],
            "model_type": "video",
            "category": "video",
            "channel_compatibility": list(CHANNEL_BASE_MODELS.keys()),  # Compatible with all channels
            "size_mb": info["size"] if downloaded else 0.0,
            "downloaded": downloaded,
            "download_path": str(model_dir) if downloaded else None,
            "description": info["desc"]
        })
    
    text_models = {
        "llm": {"version": "phi-2", "size": 2500.0, "desc": "Language model for text generation"}
    }
    
    for model_name, info in text_models.items():
        model_dir = MODELS_DIR / "text" / model_name
        downloaded = check_model_downloaded(model_name, "text")
        
        models.append({
            "name": model_name,
            "version": info["version"],
            "model_type": "text",
            "category": "text",
            "channel_compatibility": list(CHANNEL_BASE_MODELS.keys()),  # Compatible with all channels
            "size_mb": info["size"] if downloaded else 0.0,
            "downloaded": downloaded,
            "download_path": str(model_dir) if downloaded else None,
            "description": info["desc"]
        })
    
    return models


TEXT_MODELS = {
    "deepseek_llama_8b": {
        "name": "Deepseek Llama 8B PEFT v5",
        "description": "Advanced language model optimized for creative content generation",
        "model_id": "Kenazin/Deepseek-Llama-8B-peft-p-ver5",
        "size_mb": 16000,
        "vram_requirement": "high",
        "context_length": 4096,
        "recommended_use": "Creative writing, dialogue generation, content optimization"
    },
    "deepseek_r1": {
        "name": "Deepseek R1",
        "description": "Latest reasoning-focused language model",
        "model_id": "deepseek-ai/deepseek-r1-distill-llama-8b",
        "size_mb": 14000,
        "vram_requirement": "medium",
        "context_length": 8192,
        "recommended_use": "Complex reasoning, analysis, technical content"
    }
}

EDITING_MODELS = {
    "scene_detection": {
        "name": "Scene Detection Model",
        "description": "AI model for automatic scene boundary detection in videos",
        "model_id": "scene-detection/transnetv2",
        "size_mb": 800,
        "vram_requirement": "low",
        "supported_formats": ["mp4", "avi", "mov", "mkv"]
    },
    "highlight_extraction": {
        "name": "Highlight Extraction",
        "description": "Identifies exciting moments and highlights in gameplay footage",
        "model_id": "highlight-detection/gaming-v1",
        "size_mb": 1200,
        "vram_requirement": "medium",
        "supported_content": ["gaming", "sports", "action"]
    },
    "auto_editor": {
        "name": "Auto Video Editor",
        "description": "Automated video editing with cuts, transitions, and effects",
        "model_id": "auto-edit/video-editor-v2",
        "size_mb": 2000,
        "vram_requirement": "medium",
        "features": ["auto_cuts", "transitions", "color_correction"]
    },
    "shorts_generator": {
        "name": "Shorts Generator",
        "description": "AI model for creating engaging short-form vertical videos",
        "model_id": "shorts-ai/vertical-video-gen",
        "size_mb": 1500,
        "vram_requirement": "medium",
        "output_format": "vertical",
        "optimal_duration": "15-60s"
    }
}

def check_model_downloaded(model_id: str, model_type: str) -> bool:
    """
    Check if a model is downloaded.
    Priority: Database state > Custom settings > File system checks
    """
    import json
    from pathlib import Path as PathLib
    
    # Priority 1: Check database state first
    try:
        from .database import get_db, DBModel
        db = next(get_db())
        
        # Query database for this specific model
        db_model = db.query(DBModel).filter(
            DBModel.name == model_id,
            DBModel.model_type == model_type
        ).first()
        
        if db_model and db_model.downloaded:
            # Model is marked as downloaded in database
            db.close()
            return True
        
        db.close()
    except Exception as e:
        # Database check failed, continue with file system checks
        pass
    
    # Priority 2: Check custom settings file paths
    settings_file = PathLib.home() / ".ai_project_manager_settings.json"
    if settings_file.exists():
        try:
            with open(settings_file, 'r') as f:
                settings = json.load(f)
                model_paths = settings.get("model_paths", {})
                
                if model_type in model_paths and model_paths[model_type]:
                    custom_path = PathLib(model_paths[model_type])
                    if custom_path.exists():
                        if model_type == "base":
                            model_dir = custom_path / model_id
                        else:
                            model_dir = custom_path / model_id
                        
                        if model_dir.exists() and any(model_dir.iterdir()):
                            return True
        except Exception:
            pass
    
    # Priority 3: Check POSSIBLE_MODEL_PATHS
    from config import POSSIBLE_MODEL_PATHS
    
    for base_path in POSSIBLE_MODEL_PATHS:
        models_dir = base_path / model_type / model_id
        if models_dir.exists() and any(models_dir.iterdir()):
            return True
    
    # Priority 4: Check default MODELS_DIR
    models_dir = Path(MODELS_DIR) / model_type / model_id
    return models_dir.exists() and any(models_dir.iterdir())

def check_model_files_exist(model_dir: Path, expected_extensions: list[str] | None = None) -> bool:
    """Check if model files exist in directory with expected extensions."""
    if not model_dir.exists():
        return False
    
    if expected_extensions is None:
        expected_extensions = ['.safetensors', '.ckpt', '.bin', '.pt', '.pth']
    
    model_files = []
    for ext in expected_extensions:
        model_files.extend(model_dir.glob(f"*{ext}"))
        model_files.extend(model_dir.glob(f"**/*{ext}"))
    
    return len(model_files) > 0

def download_model(model_name: str, db: Session, model_id: int, hf_token: Optional[str] = None):
    """
    Download a model from HuggingFace Hub or Civitai.
    
    Args:
        model_name: Name of the model to download
        db: Database session
        model_id: ID of the model in the database
        hf_token: HuggingFace API token for authenticated downloads
    """
    # Get model from database
    db_model = db.query(DBModel).filter(DBModel.id == model_id).first()
    
    if db_model is None:
        return
    
    try:
        # Determine model type and directory
        model_type = db_model.model_type
        
        # Set up directories and file extensions based on model type
        if model_type == "base":
            model_dir = MODELS_DIR / "base" / model_name
            file_size = 2000.0
            version = BASE_MODEL_VERSIONS.get(model_name, "v1.0")
            file_extension = "safetensors"
        elif model_type == "lora":
            model_dir = MODELS_DIR / "loras" / model_name
            file_size = 150.0
            version = MODEL_VERSIONS.get(model_name, "v1.0")
            file_extension = "safetensors"
        elif model_type == "audio":
            model_dir = MODELS_DIR / "audio" / model_name
            file_size = 1000.0
            version = "latest"
            file_extension = "bin"
        elif model_type == "video":
            model_dir = MODELS_DIR / "video" / model_name
            file_size = 1000.0
            version = "latest"
            file_extension = "bin"
        elif model_type == "text":
            model_dir = MODELS_DIR / "text" / model_name
            file_size = 2500.0
            version = "latest"
            file_extension = "bin"
        elif model_type == "editing":
            model_dir = MODELS_DIR / "editing" / model_name
            file_size = 1500.0
            version = "latest"
            file_extension = "bin"
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create model directory
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Update progress in database - starting download
        db_model.size_mb = 0.1
        db.commit()
        
        if model_type == "lora" and model_name in CIVITAI_LORA_MODELS:
            # Get Civitai API key from database
            civitai_token = None
            try:
                from .database import DBSettings
                token_setting = db.query(DBSettings).filter(DBSettings.key == "civitai_token").first()
                if token_setting:
                    civitai_token = token_setting.value
                    logger.info("Using Civitai API token from database")
            except Exception as e:
                logger.warning(f"Error retrieving Civitai token: {str(e)}")
            
            return download_from_civitai(model_name, model_dir, db_model, db, civitai_token)
        
        return download_from_huggingface(model_name, model_type, model_dir, db_model, db, hf_token, version)
        
    except Exception as e:
        # Update model info on error
        db_model.downloaded = False
        db_model.download_path = None
        db.commit()
        
        logger.error(f"Model download failed: {str(e)}")
        return False

def download_from_civitai(model_name: str, model_dir: Path, db_model, db: Session, civitai_token: Optional[str] = None):
    """Download LoRA model from Civitai with retry logic and better error handling."""
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            if model_name not in CIVITAI_LORA_MODELS:
                logger.error(f"LoRA model {model_name} not found in Civitai mappings")
                return False
            
            model_info = CIVITAI_LORA_MODELS[model_name]
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            if civitai_token:
                headers['Authorization'] = f'Bearer {civitai_token}'
                logger.info(f"Using Civitai API token for authenticated download")
            
            download_url = None
            logger.info("Using API approach for CivitAI download (direct URLs return 403)...")
            
            version_api_url = f"https://civitai.com/api/v1/model-versions/{model_info['version_id']}"
            
            try:
                version_response = requests.get(version_api_url, headers=headers, timeout=10)
                if version_response.status_code == 200:
                    version_data = version_response.json()
                    
                    if "downloadUrl" in version_data:
                        download_url = version_data["downloadUrl"]
                        logger.info(f"Found API download URL: {download_url}")
                    elif "files" in version_data and len(version_data["files"]) > 0:
                        for file in version_data["files"]:
                            if "downloadUrl" in file:
                                download_url = file["downloadUrl"]
                                logger.info(f"Found file download URL: {download_url}")
                                break
                else:
                    logger.warning(f"Version API returned status {version_response.status_code}")
                    
                    model_api_url = f"https://civitai.com/api/v1/models/{model_info['model_id']}"
                    model_response = requests.get(model_api_url, headers=headers, timeout=10)
                    
                    if model_response.status_code == 200:
                        model_data = model_response.json()
                        
                        if "modelVersions" in model_data and len(model_data["modelVersions"]) > 0:
                            for version in model_data["modelVersions"]:
                                if str(version.get("id")) == str(model_info['version_id']):
                                    if "downloadUrl" in version:
                                        download_url = version["downloadUrl"]
                                        logger.info(f"Found model API download URL: {download_url}")
                                        break
                                    elif "files" in version and len(version["files"]) > 0:
                                        for file in version["files"]:
                                            if "downloadUrl" in file:
                                                download_url = file["downloadUrl"]
                                                logger.info(f"Found model file download URL: {download_url}")
                                                break
                        
            except Exception as e:
                logger.warning(f"Error with API approach: {str(e)}")
            
            if not download_url:
                raise Exception(f"No valid download URL found for model {model_name}")
            
            output_path = model_dir / f"{model_name}.safetensors"
            
            logger.info(f"Downloading {model_name} from: {download_url}")
            
            response = requests.get(download_url, stream=True, headers=headers, timeout=30, allow_redirects=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        # Update progress in database
                        if total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            db_model.size_mb = progress
                            db.commit()
            
            if not output_path.exists() or output_path.stat().st_size == 0:
                raise Exception("Downloaded file is empty or missing")
            
            # Update database with successful download
            db_model.downloaded = True
            db_model.download_path = str(model_dir)
            db_model.size_mb = os.path.getsize(output_path) / (1024 * 1024)
            db.commit()
            
            logger.info(f"LoRA model {model_name} downloaded successfully from Civitai")
            return True
            
        except Exception as e:
            logger.warning(f"Download attempt {attempt + 1} failed: {str(e)}")
            
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"All {max_retries} download attempts failed for {model_name}")
                
                error_file = model_dir / "download_error.txt"
                with open(error_file, "w") as f:
                    f.write(f"Download failed for {model_name} LoRA model after {max_retries} attempts: {str(e)}")
                
                # Update model info with error status
                db_model.downloaded = False
                db_model.download_path = str(model_dir)
                db.commit()
                
                return False
    
    return False

def download_from_huggingface(model_name: str, model_type: str, model_dir: Path, db_model, db: Session, hf_token: Optional[str], version: str, max_retries: int = 3, timeout: int = 120):
    """Download model from HuggingFace with retry logic and extended timeout."""
    import time
    
    for attempt in range(max_retries):
        try:
            from huggingface_hub import hf_hub_download, login, HfFolder, HfApi
            
            repo_id = HF_MODEL_REPOS.get(model_name)
            if not repo_id:
                raise Exception(f"No HuggingFace repository found for {model_name}")
            
            if hf_token:
                login(token=hf_token)
                logger.info(f"Authenticated with HuggingFace using provided token")
            elif HfFolder.get_token():
                logger.info(f"Using existing HuggingFace authentication")
            else:
                logger.warning(f"No HuggingFace token provided - attempting anonymous download")
            
            api = HfApi()
            files = api.list_repo_files(repo_id)
            
            extensions = ['.safetensors', '.ckpt', '.bin', '.pt', '.pth']
            model_files = [f for f in files if any(f.endswith(ext) for ext in extensions)]
            
            if not model_files:
                raise Exception(f"No model files found in repository {repo_id}")
            
            filename = model_files[0]
            
            logger.info(f"Downloading {model_name} from {repo_id}, file: {filename} (attempt {attempt + 1}/{max_retries})")
            
            try:
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=str(model_dir),
                    local_dir_use_symlinks=False,
                    timeout=timeout
                )
            except TypeError as e:
                if "timeout" in str(e):
                    logger.warning(f"HuggingFace hub version doesn't support timeout parameter, retrying without timeout")
                    downloaded_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        local_dir=str(model_dir),
                        local_dir_use_symlinks=False
                    )
                else:
                    raise
            
            db_model.downloaded = True
            db_model.download_path = str(model_dir)
            db_model.size_mb = os.path.getsize(downloaded_path) / (1024 * 1024)
            db_model.version = version
            db.commit()
            
            logger.info(f"{model_type.title()} model {model_name} downloaded successfully from HuggingFace")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading model from HuggingFace (attempt {attempt + 1}): {str(e)}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 15
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                error_file = model_dir / "download_error.txt"
                with open(error_file, "w") as f:
                    f.write(f"Download failed for {model_name} {model_type} model after {max_retries} attempts: {str(e)}")
                
                db_model.downloaded = False
                db_model.download_path = str(model_dir)
                db_model.version = version
                db.commit()
                
                logger.error(f"Failed to download {model_type} model {model_name} after {max_retries} attempts - see error log for details")
                return False
