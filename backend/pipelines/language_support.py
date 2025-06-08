"""Enhanced multi-language support for AI Project Manager pipelines."""

import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = {
    'en': {
        'name': 'English',
        'voice_code': 'en',
        'tts_model': 'tts_models/en/ljspeech/tacotron2-DDC',
        'xtts_supported': True,
        'bark_supported': True
    },
    'ja': {
        'name': 'Japanese',
        'voice_code': 'ja',
        'tts_model': 'tts_models/ja/kokoro/tacotron2-DDC',
        'xtts_supported': True,
        'bark_supported': True
    },
    'es': {
        'name': 'Spanish',
        'voice_code': 'es',
        'tts_model': 'tts_models/es/mai/tacotron2-DDC',
        'xtts_supported': True,
        'bark_supported': True
    },
    'zh': {
        'name': 'Chinese (Mandarin)',
        'voice_code': 'zh-cn',
        'tts_model': 'tts_models/zh-CN/baker/tacotron2-DDC',
        'xtts_supported': True,
        'bark_supported': False
    },
    'hi': {
        'name': 'Hindi',
        'voice_code': 'hi',
        'tts_model': 'tts_models/hi/male/tacotron2-DDC',
        'xtts_supported': True,
        'bark_supported': False
    },
    'ar': {
        'name': 'Arabic',
        'voice_code': 'ar',
        'tts_model': 'tts_models/ar/male/tacotron2-DDC',
        'xtts_supported': True,
        'bark_supported': False
    },
    'bn': {
        'name': 'Bengali',
        'voice_code': 'bn',
        'tts_model': 'tts_models/bn/male/tacotron2-DDC',
        'xtts_supported': True,
        'bark_supported': False
    },
    'pt': {
        'name': 'Portuguese',
        'voice_code': 'pt',
        'tts_model': 'tts_models/pt/male/tacotron2-DDC',
        'xtts_supported': True,
        'bark_supported': True
    },
    'ru': {
        'name': 'Russian',
        'voice_code': 'ru',
        'tts_model': 'tts_models/ru/male/tacotron2-DDC',
        'xtts_supported': True,
        'bark_supported': False
    },
    'fr': {
        'name': 'French',
        'voice_code': 'fr',
        'tts_model': 'tts_models/fr/male/tacotron2-DDC',
        'xtts_supported': True,
        'bark_supported': True
    },
    'de': {
        'name': 'German',
        'voice_code': 'de',
        'tts_model': 'tts_models/de/male/tacotron2-DDC',
        'xtts_supported': True,
        'bark_supported': True
    }
}

def get_language_config(language_code: str) -> Dict[str, Any]:
    """Get language configuration for the specified language code."""
    return SUPPORTED_LANGUAGES.get(language_code, SUPPORTED_LANGUAGES['en'])

def get_supported_languages() -> List[str]:
    """Get list of supported language codes."""
    return list(SUPPORTED_LANGUAGES.keys())

def get_language_name(language_code: str) -> str:
    """Get human-readable language name."""
    config = get_language_config(language_code)
    return config['name']

def get_voice_code(language_code: str) -> str:
    """Get voice code for TTS generation."""
    config = get_language_config(language_code)
    return config['voice_code']

def get_tts_model(language_code: str) -> str:
    """Get TTS model for the specified language."""
    config = get_language_config(language_code)
    return config['tts_model']

def is_xtts_supported(language_code: str) -> bool:
    """Check if XTTS supports this language."""
    config = get_language_config(language_code)
    return config.get('xtts_supported', False)

def is_bark_supported(language_code: str) -> bool:
    """Check if Bark supports this language."""
    config = get_language_config(language_code)
    return config.get('bark_supported', False)

def detect_script_language(script_text: str) -> str:
    """Detect language of script text (basic implementation)."""
    if any('\u3040' <= char <= '\u309F' or '\u30A0' <= char <= '\u30FF' or '\u4E00' <= char <= '\u9FAF' for char in script_text):
        if '\u3040' <= script_text[0] <= '\u309F' or '\u30A0' <= script_text[0] <= '\u30FF':
            return 'ja'  # Japanese
        else:
            return 'zh'  # Chinese
    elif any('\u0600' <= char <= '\u06FF' for char in script_text):
        return 'ar'  # Arabic
    elif any('\u0900' <= char <= '\u097F' for char in script_text):
        return 'hi'  # Hindi
    elif any('\u0980' <= char <= '\u09FF' for char in script_text):
        return 'bn'  # Bengali
    elif any('\u0400' <= char <= '\u04FF' for char in script_text):
        return 'ru'  # Russian
    else:
        return 'en'

def enhance_script_with_language(script_data: Dict[str, Any], language_code: str = None) -> Dict[str, Any]:
    """Enhance script data with language-specific information."""
    if not language_code:
        script_text = ""
        for scene in script_data.get('scenes', []):
            script_text += scene.get('description', '') + " "
        language_code = detect_script_language(script_text)
    
    script_data['language'] = language_code
    script_data['language_config'] = get_language_config(language_code)
    
    for character in script_data.get('characters', []):
        character['language'] = language_code
        character['voice_code'] = get_voice_code(language_code)
        character['tts_model'] = get_tts_model(language_code)
    
    logger.info(f"Enhanced script with language: {get_language_name(language_code)} ({language_code})")
    return script_data

def get_language_specific_prompts(language_code: str) -> Dict[str, str]:
    """Get language-specific prompts for content generation."""
    prompts = {
        'en': {
            'scene_expansion': "Expand this scene with more detail and dialogue:",
            'character_description': "Describe this character in detail:",
            'combat_description': "Describe this combat scene with action and choreography:"
        },
        'ja': {
            'scene_expansion': "このシーンをより詳細に、対話を含めて展開してください：",
            'character_description': "このキャラクターを詳しく説明してください：",
            'combat_description': "この戦闘シーンをアクションと振り付けで説明してください："
        },
        'es': {
            'scene_expansion': "Expande esta escena con más detalle y diálogo:",
            'character_description': "Describe este personaje en detalle:",
            'combat_description': "Describe esta escena de combate con acción y coreografía:"
        }
    }
    
    return prompts.get(language_code, prompts['en'])
