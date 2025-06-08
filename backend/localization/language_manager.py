import json
import os
from typing import Dict, List, Optional
from pathlib import Path

class LanguageManager:
    """Manages multi-language support for the AI Project Manager."""
    
    SUPPORTED_LANGUAGES = {
        'en': 'English',
        'nl': 'Nederlands',
        'de': 'Deutsch',
        'fr': 'Français',
        'es': 'Español',
        'zh': '中文',
        'ja': '日本語'
    }
    
    def __init__(self):
        self.current_language = 'en'
        self.translations = {}
        self.load_all_translations()
    
    def load_all_translations(self):
        """Load all translation files."""
        translations_dir = Path(__file__).parent / 'translations'
        
        for lang_code in self.SUPPORTED_LANGUAGES.keys():
            translation_file = translations_dir / f'{lang_code}.json'
            if translation_file.exists():
                with open(translation_file, 'r', encoding='utf-8') as f:
                    self.translations[lang_code] = json.load(f)
            else:
                self.translations[lang_code] = {}
    
    def set_language(self, language_code: str):
        """Set the current language."""
        if language_code in self.SUPPORTED_LANGUAGES:
            self.current_language = language_code
    
    def get_text(self, key: str, language: Optional[str] = None) -> str:
        """Get translated text for a key."""
        lang = language or self.current_language
        
        if lang in self.translations and key in self.translations[lang]:
            return self.translations[lang][key]
        
        if key in self.translations.get('en', {}):
            return self.translations['en'][key]
        
        return key
    
    def get_all_languages(self) -> Dict[str, str]:
        """Get all supported languages."""
        return self.SUPPORTED_LANGUAGES.copy()
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        return list(self.SUPPORTED_LANGUAGES.keys())
    
    def get_voice_language_code(self, language: str) -> str:
        """Get voice synthesis language code."""
        voice_codes = {
            'en': 'en-US',
            'nl': 'nl-NL',
            'de': 'de-DE',
            'fr': 'fr-FR',
            'es': 'es-ES',
            'zh': 'zh-CN',
            'ja': 'ja-JP'
        }
        return voice_codes.get(language, 'en-US')
    
    def get_llm_language_prompt(self, language: str) -> str:
        """Get LLM prompt for specific language."""
        prompts = {
            'en': "Generate content in English.",
            'nl': "Genereer inhoud in het Nederlands.",
            'de': "Generiere Inhalte auf Deutsch.",
            'fr': "Générez du contenu en français.",
            'es': "Genera contenido en español.",
            'zh': "用中文生成内容。",
            'ja': "日本語でコンテンツを生成してください。"
        }
        return prompts.get(language, prompts['en'])

language_manager = LanguageManager()
