import asyncio
import logging
from typing import Dict, List, Optional, Any
from .language_manager import language_manager

logger = logging.getLogger(__name__)

class ScriptTranslator:
    """Handles automatic script translation for multi-language content generation."""
    
    def __init__(self):
        self.supported_languages = language_manager.get_all_languages()
    
    async def translate_script_to_languages(self, script_data: Dict, target_languages: List[str]) -> Dict[str, Dict]:
        """
        Translate script to multiple target languages.
        Returns a dictionary with language codes as keys and translated scripts as values.
        """
        translated_scripts = {}
        
        for lang_code in target_languages:
            if lang_code not in self.supported_languages:
                logger.warning(f"Language {lang_code} not supported, skipping")
                continue
                
            try:
                translated_script = await self._translate_script_to_language(script_data, lang_code)
                translated_scripts[lang_code] = translated_script
                logger.info(f"Successfully translated script to {lang_code}")
            except Exception as e:
                logger.error(f"Failed to translate script to {lang_code}: {e}")
                translated_scripts[lang_code] = script_data.copy()
        
        return translated_scripts
    
    async def _translate_script_to_language(self, script_data: Dict, target_language: str) -> Dict:
        """Translate a single script to target language."""
        translated_script = script_data.copy()
        
        language_prompt = language_manager.get_llm_language_prompt(target_language)
        
        if 'scenes' in script_data:
            translated_scenes = []
            for scene in script_data['scenes']:
                translated_scene = await self._translate_scene(scene, target_language, language_prompt)
                translated_scenes.append(translated_scene)
            translated_script['scenes'] = translated_scenes
        
        if 'characters' in script_data:
            translated_characters = []
            for character in script_data['characters']:
                translated_character = await self._translate_character(character, target_language, language_prompt)
                translated_characters.append(translated_character)
            translated_script['characters'] = translated_characters
        
        if 'locations' in script_data:
            translated_locations = []
            for location in script_data['locations']:
                translated_location = await self._translate_location(location, target_language, language_prompt)
                translated_locations.append(translated_location)
            translated_script['locations'] = translated_locations
        
        if 'title' in script_data:
            translated_script['title'] = await self._translate_text(script_data['title'], target_language, language_prompt)
        
        return translated_script
    
    async def _translate_scene(self, scene: Any, target_language: str, language_prompt: str) -> Any:
        """Translate a scene to target language."""
        if isinstance(scene, str):
            return await self._translate_text(scene, target_language, language_prompt)
        elif isinstance(scene, dict):
            translated_scene = scene.copy()
            if 'description' in scene:
                translated_scene['description'] = await self._translate_text(scene['description'], target_language, language_prompt)
            if 'dialogue' in scene:
                translated_scene['dialogue'] = await self._translate_text(scene['dialogue'], target_language, language_prompt)
            return translated_scene
        else:
            return scene
    
    async def _translate_character(self, character: Any, target_language: str, language_prompt: str) -> Any:
        """Translate character information to target language."""
        if isinstance(character, str):
            return await self._translate_text(character, target_language, language_prompt)
        elif isinstance(character, dict):
            translated_character = character.copy()
            if 'name' in character:
                pass
            if 'description' in character:
                translated_character['description'] = await self._translate_text(character['description'], target_language, language_prompt)
            return translated_character
        else:
            return character
    
    async def _translate_location(self, location: Any, target_language: str, language_prompt: str) -> Any:
        """Translate location information to target language."""
        if isinstance(location, str):
            return await self._translate_text(location, target_language, language_prompt)
        elif isinstance(location, dict):
            translated_location = location.copy()
            if 'name' in location:
                translated_location['name'] = await self._translate_text(location['name'], target_language, language_prompt)
            if 'description' in location:
                translated_location['description'] = await self._translate_text(location['description'], target_language, language_prompt)
            return translated_location
        else:
            return location
    
    async def _translate_text(self, text: str, target_language: str, language_prompt: str) -> str:
        """
        Translate text to target language using LLM.
        """
        try:
            from ..pipelines.ai_models import load_llm
            from ..pipelines.pipeline_utils import generate_text_with_kernelllm
            
            llm_model = load_llm()
            if llm_model is None:
                logger.warning("No LLM model available for translation, using fallback")
                try:
                    from ..pipelines.pipeline_utils import translate_text_multilang
                    result = translate_text_multilang(text, [target_language])
                    return result.get(target_language, text)
                except Exception:
                    return text
            
            translation_prompt = f"""
            {language_prompt}
            
            Translate the following text while preserving its meaning, context, and emotional tone:
            "{text}"
            
            Provide only the translated text without any additional explanation or formatting.
            """
            
            try:
                translated_text = generate_text_with_kernelllm(translation_prompt, max_length=512)
                if translated_text and translated_text.strip() != text.strip():
                    logger.info(f"Successfully translated text to {target_language}")
                    return translated_text.strip()
                else:
                    logger.warning(f"LLM translation failed, using fallback for {target_language}")
                    from ..pipelines.pipeline_utils import translate_text_multilang
                    result = translate_text_multilang(text, [target_language])
                    return result.get(target_language, text)
            except Exception as e:
                logger.error(f"LLM translation error: {e}, using fallback")
                from ..pipelines.pipeline_utils import translate_text_multilang
                result = translate_text_multilang(text, [target_language])
                return result.get(target_language, text)
                
        except Exception as e:
            logger.error(f"Translation failed for text '{text[:50]}...': {e}")
            return text
    
    def _translate_text_with_llm(self, text: str, target_language: str) -> str:
        """Enhanced LLM-based translation with proper error handling."""
        try:
            from ..pipelines.ai_models import load_llm
            from ..pipelines.pipeline_utils import generate_text_with_kernelllm
            
            llm_prompt = f"""Translate the following text to {target_language}. 
            Maintain the emotional tone and meaning. Only return the translation:
            
            Text: {text}
            
            Translation:"""
            
            llm = load_llm()
            if llm:
                translated = generate_text_with_kernelllm(llm_prompt, max_length=200)
                if translated and translated.strip() != text.strip():
                    return translated.strip()
            
            from ..pipelines.pipeline_utils import translate_text_multilang
            translated = translate_text_multilang(text, "en", target_language)
            if translated and translated.strip() != text.strip():
                return translated.strip()
                
            return text
            
        except Exception as e:
            logger.error(f"Enhanced translation error for {target_language}: {e}")
            return text

script_translator = ScriptTranslator()
