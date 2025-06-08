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
            llm_model = self._load_llm()
            if llm_model is None:
                logger.warning("No LLM model available for translation, using fallback")
                return self._translate_text_fallback(text, target_language)
            
            translation_prompt = f"""
            {language_prompt}
            
            Translate the following text while preserving its meaning, context, and emotional tone:
            "{text}"
            
            Provide only the translated text without any additional explanation or formatting.
            """
            
            try:
                translated_text = self._generate_text_with_llm(llm_model, translation_prompt, max_length=512)
                if translated_text and translated_text.strip() != text.strip():
                    logger.info(f"Successfully translated text to {target_language}")
                    return translated_text.strip()
                else:
                    logger.warning(f"LLM translation failed, using fallback for {target_language}")
                    return self._translate_text_fallback(text, target_language)
            except Exception as e:
                logger.error(f"LLM translation error: {e}, using fallback")
                return self._translate_text_fallback(text, target_language)
                
        except Exception as e:
            logger.error(f"Translation failed for text '{text[:50]}...': {e}")
            return text
    
    def _load_llm(self):
        """Load LLM model for translation."""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            model_name = "microsoft/DialoGPT-medium"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            if torch.cuda.is_available():
                model = model.to("cuda")
            
            return {
                "model": model,
                "tokenizer": tokenizer,
                "device": "cuda" if torch.cuda.is_available() else "cpu"
            }
        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            return None
    
    def _generate_text_with_llm(self, llm_model: dict, prompt: str, max_length: int = 512) -> str:
        """Generate text using loaded LLM model."""
        try:
            import torch
            
            inputs = llm_model["tokenizer"](prompt, return_tensors="pt", truncation=True, max_length=max_length)
            if torch.cuda.is_available():
                inputs = {k: v.to(llm_model["device"]) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = llm_model["model"].generate(
                    **inputs,
                    max_length=inputs["input_ids"].shape[1] + 100,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=llm_model["tokenizer"].eos_token_id
                )
            
            generated_text = llm_model["tokenizer"].decode(outputs[0], skip_special_tokens=True)
            return generated_text[len(prompt):].strip()
            
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            return ""
    
    def _translate_text_fallback(self, text: str, target_language: str) -> str:
        """Fallback translation using simple replacements."""
        language_mappings = {
            "es": {"Hello": "Hola", "Goodbye": "Adiós", "Thank you": "Gracias"},
            "fr": {"Hello": "Bonjour", "Goodbye": "Au revoir", "Thank you": "Merci"},
            "de": {"Hello": "Hallo", "Goodbye": "Auf Wiedersehen", "Thank you": "Danke"},
            "it": {"Hello": "Ciao", "Goodbye": "Arrivederci", "Thank you": "Grazie"},
            "pt": {"Hello": "Olá", "Goodbye": "Tchau", "Thank you": "Obrigado"},
            "ru": {"Hello": "Привет", "Goodbye": "До свидания", "Thank you": "Спасибо"},
            "ja": {"Hello": "こんにちは", "Goodbye": "さようなら", "Thank you": "ありがとう"},
            "ko": {"Hello": "안녕하세요", "Goodbye": "안녕히 가세요", "Thank you": "감사합니다"},
            "zh": {"Hello": "你好", "Goodbye": "再见", "Thank you": "谢谢"},
            "nl": {"Hello": "Hallo", "Goodbye": "Tot ziens", "Thank you": "Dank je"}
        }
        
        if target_language in language_mappings:
            translated = text
            for english, translation in language_mappings[target_language].items():
                translated = translated.replace(english, translation)
            return translated
        
        return text
    
    def _translate_text_with_llm(self, text: str, target_language: str) -> str:
        """Enhanced LLM-based translation with proper error handling."""
        try:
            llm_prompt = f"""Translate the following text to {target_language}. 
            Maintain the emotional tone and meaning. Only return the translation:
            
            Text: {text}
            
            Translation:"""
            
            llm = self._load_llm()
            if llm:
                translated = self._generate_text_with_llm(llm, llm_prompt, max_length=200)
                if translated and translated.strip() != text.strip():
                    return translated.strip()
            
            translated = self._translate_text_fallback(text, target_language)
            if translated and translated.strip() != text.strip():
                return translated.strip()
                
            return text
            
        except Exception as e:
            logger.error(f"Enhanced translation error for {target_language}: {e}")
            return text

script_translator = ScriptTranslator()
