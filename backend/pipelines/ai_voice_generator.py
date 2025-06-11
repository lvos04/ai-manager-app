"""
Real AI Voice Generator implementation with Bark and XTTS integration.
Replaces placeholder voice generation with actual AI models.
"""

import os
import sys
import json
import logging
import tempfile
import shutil
import subprocess
import time
import gc
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple

import torch
import numpy as np
import scipy.io.wavfile as wavfile

logger = logging.getLogger(__name__)

class AIVoiceGenerator:
    """Real AI voice generation using Bark and XTTS models."""
    
    def __init__(self, vram_tier: str = "medium"):
        self.vram_tier = vram_tier
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.models = {}
        self.current_model = None
        
        self.model_settings = {
            "bark": {
                "low": {"use_gpu": False, "offload_cpu": True, "precision": "float32"},
                "medium": {"use_gpu": True, "offload_cpu": True, "precision": "float16"},
                "high": {"use_gpu": True, "offload_cpu": False, "precision": "float16"},
                "ultra": {"use_gpu": True, "offload_cpu": False, "precision": "float16"}
            },
            "xtts": {
                "low": {"use_gpu": False, "streaming": False, "precision": "float32"},
                "medium": {"use_gpu": True, "streaming": False, "precision": "float16"},
                "high": {"use_gpu": True, "streaming": True, "precision": "float16"},
                "ultra": {"use_gpu": True, "streaming": True, "precision": "float16"}
            }
        }
        
        self.language_support = {
            "bark": ["en", "ja", "es", "pt", "ru", "fr", "de"],
            "xtts": ["en", "ja", "es", "zh", "hi", "ar", "bn", "pt", "ru", "fr", "de"]
        }
        
        self.voice_presets = {
            "bark": {
                "en": ["v2/en_speaker_0", "v2/en_speaker_1", "v2/en_speaker_2", "v2/en_speaker_3"],
                "ja": ["v2/ja_speaker_0", "v2/ja_speaker_1"],
                "es": ["v2/es_speaker_0", "v2/es_speaker_1"],
                "pt": ["v2/pt_speaker_0", "v2/pt_speaker_1"],
                "ru": ["v2/ru_speaker_0", "v2/ru_speaker_1"],
                "fr": ["v2/fr_speaker_0", "v2/fr_speaker_1"],
                "de": ["v2/de_speaker_0", "v2/de_speaker_1"]
            }
        }
    
    def get_best_model_for_language(self, language: str) -> str:
        """Select optimal voice model based on language."""
        if language in self.language_support["bark"]:
            return "bark"
        elif language in self.language_support["xtts"]:
            return "xtts"
        else:
            return "bark"
    
    def load_model(self, model_name: str) -> bool:
        """Load voice generation model."""
        try:
            if model_name == self.current_model and model_name in self.models:
                return True
            
            self.force_cleanup_all_models()
            
            logger.info(f"Loading voice model: {model_name}")
            
            if model_name == "bark":
                return self._load_bark()
            elif model_name == "xtts":
                return self._load_xtts()
            else:
                logger.warning(f"Unknown voice model: {model_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading voice model {model_name}: {e}")
            return False
    
    def _load_bark(self) -> bool:
        """Load Bark voice model with fallback handling."""
        try:
            try:
                from bark import SAMPLE_RATE, generate_audio, preload_models
                from bark.generation import set_seed
            except ImportError:
                logger.warning("Bark package not available (Python 3.12 compatibility issue)")
                return False
            
            settings = self.model_settings["bark"].get(self.vram_tier, self.model_settings["bark"]["medium"])
            
            if settings["use_gpu"] and torch.cuda.is_available():
                os.environ["SUNO_USE_SMALL_MODELS"] = "False"
                os.environ["SUNO_OFFLOAD_CPU"] = "True" if settings["offload_cpu"] else "False"
            else:
                os.environ["SUNO_USE_SMALL_MODELS"] = "True"
                os.environ["SUNO_OFFLOAD_CPU"] = "True"
            
            preload_models()
            set_seed(42)
            
            self.models["bark"] = {
                "generate_audio": generate_audio,
                "sample_rate": SAMPLE_RATE,
                "settings": settings
            }
            
            self.current_model = "bark"
            logger.info("Bark model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading Bark: {e}")
            return False
    
    def _load_xtts(self) -> bool:
        """Load XTTS voice model with fallback for missing TTS package."""
        try:
            try:
                from TTS.api import TTS
            except ImportError:
                logger.warning("TTS package not available (Python 3.12 compatibility issue)")
                return False
            
            settings = self.model_settings["xtts"].get(self.vram_tier, self.model_settings["xtts"]["medium"])
            
            device = "cuda" if settings["use_gpu"] and torch.cuda.is_available() else "cpu"
            
            tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
            
            self.models["xtts"] = {
                "tts": tts,
                "settings": settings,
                "device": device
            }
            
            self.current_model = "xtts"
            logger.info("XTTS model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading XTTS: {e}")
            return False
    
    def generate_voice(self, text: str, model_name: str, output_path: str, 
                      language: str = "en", character_voice: str = "default") -> bool:
        """Generate voice audio using specified model."""
        try:
            if not self.load_model(model_name):
                logger.error(f"Failed to load voice model: {model_name}")
                self._log_voice_generation_error(text, output_path, f"Failed to load voice model: {model_name}")
                return False
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            logger.info(f"Generating voice with {model_name}: {text[:50]}...")
            
            if model_name == "bark":
                return self._generate_bark_voice(text, output_path, language, character_voice)
            elif model_name == "xtts":
                return self._generate_xtts_voice(text, output_path, language, character_voice)
            else:
                logger.error(f"Unknown voice generation method for {model_name}")
                self._log_voice_generation_error(text, output_path, f"Unknown voice generation method for {model_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error generating voice: {e}")
            self._log_voice_generation_error(text, output_path, str(e))
            return False
    
    def _generate_bark_voice(self, text: str, output_path: str, language: str, character_voice: str) -> bool:
        """Generate voice using Bark."""
        try:
            bark_model = self.models["bark"]
            generate_audio = bark_model["generate_audio"]
            sample_rate = bark_model["sample_rate"]
            
            voice_preset = self._get_bark_voice_preset(language, character_voice)
            
            if len(text) > 200:
                return self._generate_long_bark_voice(text, output_path, voice_preset, generate_audio, sample_rate)
            
            audio_array = generate_audio(text, history_prompt=voice_preset)
            
            audio_array = (audio_array * 32767).astype(np.int16)
            
            wavfile.write(output_path, sample_rate, audio_array)
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                logger.info(f"Bark voice generated successfully: {output_path}")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error in Bark generation: {e}")
            return False
    
    def _generate_long_bark_voice(self, text: str, output_path: str, voice_preset: str, 
                                 generate_audio, sample_rate: int) -> bool:
        """Generate long voice using Bark with text splitting."""
        try:
            sentences = self._split_text_for_bark(text)
            audio_segments = []
            
            for i, sentence in enumerate(sentences):
                if not sentence.strip():
                    continue
                
                logger.info(f"Generating Bark segment {i+1}/{len(sentences)}")
                
                audio_array = generate_audio(sentence.strip(), history_prompt=voice_preset)
                audio_segments.append(audio_array)
                
                if i < len(sentences) - 1:
                    silence = np.zeros(int(sample_rate * 0.3))
                    audio_segments.append(silence)
            
            if audio_segments:
                combined_audio = np.concatenate(audio_segments)
                combined_audio = (combined_audio * 32767).astype(np.int16)
                
                wavfile.write(output_path, sample_rate, combined_audio)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error in long Bark generation: {e}")
            return False
    
    def _generate_xtts_voice(self, text: str, output_path: str, language: str, character_voice: str) -> bool:
        """Generate voice using XTTS."""
        try:
            xtts_model = self.models["xtts"]
            tts = xtts_model["tts"]
            
            speaker_wav = self._get_xtts_speaker_reference(language, character_voice)
            
            if len(text) > 500:
                return self._generate_long_xtts_voice(text, output_path, tts, language, speaker_wav)
            
            tts.tts_to_file(
                text=text,
                speaker_wav=speaker_wav,
                language=language,
                file_path=output_path
            )
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                logger.info(f"XTTS voice generated successfully: {output_path}")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error in XTTS generation: {e}")
            return False
    
    def _generate_long_xtts_voice(self, text: str, output_path: str, tts, language: str, speaker_wav: str) -> bool:
        """Generate long voice using XTTS with text splitting."""
        try:
            sentences = self._split_text_for_xtts(text)
            temp_files = []
            
            for i, sentence in enumerate(sentences):
                if not sentence.strip():
                    continue
                
                logger.info(f"Generating XTTS segment {i+1}/{len(sentences)}")
                
                temp_file = f"{output_path}.temp_{i}.wav"
                temp_files.append(temp_file)
                
                tts.tts_to_file(
                    text=sentence.strip(),
                    speaker_wav=speaker_wav,
                    language=language,
                    file_path=temp_file
                )
            
            if temp_files:
                self._combine_audio_files(temp_files, output_path)
                
                for temp_file in temp_files:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error in long XTTS generation: {e}")
            return False
    
    def _get_bark_voice_preset(self, language: str, character_voice: str) -> str:
        """Get Bark voice preset for language and character."""
        if language in self.voice_presets["bark"]:
            presets = self.voice_presets["bark"][language]
            if character_voice == "female" and len(presets) > 1:
                return presets[1]
            elif character_voice == "male" and len(presets) > 0:
                return presets[0]
            else:
                return presets[0] if presets else "v2/en_speaker_0"
        else:
            return "v2/en_speaker_0"
    
    def _get_xtts_speaker_reference(self, language: str, character_voice: str) -> str:
        """Get XTTS speaker reference audio."""
        try:
            models_dir = Path.home() / "repos" / "ai-manager-app" / "models" / "audio" / "xtts_speakers"
            
            speaker_files = {
                "en": {
                    "male": models_dir / "en_male_speaker.wav",
                    "female": models_dir / "en_female_speaker.wav",
                    "default": models_dir / "en_default_speaker.wav"
                },
                "ja": {
                    "male": models_dir / "ja_male_speaker.wav",
                    "female": models_dir / "ja_female_speaker.wav",
                    "default": models_dir / "ja_default_speaker.wav"
                }
            }
            
            lang_speakers = speaker_files.get(language, speaker_files["en"])
            speaker_file = lang_speakers.get(character_voice, lang_speakers["default"])
            
            if speaker_file.exists():
                return str(speaker_file)
            
            return str(lang_speakers["default"]) if lang_speakers["default"].exists() else None
            
        except Exception as e:
            logger.error(f"Error getting XTTS speaker reference: {e}")
            return None
    
    def _split_text_for_bark(self, text: str) -> List[str]:
        """Split text into chunks suitable for Bark."""
        sentences = []
        current_chunk = ""
        
        for sentence in text.split('.'):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if len(current_chunk + sentence) < 150:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    sentences.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            sentences.append(current_chunk.strip())
        
        return sentences
    
    def _split_text_for_xtts(self, text: str) -> List[str]:
        """Split text into chunks suitable for XTTS."""
        sentences = []
        current_chunk = ""
        
        for sentence in text.split('.'):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if len(current_chunk + sentence) < 400:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    sentences.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            sentences.append(current_chunk.strip())
        
        return sentences
    
    def _combine_audio_files(self, audio_files: List[str], output_path: str):
        """Combine multiple audio files into one."""
        try:
            from pydub import AudioSegment
            
            combined = AudioSegment.empty()
            
            for audio_file in audio_files:
                if os.path.exists(audio_file):
                    audio = AudioSegment.from_wav(audio_file)
                    combined += audio
                    combined += AudioSegment.silent(duration=300)  # 300ms silence between segments
            
            combined.export(output_path, format="wav")
            
        except Exception as e:
            logger.error(f"Error combining audio files: {e}")
            self._combine_audio_with_pydub(audio_files, output_path)
    
    def _combine_audio_with_pydub(self, audio_files: List[str], output_path: str):
        """Combine audio files using pydub instead of FFmpeg fallback."""
        try:
            from pydub import AudioSegment
            
            combined = AudioSegment.empty()
            
            for audio_file in audio_files:
                if os.path.exists(audio_file):
                    try:
                        audio = AudioSegment.from_file(audio_file)
                        combined += audio
                    except Exception as e:
                        logger.warning(f"Could not load audio file {audio_file}: {e}")
            
            if len(combined) > 0:
                combined.export(output_path, format="wav")
                logger.info("Audio files combined successfully with pydub")
            else:
                logger.error("No valid audio files to combine")
                
        except Exception as e:
            logger.error(f"Error in pydub audio combination: {e}")
    
    def _log_voice_generation_error(self, text: str, output_path: str, error_message: str):
        """Log voice generation error to output directory."""
        try:
            from ..utils.error_handler import PipelineErrorHandler
            import os
            
            output_dir = os.path.dirname(output_path) if output_path else '/tmp'
            error_handler = PipelineErrorHandler()
            voice_error = Exception(f"Voice generation failed: {error_message}")
            error_handler.log_error_to_output(
                error=voice_error,
                output_path=output_dir,
                context={
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "output_path": output_path,
                    "error_details": error_message
                }
            )
            logger.error(f"Voice generation failed, error logged to output directory")
            
        except Exception as e:
            logger.error(f"Error logging voice generation failure: {e}")
    
    def force_cleanup_all_models(self):
        """Force cleanup of all loaded voice models."""
        try:
            for model_name in list(self.models.keys()):
                if model_name in self.models:
                    del self.models[model_name]
            
            self.models.clear()
            self.current_model = None
            
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info("All voice models cleaned up")
            
        except Exception as e:
            logger.error(f"Error in voice model cleanup: {e}")
