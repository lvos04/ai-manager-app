import asyncio
import logging
import os
from typing import Dict, List, Optional, Any
from pathlib import Path
from .script_translator import script_translator
from .language_manager import language_manager

logger = logging.getLogger(__name__)

class MultiLanguagePipelineManager:
    """Manages multi-language pipeline execution for separate video generation per language."""
    
    def __init__(self):
        self.supported_languages = language_manager.get_all_languages()
    
    async def execute_multi_language_pipeline(self, scenes: List[Dict], config: Dict, selected_languages: List[str]) -> Dict[str, Any]:
        """
        Execute sequential multi-language pipeline: generate base video first, then create language-specific audio.
        """
        results = {}
        successful_languages = []
        failed_languages = []
        
        base_output_path = Path(config.get("output_path", "/tmp/multi_lang_output"))
        base_output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Step 1: Generating base video content...")
        base_config = config.copy()
        base_video_path = base_output_path / "base_video"
        base_video_path.mkdir(parents=True, exist_ok=True)
        
        base_config.update({
            "output_path": str(base_video_path),
            "language": "en",
            "audio_only": False,
            "generate_base_video": True
        })
        
        (base_output_path / "base_video").mkdir(parents=True, exist_ok=True)
        
        try:
            from ..core.async_pipeline_manager import get_async_pipeline_manager
            async_manager = get_async_pipeline_manager()
            
            base_result = await async_manager.execute_pipeline_async(scenes, base_config)
            if not base_result.get("success", False):
                raise Exception("Base video generation failed")
            
            logger.info("Base video generation completed successfully")
            
        except Exception as e:
            logger.error(f"Base video generation failed: {e}")
            return {
                "multi_language_results": {},
                "performance_metrics": {
                    "success_rate": 0,
                    "total_languages": len(selected_languages),
                    "successful_languages": 0,
                    "failed_languages": len(selected_languages),
                    "error": "Base video generation failed"
                }
            }
        
        for language_code in selected_languages:
            if language_code not in self.supported_languages:
                logger.warning(f"Language {language_code} not supported, skipping")
                failed_languages.append(language_code)
                continue
            
            try:
                language_result = await self._create_language_specific_version(
                    scenes, config, language_code, base_output_path, base_result
                )
                results[language_code] = language_result
                successful_languages.append(language_code)
                logger.info(f"Successfully completed language-specific processing for {language_code}")
                
            except Exception as e:
                logger.error(f"Language processing failed for {language_code}: {e}")
                results[language_code] = {"error": str(e)}
                failed_languages.append(language_code)
        
        total_languages = len(selected_languages)
        successful_count = len(successful_languages)
        success_rate = successful_count / total_languages if total_languages > 0 else 0
        
        return {
            "multi_language_results": results,
            "performance_metrics": {
                "success_rate": success_rate,
                "total_languages": total_languages,
                "successful_languages": successful_count,
                "failed_languages": len(failed_languages),
                "successful_language_codes": successful_languages,
                "failed_language_codes": failed_languages
            }
        }
    
    async def _create_language_specific_version(self, scenes: List[Dict], config: Dict, language_code: str, base_output_path: Path, base_result: Dict) -> Dict[str, Any]:
        """Create language-specific version by combining base video with translated audio."""
        lang_output_dir = base_output_path / f"output_{language_code}"
        lang_output_dir.mkdir(parents=True, exist_ok=True)
        
        if language_code != 'en':
            translated_scenes = await self._translate_scenes_for_language(scenes, language_code)
        else:
            translated_scenes = scenes
        
        lang_config = config.copy()
        lang_config.update({
            "output_path": str(lang_output_dir),
            "language": language_code,
            "voice_language": language_manager.get_voice_language_code(language_code),
            "llm_language_prompt": language_manager.get_llm_language_prompt(language_code),
            "audio_only": True,
            "base_video_path": str(base_output_path / "base_video")
        })
        
        from ..core.async_pipeline_manager import get_async_pipeline_manager
        async_manager = get_async_pipeline_manager()
        
        audio_result = await async_manager.execute_pipeline_async(translated_scenes, lang_config)
        
        if not audio_result.get("success", False):
            raise Exception(f"Audio generation failed for {language_code}")
        
        await self._combine_video_with_audio(base_output_path, lang_output_dir, language_code, lang_config)
        
        await self._generate_shorts_for_language(translated_scenes, lang_config, language_code)
        
        return audio_result
    
    async def _translate_scenes_for_language(self, scenes: List[Dict], language_code: str) -> List[Dict]:
        """Translate scenes to target language."""
        try:
            script_data = {"scenes": scenes}
            translated_scripts = await script_translator.translate_script_to_languages(script_data, [language_code])
            translated_scenes = translated_scripts.get(language_code, {}).get('scenes', scenes)
            
            logger.info(f"Successfully translated {len(scenes)} scenes to {language_code}")
            return translated_scenes
            
        except Exception as e:
            logger.error(f"Translation failed for {language_code}, using original scenes: {e}")
            return scenes
    
    async def _generate_shorts_for_language(self, scenes: List[Dict], config: Dict, language_code: str):
        """Generate short-form videos for the language."""
        try:
            from ..core.async_pipeline_manager import get_async_pipeline_manager
            
            shorts_output_dir = Path(config["output_path"]) / "shorts"
            shorts_output_dir.mkdir(parents=True, exist_ok=True)
            
            async_manager = get_async_pipeline_manager()
            
            for i, scene in enumerate(scenes[:3]):
                short_config = config.copy()
                short_config.update({
                    "output_path": str(shorts_output_dir),
                    "video_format": "vertical",
                    "duration": min(float(str(scene.get("duration", 10.0))), 60.0),
                    "output_filename": f"short_{i+1}_{language_code}.mp4"
                })
                
                logger.info(f"Generating short {i+1} for {language_code}")
                
                short_result = await async_manager.execute_pipeline_async([scene], short_config)
                
                if short_result.get("success"):
                    logger.info(f"Short {i+1} generated successfully for {language_code}")
                else:
                    logger.error(f"Short {i+1} generation failed for {language_code}: {short_result.get('error')}")
                
        except Exception as e:
            logger.error(f"Shorts generation failed for {language_code}: {e}")
    
    async def _assemble_final_video(self, output_dir: Path, language_code: str, config: Optional[Dict] = None):
        """Assemble individual scene videos into final video."""
        try:
            from ..pipelines.channel_specific.anime_pipeline import combine_scenes_to_episode
            import glob
            
            if config is None:
                config = {}
            
            scenes_dir = output_dir
            scene_files = sorted(glob.glob(str(scenes_dir / "scene_*.mp4")))
            
            if scene_files:
                final_video_path = output_dir / f"final_video_{language_code}.mp4"
                logger.info(f"Assembling {len(scene_files)} scenes into final video: {final_video_path}")
                
                combine_scenes_to_episode(scenes_dir, str(final_video_path), 
                                        frame_interpolation_enabled=config.get("frame_interpolation_enabled", True),
                                        render_fps=config.get("render_fps", 24),
                                        output_fps=config.get("output_fps", 24))
                logger.info(f"Final video assembled successfully for {language_code}")
            else:
                logger.warning(f"No scene videos found for final assembly in {language_code}")
                
        except Exception as e:
            logger.error(f"Final video assembly failed for {language_code}: {e}")

    async def _combine_video_with_audio(self, base_path: Path, lang_output_dir: Path, language_code: str, config: Dict):
        """Combine base video with language-specific audio tracks."""
        try:
            from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
            import glob
            
            base_video_files = sorted(glob.glob(str(base_path / "base_video" / "scene_*.mp4")))
            audio_files = sorted(glob.glob(str(lang_output_dir / "voice_*.wav")))
            music_files = sorted(glob.glob(str(lang_output_dir / "music_*.wav")))
            
            combined_scenes = []
            
            for i, video_file in enumerate(base_video_files):
                video_clip = VideoFileClip(video_file)
                
                if i < len(audio_files) and os.path.exists(audio_files[i]):
                    voice_clip = AudioFileClip(audio_files[i])
                    
                    if i < len(music_files) and os.path.exists(music_files[i]):
                        music_clip = AudioFileClip(music_files[i])
                        music_clip = music_clip.volumex(0.3)
                        final_audio = CompositeAudioClip([voice_clip, music_clip])
                    else:
                        final_audio = voice_clip
                    
                    final_clip = video_clip.set_audio(final_audio)
                else:
                    final_clip = video_clip
                
                scene_output = lang_output_dir / f"scene_{i+1:03d}_{language_code}.mp4"
                final_clip.write_videofile(str(scene_output), codec='libx264', audio_codec='aac')
                combined_scenes.append(str(scene_output))
                
                video_clip.close()
                if 'voice_clip' in locals():
                    voice_clip.close()
                if 'music_clip' in locals():
                    music_clip.close()
                if 'final_audio' in locals():
                    final_audio.close()
                final_clip.close()
            
            await self._assemble_final_video(lang_output_dir, language_code, config)
            
            logger.info(f"Successfully combined video with audio for {language_code}")
            
        except Exception as e:
            logger.error(f"Video-audio combination failed for {language_code}: {e}")
            raise
        finally:
            try:
                self._force_memory_cleanup()
            except Exception as cleanup_error:
                logger.warning(f"Memory cleanup failed: {cleanup_error}")
    
    def _force_memory_cleanup(self):
        """Force memory cleanup."""
        try:
            import gc
            gc.collect()
            
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except ImportError:
                pass
                
        except Exception as e:
            logger.error(f"Memory cleanup error: {e}")

multi_language_pipeline_manager = MultiLanguagePipelineManager()
