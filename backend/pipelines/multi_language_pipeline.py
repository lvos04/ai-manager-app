import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import os

logger = logging.getLogger(__name__)

class MultiLanguagePipeline:
    """Pipeline for processing content in multiple languages simultaneously."""
    
    def __init__(self):
        self.supported_languages = {
            'en': 'English',
            'nl': 'Dutch', 
            'de': 'German',
            'fr': 'French',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'es': 'Spanish'
        }
    
    async def run_async(self, input_path: str, output_path: str, languages: List[str],
                       channel_type: str = "anime", base_model: str = "stable_diffusion_1_5",
                       lora_models: List[str] = None, lora_paths: Dict[str, str] = None,
                       render_fps: int = 24, output_fps: int = 24,
                       frame_interpolation_enabled: bool = True) -> str:
        """Run multi-language pipeline asynchronously."""
        try:
            logger.info(f"Starting multi-language pipeline for languages: {languages}")
            
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            results = {}
            tasks = []
            
            for language in languages:
                if language not in self.supported_languages:
                    logger.warning(f"Language {language} not supported, skipping")
                    continue
                
                lang_output_dir = output_dir / f"output_{language}"
                lang_output_dir.mkdir(parents=True, exist_ok=True)
                
                project_data = {
                    'input_path': input_path,
                    'output_path': str(lang_output_dir),
                    'channel_type': channel_type,
                    'base_model': base_model,
                    'lora_models': lora_models or [],
                    'lora_paths': lora_paths or {},
                    'language': language,
                    'render_fps': render_fps,
                    'output_fps': output_fps,
                    'frame_interpolation_enabled': frame_interpolation_enabled
                }
                
                task = self._process_language(project_data, language)
                tasks.append(task)
            
            if not tasks:
                logger.error("No valid languages to process")
                return output_path
            
            language_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(language_results):
                language = languages[i] if i < len(languages) else f"lang_{i}"
                if isinstance(result, Exception):
                    logger.error(f"Error processing language {language}: {result}")
                    results[language] = {"success": False, "error": str(result)}
                else:
                    results[language] = {"success": True, "output_path": result}
            
            self._create_multi_language_manifest(output_dir, results, languages)
            
            logger.info(f"Multi-language pipeline completed for {len(results)} languages")
            return str(output_dir)
            
        except Exception as e:
            logger.error(f"Multi-language pipeline failed: {e}")
            raise
    
    async def _process_language(self, project_data: Dict[str, Any], language: str) -> Optional[str]:
        """Process content for a specific language."""
        try:
            from ..core.async_pipeline_manager import AsyncPipelineManager
            
            logger.info(f"Processing content for language: {language}")
            
            async_manager = AsyncPipelineManager()
            result = await async_manager.execute_pipeline_async(project_data)
            
            if result.get("success"):
                logger.info(f"Multi-language pipeline completed successfully for {language}")
                return result.get("output_path", project_data['output_path'])
            else:
                logger.error(f"Multi-language pipeline failed for {language}: {result.get('error', 'Unknown error')}")
                return None
                
        except Exception as e:
            logger.error(f"Error in multi-language pipeline execution for {language}: {e}")
            return None
    
    def _create_multi_language_manifest(self, output_dir: Path, results: Dict[str, Any], languages: List[str]):
        """Create a manifest file for multi-language output."""
        try:
            manifest = {
                "type": "multi_language_content",
                "languages": languages,
                "supported_languages": self.supported_languages,
                "results": results,
                "total_languages": len(languages),
                "successful_languages": sum(1 for r in results.values() if r.get("success")),
                "failed_languages": sum(1 for r in results.values() if not r.get("success"))
            }
            
            manifest_file = output_dir / "multi_language_manifest.json"
            with open(manifest_file, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Multi-language manifest created: {manifest_file}")
            
        except Exception as e:
            logger.error(f"Error creating multi-language manifest: {e}")
