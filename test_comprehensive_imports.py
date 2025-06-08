#!/usr/bin/env python3
"""Comprehensive test to verify all external imports are removed and functionality works."""

import sys
import os
import subprocess
from pathlib import Path

def test_all_imports():
    """Test all imports work correctly."""
    print("🔍 Testing all imports...")
    
    try:
        from backend.pipelines.channel_specific.anime_pipeline import AnimeChannelPipeline
        print("✓ anime_pipeline imports successfully")
    except Exception as e:
        print(f"✗ anime_pipeline import error: {e}")
        return False
    
    try:
        from backend.pipelines.channel_specific.gaming_pipeline import GamingChannelPipeline
        print("✓ gaming_pipeline imports successfully")
    except Exception as e:
        print(f"✗ gaming_pipeline import error: {e}")
        return False
    
    try:
        from backend.pipelines.channel_specific.manga_pipeline import MangaChannelPipeline
        print("✓ manga_pipeline imports successfully")
    except Exception as e:
        print(f"✗ manga_pipeline import error: {e}")
        return False
    
    try:
        from backend.pipelines.channel_specific.superhero_pipeline import SuperheroChannelPipeline
        print("✓ superhero_pipeline imports successfully")
    except Exception as e:
        print(f"✗ superhero_pipeline import error: {e}")
        return False
    
    try:
        from backend.pipelines.channel_specific.marvel_dc_pipeline import MarvelDCChannelPipeline
        print("✓ marvel_dc_pipeline imports successfully")
    except Exception as e:
        print(f"✗ marvel_dc_pipeline import error: {e}")
        return False
    
    try:
        from backend.pipelines.channel_specific.original_manga_pipeline import OriginalMangaChannelPipeline
        print("✓ original_manga_pipeline imports successfully")
    except Exception as e:
        print(f"✗ original_manga_pipeline import error: {e}")
        return False
    
    try:
        from backend.ai_tasks import run_pipeline_sync
        print("✓ ai_tasks imports successfully")
    except Exception as e:
        print(f"✗ ai_tasks import error: {e}")
        return False
    
    try:
        from backend.localization.script_translator import script_translator
        print("✓ script_translator imports successfully")
    except Exception as e:
        print(f"✗ script_translator import error: {e}")
        return False
    
    try:
        from backend.localization.multi_language_pipeline import multi_language_pipeline_manager
        print("✓ multi_language_pipeline imports successfully")
    except Exception as e:
        print(f"✗ multi_language_pipeline import error: {e}")
        return False
    
    try:
        from backend.core.async_pipeline_manager import get_async_pipeline_manager
        print("✓ async_pipeline_manager imports successfully")
    except Exception as e:
        print(f"✗ async_pipeline_manager import error: {e}")
        return False
    
    try:
        from backend.core.predictive_model_loader import get_predictive_loader
        print("✓ predictive_model_loader imports successfully")
    except Exception as e:
        print(f"✗ predictive_model_loader import error: {e}")
        return False
    
    try:
        from backend.optimization_integration import get_optimized_executor
        print("✓ optimization_integration imports successfully")
    except Exception as e:
        print(f"✗ optimization_integration import error: {e}")
        return False
    
    return True

def search_for_external_imports():
    """Search for any remaining external imports."""
    print("\n🔍 Searching for remaining external imports...")
    
    external_modules = [
        "script_expander", "ai_models", "pipeline_utils", "combat_scene_generator",
        "video_generation", "frame_interpolation", "language_support", 
        "game_recording_processor", "shorts_generator", "ai_shorts_generator", "upscaling"
    ]
    
    found_imports = []
    
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    for module in external_modules:
                        if f"from .{module}" in content or f"from ..{module}" in content or f"import {module}" in content:
                            found_imports.append(f"{file_path}: {module}")
                            
                except Exception as e:
                    continue
    
    if found_imports:
        print("❌ Found remaining external imports:")
        for imp in found_imports:
            print(f"  - {imp}")
        return False
    else:
        print("✅ No external imports found!")
        return True

def test_pipeline_execution():
    """Test basic pipeline execution."""
    print("\n🔍 Testing pipeline execution...")
    
    try:
        from backend.pipelines.channel_specific.anime_pipeline import run as run_anime
        
        import tempfile
        import yaml
        
        test_script = {
            "scenes": [
                {"description": "Test scene 1", "dialogue": "Hello world", "duration": 2.0}
            ],
            "characters": ["Test Character"],
            "locations": ["Test Location"]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_script, f)
            test_input_path = f.name
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_anime(
                input_path=test_input_path,
                output_path=temp_dir,
                base_model="stable_diffusion_1_5",
                lora_models=[],
                db_run=None,
                db=None,
                render_fps=24,
                output_fps=24,
                frame_interpolation_enabled=False,
                language="en"
            )
        
        import os
        os.unlink(test_input_path)
        
        if result:
            print("✅ Pipeline execution test passed")
            return True
        else:
            print("❌ Pipeline execution returned None")
            return False
            
    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        return False

def main():
    """Run comprehensive tests."""
    print("🚀 Starting comprehensive import and functionality tests...\n")
    
    os.chdir("/home/ubuntu/repos/ai-manager-app")
    sys.path.insert(0, ".")
    
    all_passed = True
    
    if not test_all_imports():
        all_passed = False
    
    if not search_for_external_imports():
        all_passed = False
    
    if not test_pipeline_execution():
        all_passed = False
    
    print(f"\n{'='*50}")
    if all_passed:
        print("🎉 ALL TESTS PASSED! External imports successfully removed and functionality verified.")
    else:
        print("❌ SOME TESTS FAILED! Please review and fix the issues above.")
    print(f"{'='*50}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
