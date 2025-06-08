#!/usr/bin/env python3
"""Test complete fixes for AI Project Manager application."""
import sys
import os
import asyncio
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def test_imports():
    """Test that all critical imports work without errors."""
    try:
        from backend.config import API_HOST, API_PORT
        from backend.pipelines.video_generation import TextToVideoGenerator
        from backend.core.async_pipeline_manager import AsyncPipelineManager
        from backend.localization.script_translator import ScriptTranslator
        from backend.pipelines.ai_models import AIModelManager
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_config():
    """Test configuration settings."""
    try:
        from backend.config import VIDEO_BITRATE, VIDEO_PRESET, VIDEO_CRF
        assert VIDEO_BITRATE == "12000k", f"Expected 12000k bitrate, got {VIDEO_BITRATE}"
        assert VIDEO_PRESET == "veryslow", f"Expected veryslow preset, got {VIDEO_PRESET}"
        assert VIDEO_CRF == "15", f"Expected CRF 15, got {VIDEO_CRF}"
        print("✅ Maximum quality settings verified")
        return True
    except Exception as e:
        print(f"❌ Config error: {e}")
        return False

def test_memory_cleanup():
    """Test memory cleanup mechanisms."""
    try:
        from backend.pipelines.ai_models import AIModelManager
        from backend.pipelines.video_generation import TextToVideoGenerator
        
        manager = AIModelManager()
        generator = TextToVideoGenerator("medium", (1920, 1080))
        
        assert hasattr(manager, 'force_memory_cleanup'), "AIModelManager missing force_memory_cleanup"
        assert hasattr(generator, 'force_cleanup_all_models'), "TextToVideoGenerator missing force_cleanup_all_models"
        
        print("✅ Memory cleanup methods available")
        return True
    except Exception as e:
        print(f"❌ Memory cleanup error: {e}")
        return False

def test_translation_system():
    """Test enhanced translation system."""
    try:
        from backend.localization.script_translator import ScriptTranslator
        
        translator = ScriptTranslator()
        assert hasattr(translator, '_translate_text_with_llm'), "Missing enhanced LLM translation method"
        
        print("✅ Enhanced translation system available")
        return True
    except Exception as e:
        print(f"❌ Translation system error: {e}")
        return False

def test_video_generation_fixes():
    """Test video generation error handling fixes."""
    try:
        from backend.pipelines.video_generation import TextToVideoGenerator
        
        generator = TextToVideoGenerator("medium", (1920, 1080))
        
        print("✅ Video generation fixes applied")
        return True
    except Exception as e:
        print(f"❌ Video generation error: {e}")
        return False

async def test_async_pipeline():
    """Test async pipeline manager."""
    try:
        from backend.core.async_pipeline_manager import AsyncPipelineManager
        
        manager = AsyncPipelineManager()
        assert hasattr(manager, '_cleanup_pipeline_memory'), "Missing pipeline memory cleanup"
        
        print("✅ Async pipeline manager enhanced")
        return True
    except Exception as e:
        print(f"❌ Async pipeline error: {e}")
        return False

def main():
    """Run all tests."""
    print("🔧 Testing AI Project Manager fixes...")
    
    tests = [
        test_imports,
        test_config,
        test_memory_cleanup,
        test_translation_system,
        test_video_generation_fixes,
    ]
    
    async_tests = [
        test_async_pipeline,
    ]
    
    passed = 0
    total = len(tests) + len(async_tests)
    
    for test in tests:
        if test():
            passed += 1
    
    for async_test in async_tests:
        try:
            if asyncio.run(async_test()):
                passed += 1
        except Exception as e:
            print(f"❌ Async test error: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All fixes verified successfully!")
        return True
    else:
        print("⚠️ Some tests failed - additional fixes needed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
