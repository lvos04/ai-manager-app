#!/usr/bin/env python3
"""
Comprehensive test script to verify all pipeline fixes are working correctly.
"""

import sys
import os

sys.path.insert(0, os.path.abspath('.'))

def test_imports():
    """Test that all critical imports work correctly."""
    try:
        from backend.pipelines.utils.error_handler import PipelineErrorHandler
        print("✅ PipelineErrorHandler import successful")
        
        from backend.pipelines.channel_specific.anime_pipeline import AnimeChannelPipeline
        print("✅ AnimeChannelPipeline import successful")
        
        from backend.pipelines.channel_specific.gaming_pipeline import GamingChannelPipeline
        print("✅ GamingChannelPipeline import successful")
        
        from backend.pipelines.channel_specific.superhero_pipeline import SuperheroChannelPipeline
        print("✅ SuperheroChannelPipeline import successful")
        
        from backend.pipelines.channel_specific.manga_pipeline import MangaChannelPipeline
        print("✅ MangaChannelPipeline import successful")
        
        from backend.pipelines.channel_specific.marvel_dc_pipeline import MarvelDCChannelPipeline
        print("✅ MarvelDCChannelPipeline import successful")
        
        from backend.pipelines.channel_specific.original_manga_pipeline import OriginalMangaChannelPipeline
        print("✅ OriginalMangaChannelPipeline import successful")
        
        from backend.pipelines.channel_specific.base_pipeline import BasePipeline
        print("✅ BasePipeline import successful")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_error_handler():
    """Test that the error handler works correctly."""
    try:
        from backend.pipelines.utils.error_handler import PipelineErrorHandler
        handler = PipelineErrorHandler()
        
        test_output_dir = "/tmp/test_error_logs"
        os.makedirs(test_output_dir, exist_ok=True)
        
        error_log_path = handler.log_error(
            error_type="TEST_ERROR",
            error_message="This is a comprehensive test error",
            output_dir=test_output_dir,
            context={"test": "comprehensive_fixes"}
        )
        
        if os.path.exists(error_log_path):
            print("✅ Error handler logging works correctly")
            return True
        else:
            print("❌ Error handler failed to create log file")
            return False
            
    except Exception as e:
        print(f"❌ Error handler test failed: {e}")
        return False

def test_method_signatures():
    """Test that method signatures are correct."""
    try:
        from backend.pipelines.channel_specific.base_pipeline import BasePipeline
        
        base = BasePipeline(channel_type="test")
        base.current_output_dir = "/tmp"
        
        base._log_llm_scene_error(1, "Test error message")
        print("✅ _log_llm_scene_error method signature correct")
        
        base._log_music_generation_error("/tmp/test.mp3", "Test music error")
        print("✅ _log_music_generation_error method signature correct")
        
        return True
    except Exception as e:
        print(f"❌ Method signature test failed: {e}")
        return False

def test_no_fallback_content():
    """Test that no fallback content is generated."""
    try:
        from backend.pipelines.channel_specific.base_pipeline import BasePipeline
        
        base = BasePipeline(channel_type="test")
        base.current_output_dir = "/tmp"
        
        result = base._create_efficient_video("test prompt", 5.0, "/tmp/test_video.mp4")
        
        if result is None:
            print("✅ No fallback content generated - returns None as expected")
            return True
        else:
            print("❌ Fallback content still being generated")
            return False
            
    except Exception as e:
        print(f"❌ Fallback content test failed: {e}")
        return False

def main():
    """Run all comprehensive tests."""
    print("Testing comprehensive pipeline fixes...")
    print("=" * 60)
    
    all_passed = True
    
    if not test_imports():
        all_passed = False
    
    print()
    
    if not test_error_handler():
        all_passed = False
    
    print()
    
    if not test_method_signatures():
        all_passed = False
    
    print()
    
    if not test_no_fallback_content():
        all_passed = False
    
    print()
    print("=" * 60)
    
    if all_passed:
        print("✅ ALL COMPREHENSIVE FIXES VERIFIED SUCCESSFULLY!")
        print("✅ No fallback content generation!")
        print("✅ All method signatures fixed!")
        print("✅ Error logging working correctly!")
        return 0
    else:
        print("❌ Some tests failed - additional fixes needed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
