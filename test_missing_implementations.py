#!/usr/bin/env python3
"""Test script to verify all missing implementations are fixed."""

import sys
import importlib
from pathlib import Path

def test_gaming_pipeline_imports():
    """Test that gaming pipeline imports without undefined name errors."""
    try:
        from backend.pipelines.channel_specific.gaming_pipeline import (
            GamingChannelPipeline,
            process_game_recording,
            generate_shorts_from_video,
            generate_ai_shorts,
            get_optimal_model_for_channel,
            create_scene_video_with_generation,
            create_fallback_video
        )
        print("✅ Gaming pipeline imports successful")
        return True
    except Exception as e:
        print(f"❌ Gaming pipeline import error: {e}")
        return False

def test_ai_model_manager():
    """Test AIModelManager class."""
    try:
        from backend.core.ai_model_manager import AIModelManager
        manager = AIModelManager()
        print("✅ AIModelManager creation successful")
        return True
    except Exception as e:
        print(f"❌ AIModelManager error: {e}")
        return False

def test_character_memory_cleanup():
    """Test character memory cleanup method."""
    try:
        from backend.core.character_memory import CharacterMemoryManager
        manager = CharacterMemoryManager()
        manager.cleanup_old_references(30)
        print("✅ Character memory cleanup successful")
        return True
    except Exception as e:
        print(f"❌ Character memory cleanup error: {e}")
        return False

def test_progressive_loader_optimization():
    """Test progressive loader UI optimization."""
    try:
        from backend.core.progressive_loader import ProgressiveLoader
        loader = ProgressiveLoader()
        loader.optimize_ui_responsiveness()
        print("✅ Progressive loader optimization successful")
        return True
    except Exception as e:
        print(f"❌ Progressive loader optimization error: {e}")
        return False

if __name__ == "__main__":
    tests = [
        test_gaming_pipeline_imports,
        test_ai_model_manager,
        test_character_memory_cleanup,
        test_progressive_loader_optimization
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\nResults: {passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
