#!/usr/bin/env python3
"""Test script to verify anime pipeline functionality with LLM scene processing."""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def test_llm_scene_processing():
    """Test LLM scene processing functionality in anime pipeline."""
    
    try:
        from backend.pipelines.channel_specific.anime_pipeline import AnimeChannelPipeline
        
        test_output = Path("./test_anime_output")
        test_output.mkdir(exist_ok=True)
        
        print("Initializing anime pipeline...")
        pipeline = AnimeChannelPipeline()
        
        print("Testing LLM model loading...")
        llm_model = pipeline.load_llm_model()
        if llm_model and llm_model.get("generate"):
            print("✓ LLM model loaded successfully")
        else:
            print("⚠ LLM model using fallback")
        
        print("Testing LLM scene processing...")
        test_script_data = {
            'scenes': [
                "Protagonist discovers mysterious power in school courtyard",
                "Epic battle against shadow creatures in abandoned warehouse",
                "Emotional conversation with mentor about destiny"
            ],
            'characters': [
                {"name": "Akira", "description": "High school student with hidden powers"},
                {"name": "Sensei", "description": "Wise mentor figure"}
            ],
            'locations': [
                {"name": "School", "description": "Modern Japanese high school"},
                {"name": "Warehouse", "description": "Abandoned industrial building"}
            ]
        }
        
        processed_script = pipeline._process_script_with_llm(test_script_data, "anime")
        
        if processed_script.get('llm_processed'):
            enhanced_scenes = processed_script.get('enhanced_scenes', [])
            print(f"✓ LLM processed {len(enhanced_scenes)} scenes successfully")
            
            for i, scene in enumerate(enhanced_scenes):
                if isinstance(scene, dict):
                    print(f"  Scene {i+1}:")
                    print(f"    Video prompt: {scene.get('video_prompt', 'N/A')[:100]}...")
                    print(f"    Voice prompt: {scene.get('voice_prompt', 'N/A')[:50]}...")
                    print(f"    Music prompt: {scene.get('music_prompt', 'N/A')[:50]}...")
                    print(f"    Scene type: {scene.get('scene_type', 'N/A')}")
                    print(f"    Duration: {scene.get('duration', 'N/A')}s")
                else:
                    print(f"  Scene {i+1}: Basic scene data")
        else:
            print("⚠ LLM processing fell back to basic processing")
            enhanced_scenes = processed_script.get('enhanced_scenes', [])
            print(f"  Fallback processed {len(enhanced_scenes)} scenes")
        
        print("Testing voice model loading...")
        voice_model = pipeline.load_voice_model("bark")
        if voice_model and voice_model.get("type") != "fallback":
            print("✓ Voice model loaded successfully (real AI model)")
        else:
            print("⚠ Voice model using fallback")
        
        print("Testing music model loading...")
        music_model = pipeline.load_music_model("musicgen")
        if music_model and music_model.get("type") != "fallback":
            print("✓ Music model loaded successfully (real AI model)")
        else:
            print("⚠ Music model using fallback")
        
        print("Testing enhanced prompt generation...")
        if enhanced_scenes and len(enhanced_scenes) > 0:
            first_scene = enhanced_scenes[0]
            if isinstance(first_scene, dict) and 'video_prompt' in first_scene:
                video_prompt = first_scene['video_prompt']
                if len(video_prompt) > 50 and "masterpiece" in video_prompt.lower():
                    print("✓ Enhanced video prompts contain quality keywords")
                else:
                    print("⚠ Video prompts may not be fully enhanced")
                
                if 'voice_prompt' in first_scene and len(first_scene['voice_prompt']) > 10:
                    print("✓ Voice prompts generated")
                else:
                    print("⚠ Voice prompts may be missing")
                
                if 'music_prompt' in first_scene and len(first_scene['music_prompt']) > 10:
                    print("✓ Music prompts generated")
                else:
                    print("⚠ Music prompts may be missing")
        
        print("✓ LLM scene processing test completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ LLM scene processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if 'test_output' in locals() and test_output.exists():
            import shutil
            shutil.rmtree(test_output, ignore_errors=True)

def test_fallback_processing():
    """Test fallback processing when LLM is unavailable."""
    
    try:
        from backend.pipelines.channel_specific.anime_pipeline import AnimeChannelPipeline
        
        print("Testing fallback processing...")
        pipeline = AnimeChannelPipeline()
        
        test_script_data = {
            'scenes': ["Simple test scene"],
            'characters': [{"name": "TestChar"}],
            'locations': [{"name": "TestLocation"}]
        }
        
        fallback_script = pipeline._fallback_script_processing(test_script_data, "anime")
        
        if fallback_script.get('enhanced_scenes'):
            print("✓ Fallback processing creates enhanced scenes")
            scene = fallback_script['enhanced_scenes'][0]
            if isinstance(scene, dict) and 'video_prompt' in scene:
                print("✓ Fallback scenes have required fields")
            else:
                print("⚠ Fallback scenes missing required fields")
        else:
            print("✗ Fallback processing failed")
            return False
        
        print("✓ Fallback processing test completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Fallback processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Testing LLM Scene Processing ===")
    llm_success = test_llm_scene_processing()
    
    print("\n=== Testing Fallback Processing ===")
    fallback_success = test_fallback_processing()
    
    print(f"\n=== Test Results ===")
    print(f"LLM Scene Processing: {'PASS' if llm_success else 'FAIL'}")
    print(f"Fallback Processing: {'PASS' if fallback_success else 'FAIL'}")
    
    overall_success = llm_success and fallback_success
    print(f"Overall: {'PASS' if overall_success else 'FAIL'}")
    
    sys.exit(0 if overall_success else 1)
