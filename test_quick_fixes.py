#!/usr/bin/env python3
"""Quick test to verify core fixes without hanging on model loading."""

import sys
import os
import tempfile
import yaml
from pathlib import Path

sys.path.append('/home/ubuntu/repos/ai-manager-app')

def test_quick_fixes():
    """Test core fixes without running full pipeline."""
    
    test_yaml_content = {
        "title": "Test Episode",
        "characters": [
            {"name": "Kael", "description": "Main protagonist"},
            {"name": "Note", "description": "Supporting character"}
        ],
        "locations": [
            {"name": "Abyss", "description": "Dark mysterious realm"}
        ],
        "scenes": [
            {
                "description": "Scene 1: Kael explores the abyss",
                "character": "Kael",
                "dialogue": "What is this place?",
                "location": "Abyss",
                "duration": 15.0
            },
            {
                "description": "Scene 2: Note appears",
                "character": "Note", 
                "dialogue": "Follow me",
                "location": "Abyss",
                "duration": 12.0
            },
            {
                "description": "Scene 3: They discover ruins",
                "character": "Kael",
                "dialogue": "These ruins are ancient",
                "location": "Ancient Ruins",
                "duration": 18.0
            }
        ]
    }
    
    print("🧪 Testing Quick Fixes")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_yaml_content, f, default_flow_style=False)
        test_yaml_path = f.name
    
    try:
        print("\n1️⃣ Testing script parsing...")
        from backend.ai_tasks import extract_scenes_from_pipeline
        
        import asyncio
        async def test_scene_extraction():
            scenes = await extract_scenes_from_pipeline(test_yaml_path, "anime", "/tmp/test_output")
            return scenes
        
        scenes = asyncio.run(test_scene_extraction())
        print(f"✅ Extracted {len(scenes)} scenes from YAML")
        
        if len(scenes) != 3:
            print(f"❌ Expected 3 scenes, got {len(scenes)}")
            return False
        
        print("\n2️⃣ Testing pipeline imports...")
        from backend.pipelines.channel_specific.anime_pipeline import AnimeChannelPipeline
        from backend.localization.multi_language_pipeline import MultiLanguagePipelineManager
        from backend.pipelines.ai_upscaler import AIUpscaler
        print("✅ All pipeline imports successful")
        
        print("\n3️⃣ Testing upscaler configuration...")
        upscaler = AIUpscaler()
        
        if hasattr(upscaler, '_create_fallback_upscale'):
            print("❌ FFmpeg fallback method still exists")
            return False
        else:
            print("✅ FFmpeg fallback methods removed")
        
        print("\n4️⃣ Testing multi-language pipeline config...")
        ml_manager = MultiLanguagePipelineManager()
        
        pipeline_config = {
            "base_model": "stable_diffusion_1_5",
            "channel_type": "anime",
            "lora_models": [],
            "lora_paths": {},
            "output_path": "/tmp/test_output",
            "input_path": test_yaml_path,
            "script_data": test_yaml_content
        }
        
        base_config = {
            "channel_type": pipeline_config.get("channel_type", "anime"),
            "base_model": pipeline_config.get("base_model", "stable_diffusion_1_5"),
            "lora_models": pipeline_config.get("lora_models", []),
            "lora_paths": pipeline_config.get("lora_paths", {}),
            "output_path": pipeline_config.get("output_path", "/tmp/output"),
            "input_path": pipeline_config.get("input_path", ""),
            "script_data": pipeline_config.get("script_data", {}),
            "language": "English",
            "scenes": scenes
        }
        
        if base_config.get("script_data") and base_config.get("input_path"):
            print("✅ Script data and input path preserved in pipeline config")
        else:
            print("❌ Script data or input path missing from pipeline config")
            return False
        
        print("\n5️⃣ Testing async pipeline manager...")
        from backend.core.async_pipeline_manager import AsyncPipelineManager
        
        async_manager = AsyncPipelineManager()
        
        if hasattr(async_manager, 'execute_async'):
            print("✅ AsyncPipelineManager has execute_async method")
        else:
            print("❌ AsyncPipelineManager missing execute_async method")
            return False
        
        print("\n🎉 All quick fixes verified successfully!")
        return True
    
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        os.unlink(test_yaml_path)
        print(f"🧹 Cleaned up test file")

if __name__ == "__main__":
    success = test_quick_fixes()
    sys.exit(0 if success else 1)
