#!/usr/bin/env python3
"""Test complete fixes with user's YAML file using direct pipeline execution."""

import asyncio
import sys
import os
import tempfile
import yaml
import json
from pathlib import Path

sys.path.append('/home/ubuntu/repos/ai-manager-app')

async def test_complete_fixes():
    """Test all fixes with user's YAML file."""
    
    test_yaml_content = {
        "title": "Aetherion Episode 1",
        "characters": [
            {"name": "Kael", "description": "Main protagonist"},
            {"name": "Note", "description": "Supporting character"}
        ],
        "locations": [
            {"name": "Abyss", "description": "Dark mysterious realm"}
        ],
        "scenes": [
            {
                "description": "Kael explores the mysterious abyss",
                "character": "Kael",
                "dialogue": "What is this strange place?",
                "location": "Abyss",
                "duration": 15.0
            },
            {
                "description": "Note appears to guide Kael",
                "character": "Note", 
                "dialogue": "Follow me, I know the way",
                "location": "Abyss",
                "duration": 12.0
            },
            {
                "description": "They discover ancient ruins",
                "character": "Kael",
                "dialogue": "These ruins... they're older than anything I've seen",
                "location": "Ancient Ruins",
                "duration": 18.0
            }
        ]
    }
    
    print("🧪 Testing Complete Pipeline Fixes")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_yaml_content, f, default_flow_style=False)
        test_yaml_path = f.name
    
    try:
        from backend.pipelines.channel_specific.anime_pipeline import AnimeChannelPipeline
        from backend.localization.multi_language_pipeline import MultiLanguagePipelineManager
        
        with tempfile.TemporaryDirectory() as temp_output:
            print(f"📄 Testing with YAML: {test_yaml_path}")
            print(f"📁 Output directory: {temp_output}")
            
            pipeline = AnimeChannelPipeline()
            
            result = await pipeline.run_with_script_data(
                input_path=test_yaml_path,
                output_path=temp_output,
                script_data=test_yaml_content
            )
            
            print(f"✅ Pipeline execution completed: {result}")
            
            output_dir = Path(temp_output)
            
            llm_expansion_file = output_dir / "llm_expansion.json"
            if llm_expansion_file.exists():
                with open(llm_expansion_file, 'r') as f:
                    expansion_data = json.load(f)
                    scenes_count = len(expansion_data.get('scenes', []))
                    print(f"✅ LLM expansion: {scenes_count} scenes processed")
                    if scenes_count != 3:
                        print(f"❌ Expected 3 scenes, got {scenes_count}")
                        return False
            else:
                print("❌ LLM expansion file not found")
                return False
            
            final_video_candidates = [
                output_dir / "final_video.mp4",
                output_dir / "final" / "anime_episode.mp4",
                output_dir / "anime_episode.mp4"
            ]
            
            final_video = None
            for candidate in final_video_candidates:
                if candidate.exists():
                    final_video = candidate
                    break
            
            if final_video:
                file_size = final_video.stat().st_size
                print(f"✅ Final video created: {final_video} ({file_size} bytes)")
                if file_size < 1000:
                    print(f"❌ Video file too small: {file_size} bytes")
                    return False
            else:
                print("❌ Final video not created")
                return False
            
            shorts_dir = output_dir / "shorts"
            if shorts_dir.exists():
                shorts_files = list(shorts_dir.glob("*.mp4"))
                print(f"✅ Shorts created: {len(shorts_files)} files")
                
                for short_file in shorts_files:
                    if short_file.stat().st_size > 1000:
                        print(f"✅ Short has actual content: {short_file} ({short_file.stat().st_size} bytes)")
                    else:
                        print(f"⚠️ Short file too small: {short_file} ({short_file.stat().st_size} bytes)")
            else:
                print("⚠️ No shorts directory found")
            
            print("\n🌐 Testing Multi-Language Pipeline")
            ml_manager = MultiLanguagePipelineManager()
            
            pipeline_config = {
                "base_model": "stable_diffusion_1_5",
                "channel_type": "anime",
                "lora_models": [],
                "lora_paths": {},
                "output_path": temp_output,
                "input_path": test_yaml_path,
                "script_data": test_yaml_content
            }
            
            ml_result = await ml_manager.execute_multi_language_pipeline(
                scenes=[
                    {"description": "Test scene", "dialogue": "Test dialogue", "duration": 10.0}
                ],
                languages=["English"],
                pipeline_config=pipeline_config
            )
            
            print(f"✅ Multi-language pipeline completed: {ml_result}")
            
            print("🎉 All fixes verified successfully!")
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
    success = asyncio.run(test_complete_fixes())
    sys.exit(0 if success else 1)
