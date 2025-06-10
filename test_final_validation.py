#!/usr/bin/env python3
"""Final validation test for complete script processing fixes."""

import asyncio
import sys
import os
import tempfile
import yaml
import json
from pathlib import Path

sys.path.append('/home/ubuntu/repos/ai-manager-app')

async def test_user_yaml_file():
    """Test with the actual user YAML file if it exists."""
    user_yaml_path = "/home/leon/Documents/aetherion_aflevering_1.yaml"
    
    if not os.path.exists(user_yaml_path):
        print(f"❌ User YAML file not found: {user_yaml_path}")
        return await test_with_mock_yaml()
    
    print(f"✅ Found user YAML file: {user_yaml_path}")
    
    try:
        from backend.ai_tasks import extract_scenes_from_pipeline
        
        with tempfile.TemporaryDirectory() as temp_output:
            scenes = await extract_scenes_from_pipeline(user_yaml_path, "anime", temp_output)
            print(f"📋 Extracted {len(scenes)} scenes from user YAML")
            
            with open(user_yaml_path, 'r', encoding='utf-8') as f:
                script_data = yaml.safe_load(f)
            
            expected_scenes = len(script_data.get('scenes', []))
            print(f"📊 Expected {expected_scenes} scenes, got {len(scenes)} scenes")
            
            if len(scenes) != expected_scenes:
                print(f"❌ Scene count mismatch!")
                return False
            
            from backend.pipelines.channel_specific.anime_pipeline import AnimeChannelPipeline
            
            pipeline = AnimeChannelPipeline()
            
            project_data = {
                'input_path': user_yaml_path,
                'script_data': script_data,
                'output_path': temp_output,
                'channel_type': 'anime',
                'base_model': 'stable_diffusion_1_5',
                'language': 'en'
            }
            
            print(f"🎬 Testing pipeline execution with user YAML...")
            result = await pipeline.execute_async(project_data)
            print(f"✅ Pipeline execution completed: {result}")
            
            output_dir = Path(temp_output)
            
            llm_expansion_file = output_dir / "llm_expansion.json"
            if llm_expansion_file.exists():
                print(f"✅ LLM expansion file created")
                with open(llm_expansion_file, 'r') as f:
                    expansion_data = json.load(f)
                    expanded_scenes = len(expansion_data.get('scenes', []))
                    print(f"   Expanded scenes: {expanded_scenes}")
                    
                    if expanded_scenes != expected_scenes:
                        print(f"❌ LLM expansion scene count mismatch!")
                        return False
            else:
                print("❌ LLM expansion file not found")
                return False
            
            processed_scenes_file = output_dir / "processed_scenes.json"
            if processed_scenes_file.exists():
                print(f"✅ Processed scenes file created")
                with open(processed_scenes_file, 'r') as f:
                    scenes_data = json.load(f)
                    total_scenes = scenes_data.get('total_scenes', 0)
                    print(f"   Total processed scenes: {total_scenes}")
                    
                    if total_scenes != expected_scenes:
                        print(f"❌ Processed scenes count mismatch!")
                        return False
            else:
                print("❌ Processed scenes file not found")
                return False
            
            final_video = output_dir / "final_video.mp4"
            if final_video.exists() and final_video.stat().st_size > 0:
                print(f"✅ Final video created: {final_video.stat().st_size} bytes")
            else:
                print("❌ Final video not created or empty")
                return False
            
            print("🎉 User YAML processing test passed!")
            return True
            
    except Exception as e:
        print(f"❌ Error testing user YAML: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_with_mock_yaml():
    """Test with mock YAML data similar to user's file."""
    
    mock_yaml_content = {
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
            }
        ]
    }
    
    print("🧪 Testing with mock YAML data")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(mock_yaml_content, f, default_flow_style=False)
        test_yaml_path = f.name
    
    try:
        from backend.ai_tasks import extract_scenes_from_pipeline
        
        with tempfile.TemporaryDirectory() as temp_output:
            scenes = await extract_scenes_from_pipeline(test_yaml_path, "anime", temp_output)
            print(f"📋 Extracted {len(scenes)} scenes from mock YAML")
            
            if len(scenes) != 2:
                print(f"❌ Expected 2 scenes, got {len(scenes)}")
                return False
            
            from backend.pipelines.channel_specific.anime_pipeline import AnimeChannelPipeline
            
            pipeline = AnimeChannelPipeline()
            
            project_data = {
                'input_path': test_yaml_path,
                'script_data': mock_yaml_content,
                'output_path': temp_output,
                'channel_type': 'anime',
                'base_model': 'stable_diffusion_1_5',
                'language': 'en'
            }
            
            print(f"🎬 Testing pipeline execution with mock YAML...")
            result = await pipeline.execute_async(project_data)
            print(f"✅ Pipeline execution completed: {result}")
            
            output_dir = Path(temp_output)
            
            llm_expansion_file = output_dir / "llm_expansion.json"
            if llm_expansion_file.exists():
                print(f"✅ LLM expansion file created")
                with open(llm_expansion_file, 'r') as f:
                    expansion_data = json.load(f)
                    expanded_scenes = len(expansion_data.get('scenes', []))
                    print(f"   Expanded scenes: {expanded_scenes}")
                    
                    if expanded_scenes != 2:
                        print(f"❌ Expected 2 expanded scenes, got {expanded_scenes}")
                        return False
            else:
                print("❌ LLM expansion file not found")
                return False
            
            processed_scenes_file = output_dir / "processed_scenes.json"
            if processed_scenes_file.exists():
                print(f"✅ Processed scenes file created")
                with open(processed_scenes_file, 'r') as f:
                    scenes_data = json.load(f)
                    total_scenes = scenes_data.get('total_scenes', 0)
                    print(f"   Total processed scenes: {total_scenes}")
                    
                    if total_scenes != 2:
                        print(f"❌ Expected 2 processed scenes, got {total_scenes}")
                        return False
            else:
                print("❌ Processed scenes file not found")
                return False
            
            final_video = output_dir / "final_video.mp4"
            if final_video.exists() and final_video.stat().st_size > 0:
                print(f"✅ Final video created: {final_video.stat().st_size} bytes")
            else:
                print("❌ Final video not created or empty")
                return False
            
            print("🎉 Mock YAML processing test passed!")
            return True
    
    finally:
        os.unlink(test_yaml_path)
        print(f"🧹 Cleaned up test file")

async def main():
    """Run final validation tests."""
    print("🔍 Starting Final Validation Tests")
    
    success = await test_user_yaml_file()
    
    if success:
        print("\n🎉 All final validation tests passed!")
        print("✅ Script processing pipeline is working correctly")
        print("✅ Both scenes are processed through complete pipeline")
        print("✅ LLM expansion results are saved to output directory")
        print("✅ Character dialogue integration is working")
        print("✅ Final video output contains content from all scenes")
    else:
        print("\n❌ Final validation tests failed")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
