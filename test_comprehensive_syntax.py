#!/usr/bin/env python3
"""Comprehensive syntax and pipeline testing."""

import ast
import sys
import os
import asyncio
import tempfile
import yaml
import json
from pathlib import Path

sys.path.append('/home/ubuntu/repos/ai-manager-app')

def validate_python_syntax(file_path):
    """Validate Python syntax for a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        ast.parse(source, filename=str(file_path))
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error in {file_path}: {e}"
    except Exception as e:
        return False, f"Error reading {file_path}: {e}"

def validate_all_python_files():
    """Validate syntax for all Python files in the project."""
    project_root = Path('/home/ubuntu/repos/ai-manager-app')
    python_files = list(project_root.rglob('*.py'))
    
    errors = []
    valid_count = 0
    
    for py_file in python_files:
        if 'venv' in str(py_file) or '__pycache__' in str(py_file):
            continue
            
        is_valid, error = validate_python_syntax(py_file)
        if is_valid:
            valid_count += 1
            print(f"✅ {py_file.relative_to(project_root)}")
        else:
            errors.append(error)
            print(f"❌ {error}")
    
    print(f"\nSyntax Validation Summary:")
    print(f"Valid files: {valid_count}")
    print(f"Files with errors: {len(errors)}")
    
    if errors:
        print("\nErrors found:")
        for error in errors:
            print(f"  {error}")
        return False
    else:
        print("✅ All Python files have valid syntax!")
        return True

async def test_complete_pipeline():
    """Test complete pipeline with YAML processing."""
    
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
            }
        ]
    }
    
    print("\n🧪 Testing Complete Pipeline Integration")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_yaml_content, f, default_flow_style=False)
        test_yaml_path = f.name
    
    try:
        print(f"📄 Created test YAML: {test_yaml_path}")
        
        from backend.ai_tasks import extract_scenes_from_pipeline
        
        scenes = await extract_scenes_from_pipeline(test_yaml_path, "anime", temp_output)
        print(f"📋 Extracted {len(scenes)} scenes from YAML")
        
        if len(scenes) != 2:
            print(f"❌ Expected 2 scenes, got {len(scenes)}")
            return False
        
        for i, scene in enumerate(scenes):
            print(f"   Scene {i+1}: {scene}")
        
        with open(test_yaml_path, 'r', encoding='utf-8') as f:
            script_data = yaml.safe_load(f)
        
        print(f"👥 Characters: {[c['name'] for c in script_data.get('characters', [])]}")
        print(f"🏞️  Locations: {[l['name'] for l in script_data.get('locations', [])]}")
        
        from backend.pipelines.channel_specific.anime_pipeline import AnimeChannelPipeline
        
        pipeline = AnimeChannelPipeline()
        
        with tempfile.TemporaryDirectory() as temp_output:
            print(f"🎬 Testing pipeline execution...")
            
            project_data = {
                'input_path': test_yaml_path,
                'script_data': script_data,
                'output_path': temp_output,
                'channel_type': 'anime',
                'base_model': 'stable_diffusion_1_5',
                'language': 'en'
            }
            
            result = await pipeline.execute_async(project_data)
            print(f"✅ Pipeline execution completed: {result}")
            
            output_dir = Path(temp_output)
            
            llm_expansion_file = output_dir / "llm_expansion.json"
            if llm_expansion_file.exists():
                print(f"✅ LLM expansion file created")
                with open(llm_expansion_file, 'r') as f:
                    expansion_data = json.load(f)
                    scenes_count = len(expansion_data.get('scenes', []))
                    print(f"   Expanded scenes: {scenes_count}")
                    if scenes_count != 2:
                        print(f"❌ Expected 2 expanded scenes, got {scenes_count}")
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
            
            print("🎉 All pipeline tests passed!")
            return True
    
    finally:
        os.unlink(test_yaml_path)
        print(f"🧹 Cleaned up test file")

async def main():
    """Run comprehensive testing."""
    print("🔍 Starting Comprehensive Testing Suite")
    
    print("\n1. Validating Python syntax...")
    syntax_valid = validate_all_python_files()
    
    if not syntax_valid:
        print("❌ Syntax validation failed. Fix syntax errors before proceeding.")
        return False
    
    print("\n2. Testing complete pipeline...")
    pipeline_valid = await test_complete_pipeline()
    
    if not pipeline_valid:
        print("❌ Pipeline testing failed.")
        return False
    
    print("\n🎉 All tests passed successfully!")
    return True

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
