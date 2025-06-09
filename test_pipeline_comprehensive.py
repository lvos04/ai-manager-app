#!/usr/bin/env python3

import tempfile
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def test_complete_pipeline():
    """Test complete pipeline functionality with all fixes applied."""
    print("🎬 Testing Complete AI Pipeline Functionality")
    print("=" * 60)
    
    try:
        from backend.pipelines.channel_specific.anime_pipeline import AnimeChannelPipeline
        print("✅ Successfully imported AnimeChannelPipeline")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            pipeline = AnimeChannelPipeline()
            
            test_script = {
                'scenes': [
                    {
                        'text': 'Hero discovers ancient power in mystical temple',
                        'characters': ['Hero', 'Guardian'],
                        'duration': 12,
                        'dialogue': 'Hero: What is this power? Guardian: The ancient magic awakens!'
                    },
                    {
                        'text': 'Epic battle against dark forces with new abilities',
                        'characters': ['Hero', 'Dark Lord'],
                        'duration': 15,
                        'dialogue': 'Dark Lord: You cannot defeat me! Hero: Watch me try!'
                    }
                ]
            }
            
            print(f"📁 Testing pipeline with output directory: {output_dir}")
            print(f"📝 Test script has {len(test_script['scenes'])} scenes")
            
            try:
                script_file = output_dir / "test_script.yaml"
                import yaml
                with open(script_file, 'w') as f:
                    yaml.dump(test_script, f)
                
                result = pipeline._run_pipeline(
                    input_path=str(script_file),
                    output_path=str(output_dir),
                    language='en',
                    render_fps=24,
                    output_fps=24
                )
                
                print(f"✅ Pipeline execution completed")
                print(f"📊 Result: {result}")
                
                scenes_dir = output_dir / "scenes"
                if scenes_dir.exists():
                    scene_files = list(scenes_dir.glob("*.mp4"))
                    print(f"🎥 Generated {len(scene_files)} scene files")
                    for scene_file in scene_files:
                        size = scene_file.stat().st_size if scene_file.exists() else 0
                        print(f"   - {scene_file.name}: {size} bytes")
                
                final_dir = output_dir / "final"
                if final_dir.exists():
                    final_files = list(final_dir.glob("*.mp4"))
                    print(f"🎬 Generated {len(final_files)} final video files")
                    for final_file in final_files:
                        size = final_file.stat().st_size if final_file.exists() else 0
                        print(f"   - {final_file.name}: {size} bytes")
                
                shorts_dir = output_dir / "shorts"
                if shorts_dir.exists():
                    shorts_files = list(shorts_dir.glob("*.mp4"))
                    print(f"📱 Generated {len(shorts_files)} shorts files")
                    for short_file in shorts_files:
                        size = short_file.stat().st_size if short_file.exists() else 0
                        print(f"   - {short_file.name}: {size} bytes")
                
                metadata_files = list(output_dir.glob("*.txt"))
                print(f"📄 Generated {len(metadata_files)} metadata files")
                for metadata_file in metadata_files:
                    print(f"   - {metadata_file.name}")
                    if metadata_file.stat().st_size > 0:
                        content = metadata_file.read_text()[:100]
                        print(f"     Content preview: {content}...")
                
                voice_files = list(output_dir.glob("**/*.wav"))
                print(f"🎤 Generated {len(voice_files)} voice files")
                
                return True
                
            except Exception as e:
                print(f"❌ Pipeline execution failed: {e}")
                import traceback
                traceback.print_exc()
                return False
                
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_youtube_metadata_generation():
    """Test YouTube metadata generation specifically."""
    print("\n📄 Testing YouTube Metadata Generation")
    print("-" * 40)
    
    try:
        from backend.pipelines.channel_specific.base_pipeline import BasePipeline
        
        pipeline = BasePipeline(channel_type="anime")
        
        test_scenes = [
            {
                'text': 'Hero discovers ancient power',
                'characters': ['Hero', 'Guardian'],
                'dialogue': 'Hero: What is this power?'
            },
            {
                'text': 'Epic battle with dark forces',
                'characters': ['Hero', 'Dark Lord'],
                'dialogue': 'Dark Lord: You cannot defeat me!'
            }
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            metadata = pipeline.generate_youtube_metadata(
                scenes=test_scenes,
                language='en',
                output_dir=output_dir
            )
            
            print(f"✅ Metadata generation completed")
            print(f"📝 Title: {metadata.get('title', 'N/A')}")
            print(f"📝 Description length: {len(metadata.get('description', ''))}")
            print(f"📝 Tags: {metadata.get('tags', 'N/A')}")
            
            title_file = output_dir / "title.txt"
            desc_file = output_dir / "description.txt"
            
            if title_file.exists():
                title_content = title_file.read_text().strip()
                print(f"📄 Title file: {title_content}")
            
            if desc_file.exists():
                desc_content = desc_file.read_text().strip()
                print(f"📄 Description file length: {len(desc_content)} chars")
                print(f"📄 Description preview: {desc_content[:100]}...")
            
            return True
            
    except Exception as e:
        print(f"❌ Metadata test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 AI Project Manager - Comprehensive Pipeline Test")
    print("=" * 60)
    
    metadata_success = test_youtube_metadata_generation()
    
    pipeline_success = test_complete_pipeline()
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print(f"   YouTube Metadata: {'✅ PASS' if metadata_success else '❌ FAIL'}")
    print(f"   Complete Pipeline: {'✅ PASS' if pipeline_success else '❌ FAIL'}")
    
    if metadata_success and pipeline_success:
        print("\n🎉 All tests passed! Pipeline fixes are working correctly.")
        exit(0)
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        exit(1)
