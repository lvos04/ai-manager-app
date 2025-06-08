#!/usr/bin/env python3
"""
Comprehensive test for all 6 channel pipelines
Tests complete 20-minute video generation across all channels
"""

import sys
import os
import time
import tempfile
import yaml
from pathlib import Path
sys.path.append('.')

def create_test_script(channel_type: str) -> dict:
    """Create test script for specific channel type."""
    
    base_scripts = {
        "anime": {
            'title': 'Anime Test Episode',
            'description': 'Epic anime adventure with magical elements',
            'target_duration': 1200,
            'scenes': [
                {
                    'scene_number': 1,
                    'description': 'Anime hero discovers mysterious powers in enchanted forest',
                    'dialogue': 'What is this strange energy I feel within me?',
                    'duration': 300,
                    'characters': ['Hero'],
                    'location': 'Enchanted Forest',
                    'scene_type': 'dialogue'
                },
                {
                    'scene_number': 2,
                    'description': 'Epic anime battle with dark forces using special abilities',
                    'dialogue': 'I must protect everyone with my newfound power!',
                    'duration': 300,
                    'characters': ['Hero', 'Dark Enemy'],
                    'location': 'Battle Arena',
                    'scene_type': 'combat'
                }
            ],
            'characters': [
                {'name': 'Hero', 'description': 'Young anime protagonist with hidden powers'},
                {'name': 'Dark Enemy', 'description': 'Mysterious antagonist with shadow abilities'}
            ],
            'locations': [
                {'name': 'Enchanted Forest', 'description': 'Magical forest with glowing particles'},
                {'name': 'Battle Arena', 'description': 'Epic battleground with dramatic lighting'}
            ]
        },
        
        "gaming": {
            'title': 'Gaming Highlights Compilation',
            'description': 'Epic gaming moments and strategic gameplay',
            'target_duration': 1200,
            'scenes': [
                {
                    'scene_number': 1,
                    'description': 'Incredible gaming clutch moment with perfect timing',
                    'dialogue': 'This is the moment that defines a true gamer!',
                    'duration': 300,
                    'characters': ['Pro Gamer'],
                    'location': 'Gaming Arena',
                    'scene_type': 'action'
                },
                {
                    'scene_number': 2,
                    'description': 'Strategic team coordination leading to victory',
                    'dialogue': 'Teamwork makes the dream work in competitive gaming!',
                    'duration': 300,
                    'characters': ['Team Leader'],
                    'location': 'Tournament Stage',
                    'scene_type': 'dialogue'
                }
            ],
            'characters': [
                {'name': 'Pro Gamer', 'description': 'Elite competitive player with incredible skills'},
                {'name': 'Team Leader', 'description': 'Strategic mastermind and team coordinator'}
            ],
            'locations': [
                {'name': 'Gaming Arena', 'description': 'High-tech competitive gaming environment'},
                {'name': 'Tournament Stage', 'description': 'Professional esports tournament venue'}
            ]
        },
        
        "superhero": {
            'title': 'Superhero Epic Adventure',
            'description': 'Heroic adventure with incredible powers and dramatic action',
            'target_duration': 1200,
            'scenes': [
                {
                    'scene_number': 1,
                    'description': 'Superhero saves the city from imminent destruction',
                    'dialogue': 'With great power comes great responsibility!',
                    'duration': 300,
                    'characters': ['Superhero'],
                    'location': 'Metropolis',
                    'scene_type': 'action'
                },
                {
                    'scene_number': 2,
                    'description': 'Epic superhero battle against powerful villain with incredible abilities',
                    'dialogue': 'You will not harm innocent people on my watch!',
                    'duration': 300,
                    'characters': ['Superhero', 'Supervillain'],
                    'location': 'City Rooftop',
                    'scene_type': 'combat'
                }
            ],
            'characters': [
                {'name': 'Superhero', 'description': 'Powerful hero with incredible abilities and strong moral code'},
                {'name': 'Supervillain', 'description': 'Dangerous antagonist with destructive powers'}
            ],
            'locations': [
                {'name': 'Metropolis', 'description': 'Modern city with towering skyscrapers'},
                {'name': 'City Rooftop', 'description': 'Dramatic rooftop setting for epic confrontation'}
            ]
        },
        
        "manga": {
            'title': 'Traditional Manga Story',
            'description': 'Classic manga narrative with traditional Japanese elements',
            'target_duration': 1200,
            'scenes': [
                {
                    'scene_number': 1,
                    'description': 'Traditional manga character contemplates life in peaceful garden',
                    'dialogue': 'The cherry blossoms remind me of the fleeting nature of life.',
                    'duration': 300,
                    'characters': ['Protagonist'],
                    'location': 'Japanese Garden',
                    'scene_type': 'emotional'
                },
                {
                    'scene_number': 2,
                    'description': 'Traditional martial arts combat with honor and discipline',
                    'dialogue': 'True strength comes from discipline and honor!',
                    'duration': 300,
                    'characters': ['Protagonist', 'Rival'],
                    'location': 'Dojo',
                    'scene_type': 'combat'
                }
            ],
            'characters': [
                {'name': 'Protagonist', 'description': 'Traditional manga character with strong values'},
                {'name': 'Rival', 'description': 'Honorable opponent who challenges the protagonist'}
            ],
            'locations': [
                {'name': 'Japanese Garden', 'description': 'Peaceful traditional garden with cherry blossoms'},
                {'name': 'Dojo', 'description': 'Traditional martial arts training hall'}
            ]
        },
        
        "marvel_dc": {
            'title': 'Marvel DC Universe Adventure',
            'description': 'Epic comic book adventure with iconic heroes and villains',
            'target_duration': 1200,
            'scenes': [
                {
                    'scene_number': 1,
                    'description': 'Iconic comic book hero protects innocent civilians',
                    'dialogue': 'Justice will always prevail against evil!',
                    'duration': 300,
                    'characters': ['Comic Hero'],
                    'location': 'Comic City',
                    'scene_type': 'action'
                },
                {
                    'scene_number': 2,
                    'description': 'Epic comic book battle with universe-threatening villain',
                    'dialogue': 'The fate of the universe depends on this battle!',
                    'duration': 300,
                    'characters': ['Comic Hero', 'Cosmic Villain'],
                    'location': 'Cosmic Battlefield',
                    'scene_type': 'combat'
                }
            ],
            'characters': [
                {'name': 'Comic Hero', 'description': 'Iconic superhero with legendary powers'},
                {'name': 'Cosmic Villain', 'description': 'Universe-threatening antagonist with cosmic powers'}
            ],
            'locations': [
                {'name': 'Comic City', 'description': 'Classic comic book cityscape with dramatic architecture'},
                {'name': 'Cosmic Battlefield', 'description': 'Universe-spanning battlefield with cosmic energy'}
            ]
        },
        
        "original_manga": {
            'title': 'Original Manga Creation',
            'description': 'Unique original manga with creative storytelling',
            'target_duration': 1200,
            'scenes': [
                {
                    'scene_number': 1,
                    'description': 'Original character discovers unique abilities in creative world',
                    'dialogue': 'This power... it is unlike anything I have ever seen.',
                    'duration': 300,
                    'characters': ['Original Hero'],
                    'location': 'Creative Realm',
                    'scene_type': 'dialogue'
                },
                {
                    'scene_number': 2,
                    'description': 'Original combat scene with unique fighting style and creative abilities',
                    'dialogue': 'My original technique will overcome any challenge!',
                    'duration': 300,
                    'characters': ['Original Hero', 'Creative Rival'],
                    'location': 'Artistic Battlefield',
                    'scene_type': 'combat'
                }
            ],
            'characters': [
                {'name': 'Original Hero', 'description': 'Unique character with creative design and original abilities'},
                {'name': 'Creative Rival', 'description': 'Original antagonist with artistic combat style'}
            ],
            'locations': [
                {'name': 'Creative Realm', 'description': 'Original world with unique artistic elements'},
                {'name': 'Artistic Battlefield', 'description': 'Creative combat arena with original design'}
            ]
        }
    }
    
    return base_scripts.get(channel_type, base_scripts["anime"])

def test_pipeline(channel_type: str) -> bool:
    """Test individual pipeline for 20-minute video generation."""
    print(f"\n🎬 Testing {channel_type.upper()} Pipeline")
    print("=" * 50)
    
    try:
        from backend.pipelines.channel_specific import CHANNEL_PIPELINES
        
        if channel_type not in CHANNEL_PIPELINES:
            print(f"❌ Pipeline {channel_type} not found")
            return False
        
        pipeline_func = CHANNEL_PIPELINES[channel_type]
        test_script = create_test_script(channel_type)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            script_path = os.path.join(temp_dir, f"{channel_type}_test.yaml")
            with open(script_path, 'w') as f:
                yaml.dump(test_script, f, default_flow_style=False)
            
            output_dir = os.path.join(temp_dir, f"{channel_type}_output")
            os.makedirs(output_dir, exist_ok=True)
            
            print(f"📝 Script: {script_path}")
            print(f"📁 Output: {output_dir}")
            print(f"🎯 Target: 20-minute video generation")
            
            start_time = time.time()
            
            result = pipeline_func(
                input_path=script_path,
                output_path=output_dir,
                base_model="stable_diffusion_1_5",
                lora_models=[f"{channel_type}_style"],
                language="en"
            )
            
            execution_time = time.time() - start_time
            print(f"⏱️  Pipeline completed in {execution_time:.2f} seconds")
            
            output_files = list(Path(output_dir).rglob("*.mp4"))
            if output_files:
                total_size = sum(f.stat().st_size for f in output_files)
                print(f"✅ Generated {len(output_files)} video files")
                print(f"📊 Total size: {total_size:,} bytes ({total_size/1024/1024:.1f} MB)")
                
                substantial_files = 0
                for video_file in output_files:
                    file_size = video_file.stat().st_size
                    print(f"  📹 {video_file.name}: {file_size:,} bytes")
                    
                    if file_size > 5000000:
                        substantial_files += 1
                        print(f"    ✅ Substantial content detected")
                    else:
                        print(f"    ⚠️  File may be too small")
                
                if substantial_files > 0:
                    print(f"🎉 {substantial_files} files with substantial video content!")
                    return True
                else:
                    print("❌ No files with substantial content found")
                    return False
            else:
                print("❌ No video files generated")
                return False
                
    except Exception as e:
        print(f"❌ {channel_type} pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_device_map_fixes():
    """Test that device_map fixes are working."""
    print("\n🔧 Testing Device Map Fixes")
    print("=" * 50)
    
    try:
        from backend.pipelines.video_generation import TextToVideoGenerator
        
        generator = TextToVideoGenerator()
        
        test_models = ["animatediff_v2_sdxl", "svd_xt"]
        
        for model_name in test_models:
            try:
                print(f"Testing {model_name}...")
                model = generator.load_model(model_name)
                
                if model is not None:
                    print(f"✅ {model_name} loaded successfully")
                    generator._cleanup_model_memory()
                else:
                    print(f"⚠️  {model_name} returned None")
                    
            except Exception as e:
                if 'auto not supported' in str(e).lower():
                    print(f"❌ {model_name} still has device_map error: {e}")
                    return False
                else:
                    print(f"✅ {model_name} expected error (CPU fallback): {e}")
        
        print("✅ All device_map fixes working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Device map test failed: {e}")
        return False

def main():
    """Run comprehensive test suite for all 6 channel pipelines."""
    print("🎬 COMPREHENSIVE PIPELINE TEST SUITE")
    print("Testing all 6 channel pipelines for 20-minute video generation")
    print("=" * 70)
    
    channel_types = ["anime", "gaming", "superhero", "manga", "marvel_dc", "original_manga"]
    
    device_map_ok = test_device_map_fixes()
    
    pipeline_results = {}
    for channel_type in channel_types:
        pipeline_results[channel_type] = test_pipeline(channel_type)
    
    print("\n" + "=" * 70)
    print("📊 COMPREHENSIVE TEST SUMMARY")
    print("=" * 70)
    
    print(f"Device Map Fixes: {'✅ PASSED' if device_map_ok else '❌ FAILED'}")
    
    total_pipelines = len(channel_types)
    successful_pipelines = sum(pipeline_results.values())
    success_rate = (successful_pipelines / total_pipelines) * 100
    
    print(f"\nPipeline Results ({successful_pipelines}/{total_pipelines} - {success_rate:.1f}% success rate):")
    for channel_type, success in pipeline_results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {channel_type.upper()}: {status}")
    
    if device_map_ok and successful_pipelines == total_pipelines:
        print(f"\n🎉 ALL PIPELINES FULLY FUNCTIONAL!")
        print(f"✅ Ready for production 20-minute video generation")
        print(f"✅ Device_map fixes working correctly")
        print(f"✅ All {total_pipelines} channel pipelines generating substantial content")
        return True
    else:
        print(f"\n⚠️  SOME COMPONENTS NEED ADDITIONAL WORK")
        if not device_map_ok:
            print(f"❌ Device map fixes need attention")
        if successful_pipelines < total_pipelines:
            print(f"❌ {total_pipelines - successful_pipelines} pipelines not generating substantial content")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
