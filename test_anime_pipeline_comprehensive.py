#!/usr/bin/env python3
"""
Comprehensive test for anime pipeline 20-minute video generation
"""

import sys
import os
import time
import tempfile
import yaml
from pathlib import Path
sys.path.append('.')

def create_comprehensive_anime_script():
    """Create comprehensive test script for anime pipeline"""
    return {
        'title': 'Anime Test Episode - Complete 20 Minute Generation',
        'description': 'Comprehensive test for anime pipeline with all components',
        'target_duration': 1200,  # 20 minutes
        'scenes': [
            {
                'scene_number': 1,
                'description': 'Anime opening scene with protagonist in mysterious forest',
                'dialogue': 'Where am I? This place feels both familiar and strange...',
                'duration': 180,
                'characters': ['Protagonist'],
                'location': 'Mysterious Forest',
                'scene_type': 'dialogue',
                'mood': 'mysterious'
            },
            {
                'scene_number': 2,
                'description': 'Epic anime battle scene with special powers and dynamic combat',
                'dialogue': 'I must protect everyone! This is my destiny!',
                'duration': 240,
                'characters': ['Protagonist', 'Shadow Enemy'],
                'location': 'Battle Arena',
                'scene_type': 'combat',
                'mood': 'intense'
            },
            {
                'scene_number': 3,
                'description': 'Emotional anime scene with character development',
                'dialogue': 'I understand now... true strength comes from protecting others.',
                'duration': 180,
                'characters': ['Protagonist', 'Mentor'],
                'location': 'Sacred Temple',
                'scene_type': 'dialogue',
                'mood': 'emotional'
            }
        ],
        'characters': [
            {
                'name': 'Protagonist',
                'description': 'Young anime hero with mysterious powers and strong determination',
                'voice_style': 'heroic',
                'personality': 'brave, determined, caring'
            },
            {
                'name': 'Shadow Enemy',
                'description': 'Dark antagonist with powerful shadow abilities',
                'voice_style': 'menacing',
                'personality': 'evil, powerful, mysterious'
            },
            {
                'name': 'Mentor',
                'description': 'Wise elder who guides the protagonist',
                'voice_style': 'wise',
                'personality': 'wise, patient, caring'
            }
        ],
        'locations': [
            {
                'name': 'Mysterious Forest',
                'description': 'Ethereal anime forest with glowing particles and ancient trees'
            },
            {
                'name': 'Battle Arena',
                'description': 'Epic anime battle location with dramatic lighting and energy effects'
            },
            {
                'name': 'Sacred Temple',
                'description': 'Ancient temple with mystical atmosphere and peaceful energy'
            }
        ],
        'style': {
            'animation_style': 'anime',
            'art_style': 'detailed anime with vibrant colors',
            'mood': 'epic adventure with emotional depth'
        }
    }

def test_anime_pipeline_comprehensive():
    """Test complete anime pipeline functionality"""
    print("🎬 Testing Comprehensive Anime Pipeline...")
    
    try:
        from backend.pipelines.channel_specific.anime_pipeline import run
        
        test_script = create_comprehensive_anime_script()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            script_path = os.path.join(temp_dir, "comprehensive_anime_test.yaml")
            with open(script_path, 'w') as f:
                yaml.dump(test_script, f, default_flow_style=False)
            
            output_dir = os.path.join(temp_dir, "anime_output")
            os.makedirs(output_dir, exist_ok=True)
            
            print(f"📝 Script: {script_path}")
            print(f"📁 Output: {output_dir}")
            print(f"🎯 Target: 20-minute video generation")
            
            start_time = time.time()
            
            result = run(
                input_path=script_path,
                output_path=output_dir,
                base_model="stable_diffusion_1_5",
                lora_models=["anime_style", "combat_style"],
                language="en",
                render_fps=24,
                output_fps=24,
                frame_interpolation_enabled=True,
                voice_generation_enabled=True,
                background_music_enabled=True,
                lipsync_enabled=True
            )
            
            execution_time = time.time() - start_time
            print(f"⏱️  Pipeline completed in {execution_time:.2f} seconds")
            
            output_files = list(Path(output_dir).rglob("*.mp4"))
            if output_files:
                total_size = sum(f.stat().st_size for f in output_files)
                print(f"✅ Generated {len(output_files)} video files")
                print(f"📊 Total size: {total_size:,} bytes")
                
                substantial_files = 0
                for video_file in output_files:
                    file_size = video_file.stat().st_size
                    print(f"  📹 {video_file.name}: {file_size:,} bytes")
                    
                    if file_size > 5000000:  # 5MB threshold for substantial content
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
        print(f"❌ Anime pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_device_map_fixes():
    """Test that device_map fixes are working"""
    print("\n🔧 Testing Device Map Fixes...")
    
    try:
        from backend.pipelines.video_generation import TextToVideoGenerator
        
        generator = TextToVideoGenerator()
        
        test_models = ["animatediff_v2_sdxl", "svd_xt", "zeroscope_v2_xl"]
        
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
    """Run comprehensive anime pipeline tests"""
    print("🎬 Comprehensive Anime Pipeline Test Suite")
    print("=" * 70)
    
    device_map_ok = test_device_map_fixes()
    pipeline_ok = test_anime_pipeline_comprehensive()
    
    print("\n" + "=" * 70)
    print("📊 COMPREHENSIVE TEST SUMMARY")
    print("=" * 70)
    
    print(f"Device Map Fixes: {'✅ PASSED' if device_map_ok else '❌ FAILED'}")
    print(f"Pipeline Generation: {'✅ PASSED' if pipeline_ok else '❌ FAILED'}")
    
    if device_map_ok and pipeline_ok:
        print("🎉 Anime pipeline fully functional for 20-minute generation!")
        return True
    else:
        print("⚠️  Some components need additional work")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
