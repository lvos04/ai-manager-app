#!/usr/bin/env python3
"""
Test script specifically for 20-minute anime video generation
Creates a comprehensive test script and verifies end-to-end pipeline functionality
"""

import sys
import os
import time
import tempfile
import yaml
from pathlib import Path

sys.path.append('.')

def create_20_minute_test_script():
    """Create a test script designed to generate 20-minute anime content"""
    test_script = {
        'title': '20-Minute Anime Test Episode',
        'description': 'Comprehensive test episode for 20-minute anime pipeline validation',
        'target_duration': 1200,
        'scenes': [
            {
                'scene_number': 1,
                'description': 'Opening scene: A young warrior awakens in a mystical forest, sunlight filtering through ancient trees. The camera pans across the magical landscape.',
                'dialogue': 'Where am I? This place... it feels different, filled with ancient magic.',
                'duration': 300,
                'characters': ['Protagonist'],
                'location': 'Mystical Forest',
                'scene_type': 'dialogue'
            },
            {
                'scene_number': 2,
                'description': 'The warrior discovers a hidden village under attack by shadow creatures. Villagers flee in terror as dark magic spreads.',
                'dialogue': 'I must help them! These shadow creatures will destroy everything!',
                'duration': 300,
                'characters': ['Protagonist', 'Villagers'],
                'location': 'Village',
                'scene_type': 'action'
            },
            {
                'scene_number': 3,
                'description': 'Epic combat scene: The warrior fights multiple shadow creatures using magical sword techniques. Dynamic battle choreography with special effects.',
                'dialogue': 'Light magic, grant me strength! Shadow Slash technique!',
                'duration': 300,
                'characters': ['Protagonist', 'Shadow Creatures'],
                'location': 'Village Square',
                'scene_type': 'combat'
            },
            {
                'scene_number': 4,
                'description': 'After the battle, the warrior meets the village elder who reveals the prophecy about the legendary Crystal of Light.',
                'dialogue': 'You are the chosen one from the prophecy. Only you can find the Crystal of Light and save our world.',
                'duration': 300,
                'characters': ['Protagonist', 'Village Elder'],
                'location': 'Elder Temple',
                'scene_type': 'dialogue'
            }
        ],
        'characters': [
            {
                'name': 'Protagonist',
                'description': 'Young brave warrior with determination and magical abilities',
                'voice_style': 'heroic',
                'appearance': 'Anime-style warrior with silver hair and blue eyes'
            },
            {
                'name': 'Village Elder',
                'description': 'Wise old sage with knowledge of ancient prophecies',
                'voice_style': 'wise',
                'appearance': 'Elderly man with long white beard and mystical robes'
            },
            {
                'name': 'Villagers',
                'description': 'Peaceful village inhabitants',
                'voice_style': 'neutral',
                'appearance': 'Various anime-style villagers'
            },
            {
                'name': 'Shadow Creatures',
                'description': 'Dark magical beings threatening the world',
                'voice_style': 'menacing',
                'appearance': 'Dark shadowy forms with glowing red eyes'
            }
        ],
        'locations': [
            {
                'name': 'Mystical Forest',
                'description': 'Ancient forest with magical atmosphere, glowing plants, and ethereal lighting'
            },
            {
                'name': 'Village',
                'description': 'Peaceful anime-style village with traditional architecture'
            },
            {
                'name': 'Village Square',
                'description': 'Central area of village with fountain and market stalls'
            },
            {
                'name': 'Elder Temple',
                'description': 'Sacred temple with ancient symbols and mystical artifacts'
            }
        ],
        'style_notes': {
            'animation_style': 'High-quality anime with smooth motion',
            'color_palette': 'Vibrant colors with magical lighting effects',
            'camera_work': 'Dynamic angles and cinematic shots',
            'special_effects': 'Magical particles and energy effects'
        }
    }
    return test_script

def test_20_minute_anime_generation():
    """Test complete 20-minute anime video generation"""
    print("🎬 Testing 20-minute anime video generation...")
    try:
        from backend.pipelines.channel_specific.anime_pipeline import run
        
        test_script = create_20_minute_test_script()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            script_path = os.path.join(temp_dir, "20min_test_script.yaml")
            with open(script_path, 'w') as f:
                yaml.dump(test_script, f, default_flow_style=False)
            
            output_dir = os.path.join(temp_dir, "anime_output")
            os.makedirs(output_dir, exist_ok=True)
            
            print(f"Running 20-minute anime pipeline...")
            print(f"Script: {script_path}")
            print(f"Output: {output_dir}")
            print(f"Target duration: 20 minutes (1200 seconds)")
            
            start_time = time.time()
            
            try:
                result = run(
                    input_path=script_path,
                    output_path=output_dir,
                    base_model="stable_diffusion_1_5",
                    lora_models=["anime_style", "combat_style"],
                    language="en",
                    render_fps=24,
                    output_fps=24
                )
                
                execution_time = time.time() - start_time
                print(f"Pipeline execution completed in {execution_time:.2f} seconds")
                
                output_files = list(Path(output_dir).rglob("*.mp4"))
                scene_files = [f for f in output_files if "scene_" in f.name]
                final_files = [f for f in output_files if "final" in f.name]
                
                print(f"\n📊 Generation Results:")
                print(f"Total video files: {len(output_files)}")
                print(f"Scene videos: {len(scene_files)}")
                print(f"Final videos: {len(final_files)}")
                
                total_duration = 0
                for video_file in output_files:
                    file_size = video_file.stat().st_size
                    print(f"📹 {video_file.name}: {file_size} bytes")
                    
                    try:
                        import cv2
                        cap = cv2.VideoCapture(str(video_file))
                        if cap.isOpened():
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                            duration = frame_count / fps if fps > 0 else 0
                            total_duration += duration
                            print(f"   Duration: {duration:.2f} seconds")
                            cap.release()
                    except:
                        pass
                
                print(f"\n⏱️  Total video duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
                
                if total_duration >= 1200:
                    print("✅ Successfully generated 20+ minute anime content!")
                    return True
                elif total_duration >= 600:
                    print("⚠️  Generated significant content but less than 20 minutes")
                    return True
                else:
                    print("❌ Generated content is too short")
                    return False
                    
            except Exception as pipeline_error:
                print(f"❌ Pipeline execution error: {pipeline_error}")
                return False
                
    except Exception as e:
        print(f"❌ Test setup error: {e}")
        return False

def main():
    """Run 20-minute anime generation test"""
    print("🎬 20-Minute Anime Pipeline Test")
    print("=" * 50)
    
    success = test_20_minute_anime_generation()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 20-minute anime pipeline test completed successfully!")
    else:
        print("⚠️  20-minute anime pipeline test encountered issues.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
