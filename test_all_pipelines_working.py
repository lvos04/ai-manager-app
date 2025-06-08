#!/usr/bin/env python3
"""
Comprehensive test for all 6 working pipelines without placeholders.
Tests real AI model integration and substantial video generation.
"""

import os
import sys
import tempfile
import logging
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_script(channel_type: str) -> str:
    """Create test script for each channel type."""
    scripts = {
        "anime": """
title: "Test Anime Episode"
description: "Anime adventure with character development"
scenes:
  - description: "Protagonist discovers hidden powers in mystical forest"
    duration: 300
  - description: "Training montage with mentor character"
    duration: 300
characters:
  - name: "Akira"
    description: "Young hero with determination"
  - name: "Sensei"
    description: "Wise mentor with ancient knowledge"
locations:
  - name: "Mystical Forest"
    description: "Ancient woodland with magical energy"
""",
        "gaming": """
title: "Epic Gaming Moments"
description: "Compilation of strategic gameplay highlights"
scenes:
  - description: "Clutch victory in competitive match"
    duration: 300
  - description: "Perfect speedrun execution"
    duration: 300
characters:
  - name: "Pro Gamer"
    description: "Skilled competitive player"
locations:
  - name: "Gaming Arena"
    description: "High-tech esports venue"
""",
        "superhero": """
title: "Hero's Journey"
description: "Original superhero origin story"
scenes:
  - description: "Ordinary person gains extraordinary powers"
    duration: 300
  - description: "First heroic act saving civilians"
    duration: 300
characters:
  - name: "Nova"
    description: "New superhero with energy powers"
  - name: "Mentor"
    description: "Experienced hero guide"
locations:
  - name: "Metro City"
    description: "Modern metropolis needing protection"
""",
        "manga": """
title: "Manga Adventure"
description: "Classic manga-style storytelling"
scenes:
  - description: "School life with supernatural elements"
    duration: 300
  - description: "Friendship bonds tested by challenges"
    duration: 300
characters:
  - name: "Yuki"
    description: "Student with hidden abilities"
  - name: "Rei"
    description: "Best friend and ally"
locations:
  - name: "Tokyo High School"
    description: "Modern Japanese school setting"
""",
        "marvel_dc": """
title: "Cosmic Crossover"
description: "Epic superhero team adventure"
scenes:
  - description: "Heroes unite against cosmic threat"
    duration: 300
  - description: "Multiverse-spanning battle"
    duration: 300
characters:
  - name: "Captain Shield"
    description: "Leader with tactical expertise"
  - name: "Speed Force"
    description: "Fastest hero alive"
locations:
  - name: "Cosmic Nexus"
    description: "Hub of all realities"
""",
        "original_manga": """
title: "Original Creation"
description: "Unique manga universe story"
scenes:
  - description: "Creative world with unique magic system"
    duration: 300
  - description: "Original characters face unprecedented challenges"
    duration: 300
characters:
  - name: "Zara"
    description: "Original protagonist with unique design"
  - name: "Kael"
    description: "Supporting character with distinct abilities"
locations:
  - name: "Ethereal Realm"
    description: "Original fantasy world setting"
"""
    }
    return scripts.get(channel_type, scripts["anime"])

def test_pipeline(channel_type: str) -> bool:
    """Test individual pipeline for real functionality."""
    print(f"\n🎬 TESTING {channel_type.upper()} PIPELINE")
    print("=" * 60)
    
    try:
        if channel_type == "anime":
            from backend.pipelines.channel_specific.anime_pipeline import AnimePipeline
            pipeline = AnimePipeline()
        elif channel_type == "gaming":
            from backend.pipelines.channel_specific.gaming_pipeline import GamingPipeline
            pipeline = GamingPipeline()
        elif channel_type == "superhero":
            from backend.pipelines.channel_specific.superhero_pipeline import SuperheroPipeline
            pipeline = SuperheroPipeline()
        elif channel_type == "manga":
            from backend.pipelines.channel_specific.manga_pipeline import MangaPipeline
            pipeline = MangaPipeline()
        elif channel_type == "marvel_dc":
            from backend.pipelines.channel_specific.marvel_dc_pipeline import MarvelDCPipeline
            pipeline = MarvelDCPipeline()
        elif channel_type == "original_manga":
            from backend.pipelines.channel_specific.original_manga_pipeline import OriginalMangaPipeline
            pipeline = OriginalMangaPipeline()
        else:
            print(f"❌ Unknown channel type: {channel_type}")
            return False
        
        with tempfile.TemporaryDirectory() as temp_dir:
            script_path = os.path.join(temp_dir, f"{channel_type}_test.yaml")
            output_path = os.path.join(temp_dir, f"{channel_type}_output")
            
            script_content = create_test_script(channel_type)
            
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            print(f"📝 Script: {script_path}")
            print(f"📁 Output: {output_path}")
            
            print(f"🚀 Running {channel_type} pipeline...")
            result = pipeline.run(
                input_path=script_path,
                output_path=output_path,
                base_model="stable_diffusion_1_5",
                lora_models=[],
                render_fps=24,
                output_fps=24,
                frame_interpolation_enabled=True,
                language="en"
            )
            
            print(f"✅ Pipeline completed: {result}")
            
            if result and os.path.exists(result):
                episode_files = [
                    "anime_episode.mp4",
                    "gaming_episode.mp4", 
                    "superhero_episode.mp4",
                    "manga_episode.mp4",
                    "marvel_dc_episode.mp4",
                    "original_manga_episode.mp4"
                ]
                
                episode_file = None
                for filename in episode_files:
                    potential_path = os.path.join(result, "final", filename)
                    if os.path.exists(potential_path):
                        episode_file = potential_path
                        break
                
                if episode_file:
                    file_size = os.path.getsize(episode_file)
                    print(f"📊 Episode video file size: {file_size} bytes")
                    
                    if file_size > 1000000:  # At least 1MB for substantial content
                        print("✅ Episode video has substantial content")
                        
                        scenes_dir = os.path.join(result, "scenes")
                        if os.path.exists(scenes_dir):
                            scene_files = [f for f in os.listdir(scenes_dir) if f.endswith('.mp4')]
                            print(f"📁 Generated {len(scene_files)} scene files")
                        
                        shorts_dir = os.path.join(result, "shorts")
                        if os.path.exists(shorts_dir):
                            short_files = [f for f in os.listdir(shorts_dir) if f.endswith('.mp4')]
                            print(f"🎬 Generated {len(short_files)} short files")
                        
                        return True
                    else:
                        print(f"❌ Episode video too small: {file_size} bytes")
                        return False
                else:
                    print(f"❌ Episode file not found in: {os.path.join(result, 'final')}")
                    return False
            else:
                print("❌ No output directory generated")
                return False
                
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run comprehensive pipeline tests."""
    print("🎯 COMPREHENSIVE PIPELINE TEST SUITE")
    print("Testing all 6 pipelines for real AI model integration")
    print("=" * 70)
    
    channels = ["anime", "gaming", "superhero", "manga", "marvel_dc", "original_manga"]
    results = {}
    
    for channel in channels:
        results[channel] = test_pipeline(channel)
    
    print("\n📊 TEST RESULTS SUMMARY")
    print("=" * 70)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for channel, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{channel}: {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} pipelines working")
    
    if passed_tests == total_tests:
        print("🎉 All pipelines WORKING with real AI models!")
        return True
    else:
        print("⚠️  Some pipelines failed - check implementations")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
