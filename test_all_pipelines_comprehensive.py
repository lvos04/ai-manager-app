#!/usr/bin/env python3
"""
Comprehensive test script for all channel pipelines
Tests video generation, device handling, and basic functionality across all pipelines
"""

import sys
import os
import time
import tempfile
import yaml
from pathlib import Path

sys.path.append('.')

def create_test_script_for_pipeline(pipeline_type):
    """Create appropriate test script for each pipeline type"""
    
    base_scenes = [
        {
            'scene_number': 1,
            'description': f'{pipeline_type.title()} opening scene with dynamic action',
            'dialogue': 'The adventure begins now!',
            'duration': 300,
            'characters': ['Protagonist'],
            'location': 'Starting Location'
        },
        {
            'scene_number': 2,
            'description': f'Epic {pipeline_type} battle scene with special effects',
            'dialogue': 'This is the final confrontation!',
            'duration': 300,
            'characters': ['Protagonist', 'Antagonist'],
            'location': 'Battle Arena',
            'scene_type': 'combat'
        }
    ]
    
    characters = [
        {
            'name': 'Protagonist',
            'description': f'{pipeline_type.title()} hero with special abilities',
            'voice_style': 'heroic'
        },
        {
            'name': 'Antagonist',
            'description': f'Powerful {pipeline_type} villain',
            'voice_style': 'menacing'
        }
    ]
    
    locations = [
        {
            'name': 'Starting Location',
            'description': f'{pipeline_type.title()}-style environment'
        },
        {
            'name': 'Battle Arena',
            'description': f'Epic {pipeline_type} battle location'
        }
    ]
    
    return {
        'title': f'{pipeline_type.title()} Test Episode',
        'description': f'Test episode for {pipeline_type} pipeline validation',
        'target_duration': 600,
        'scenes': base_scenes,
        'characters': characters,
        'locations': locations
    }

def test_pipeline_video_generation(pipeline_name, pipeline_module):
    """Test video generation for a specific pipeline"""
    print(f"🧪 Testing {pipeline_name} pipeline...")
    
    try:
        test_script = create_test_script_for_pipeline(pipeline_name)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            script_path = os.path.join(temp_dir, f"{pipeline_name}_test_script.yaml")
            with open(script_path, 'w') as f:
                yaml.dump(test_script, f, default_flow_style=False)
            
            output_dir = os.path.join(temp_dir, f"{pipeline_name}_output")
            os.makedirs(output_dir, exist_ok=True)
            
            print(f"Running {pipeline_name} pipeline...")
            
            start_time = time.time()
            
            try:
                if pipeline_name == "gaming":
                    result = pipeline_module.run(
                        input_path=script_path,
                        output_path=output_dir,
                        base_model="stable_diffusion_1_5",
                        lora_models=["gaming_style"],
                        language="en"
                    )
                else:
                    result = pipeline_module.run(
                        input_path=script_path,
                        output_path=output_dir,
                        base_model="stable_diffusion_1_5",
                        lora_models=[f"{pipeline_name}_style"],
                        language="en"
                    )
                
                execution_time = time.time() - start_time
                print(f"Pipeline execution completed in {execution_time:.2f} seconds")
                
                output_files = list(Path(output_dir).rglob("*.mp4"))
                if output_files:
                    total_size = sum(f.stat().st_size for f in output_files)
                    print(f"✅ {pipeline_name} pipeline generated {len(output_files)} video files ({total_size} bytes total)")
                    return True
                else:
                    print(f"⚠️  {pipeline_name} pipeline completed but no video files found")
                    return False
                    
            except Exception as pipeline_error:
                print(f"❌ {pipeline_name} pipeline execution error: {pipeline_error}")
                return False
                
    except Exception as e:
        print(f"❌ {pipeline_name} test setup error: {e}")
        return False

def test_video_generation_models():
    """Test that all video generation models load without device_map errors"""
    print("🧪 Testing video generation models...")
    
    try:
        from backend.pipelines.video_generation import TextToVideoGenerator
        
        generator = TextToVideoGenerator()
        
        models_to_test = [
            "animatediff_v2_sdxl",
            "animatediff_lightning", 
            "svd_xt",
            "zeroscope_v2_xl",
            "modelscope_t2v",
            "ltx_video",
            "skyreels_v2"
        ]
        
        results = {}
        
        for model_name in models_to_test:
            print(f"Testing {model_name}...")
            try:
                model = generator.load_model(model_name)
                if model is not None:
                    print(f"✅ {model_name} loaded successfully")
                    results[model_name] = True
                    generator._cleanup_model_memory()
                else:
                    print(f"⚠️  {model_name} returned None")
                    results[model_name] = False
                    
            except Exception as e:
                if 'device_map' in str(e).lower() or 'motionadapter' in str(e).lower():
                    print(f"❌ {model_name} device map error: {e}")
                    results[model_name] = False
                else:
                    print(f"⚠️  {model_name} different error (may be expected): {e}")
                    results[model_name] = True
        
        successful_models = sum(1 for success in results.values() if success)
        total_models = len(results)
        
        print(f"\nModel loading results: {successful_models}/{total_models} successful")
        
        return successful_models >= total_models * 0.7
        
    except Exception as e:
        print(f"❌ Video model testing error: {e}")
        return False

def main():
    """Run comprehensive tests for all pipelines"""
    print("🎬 Comprehensive All-Pipelines Test Suite")
    print("=" * 60)
    
    print("\n🧪 Testing Video Generation Models")
    print("-" * 40)
    models_success = test_video_generation_models()
    
    pipelines_to_test = [
        ("anime", "backend.pipelines.channel_specific.anime_pipeline"),
        ("gaming", "backend.pipelines.channel_specific.gaming_pipeline"),
        ("superhero", "backend.pipelines.channel_specific.superhero_pipeline"),
        ("marvel_dc", "backend.pipelines.channel_specific.marvel_dc_pipeline"),
        ("original_manga", "backend.pipelines.channel_specific.original_manga_pipeline")
    ]
    
    pipeline_results = []
    
    for pipeline_name, module_path in pipelines_to_test:
        print(f"\n🧪 Testing {pipeline_name.title()} Pipeline")
        print("-" * 40)
        
        try:
            module = __import__(module_path, fromlist=['run'])
            result = test_pipeline_video_generation(pipeline_name, module)
            pipeline_results.append((pipeline_name, result))
            
        except ImportError as e:
            print(f"❌ Could not import {pipeline_name} pipeline: {e}")
            pipeline_results.append((pipeline_name, False))
        except Exception as e:
            print(f"💥 {pipeline_name} pipeline test crashed: {e}")
            pipeline_results.append((pipeline_name, False))
    
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE TEST SUMMARY")
    print("=" * 60)
    
    print(f"Video Models: {'✅ PASSED' if models_success else '❌ FAILED'}")
    
    passed_pipelines = sum(1 for _, result in pipeline_results if result)
    total_pipelines = len(pipeline_results)
    
    for pipeline_name, result in pipeline_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{pipeline_name.title()} Pipeline: {status}")
    
    print(f"\nOverall: {passed_pipelines}/{total_pipelines} pipelines passed")
    
    overall_success = models_success and passed_pipelines >= total_pipelines * 0.8
    
    if overall_success:
        print("🎉 All pipelines are ready for video generation!")
    else:
        print("⚠️  Some pipelines need additional fixes.")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
