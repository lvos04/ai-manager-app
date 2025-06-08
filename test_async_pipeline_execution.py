#!/usr/bin/env python3
"""Test async pipeline manager execution to verify fixes work across all channel types."""

import asyncio
import tempfile
import os
import sys
from pathlib import Path

def test_async_pipeline_execution():
    """Test async pipeline manager with video/voice/music tasks."""
    print("🔍 Testing AsyncPipelineManager execution across all channel types...")
    
    try:
        from backend.core.async_pipeline_manager import get_async_pipeline_manager
        
        async def run_async_test():
            manager = get_async_pipeline_manager()
            
            with tempfile.TemporaryDirectory() as temp_dir:
                print(f"📁 Using temp directory: {temp_dir}")
                
                video_task = {
                    'type': 'video_generation',
                    'text': 'Test video scene',
                    'model': 'animatediff_v2_sdxl',
                    'output_path': f'{temp_dir}/test_video.mp4'
                }
                
                voice_task = {
                    'type': 'voice_generation',
                    'text': 'Test voice line',
                    'character_voice': 'default',
                    'output_path': f'{temp_dir}/test_voice.wav'
                }
                
                music_task = {
                    'type': 'music_generation',
                    'description': 'Test background music',
                    'duration': 3.0,
                    'output_path': f'{temp_dir}/test_music.wav'
                }
                
                tasks = [video_task, voice_task, music_task]
                print(f"🚀 Executing {len(tasks)} async tasks...")
                
                results = await manager.execute_tasks(tasks)
                
                print('\n📊 Async Pipeline Test Results:')
                for i, result in enumerate(results):
                    success = result.get("success", False)
                    time_taken = result.get("processing_time", 0)
                    print(f'  Task {i+1} ({tasks[i]["type"]}): Success={success}, Time={time_taken:.2f}s')
                    if result.get('error'):
                        print(f'    Error: {result["error"]}')
                    if success and result.get('output_path'):
                        output_path = result['output_path']
                        if os.path.exists(output_path):
                            size = os.path.getsize(output_path)
                            print(f'    Output: {output_path} ({size} bytes)')
                        else:
                            print(f'    Output: {output_path} (FILE NOT FOUND)')
                
                success_count = sum(1 for r in results if r.get('success', False))
                success_rate = success_count / len(results) if results else 0
                print(f'\n🎯 Overall Success Rate: {success_rate:.1%} ({success_count}/{len(results)})')
                
                output_files = list(Path(temp_dir).glob("*"))
                print(f"📁 Generated {len(output_files)} output files:")
                for file_path in output_files:
                    size = file_path.stat().st_size
                    print(f"  - {file_path.name}: {size} bytes")
                
                return success_rate > 0 and len(output_files) > 0
        
        success = asyncio.run(run_async_test())
        
        if success:
            print("\n✅ AsyncPipelineManager test PASSED - tasks execute successfully")
        else:
            print("\n❌ AsyncPipelineManager test FAILED - tasks still failing")
        
        return success
        
    except Exception as e:
        print(f"\n❌ AsyncPipelineManager test ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_all_channel_pipelines():
    """Test all 6 channel pipelines to verify they work with fixed AsyncPipelineManager."""
    print("\n🔍 Testing all channel pipelines with AsyncPipelineManager...")
    
    channel_types = ["anime", "gaming", "superhero", "manga", "marvel_dc", "original_manga"]
    results = {}
    
    for channel_type in channel_types:
        try:
            print(f"\n📺 Testing {channel_type} pipeline...")
            
            from backend.localization.multi_language_pipeline import multi_language_pipeline_manager
            
            test_scenes = [
                {
                    "description": f"Test scene for {channel_type}",
                    "dialogue": "This is a test scene",
                    "duration": 3.0,
                    "characters": ["TestCharacter"],
                    "location": "TestLocation"
                }
            ]
            
            with tempfile.TemporaryDirectory() as temp_dir:
                config = {
                    "channel_type": channel_type,
                    "base_model": "stable_diffusion_1_5",
                    "output_path": temp_dir,
                    "language": "en"
                }
                
                async def test_channel():
                    result = await multi_language_pipeline_manager.execute_multi_language_pipeline(
                        test_scenes, config, ["en"]
                    )
                    return result
                
                result = asyncio.run(test_channel())
                
                output_files = list(Path(temp_dir).rglob("*"))
                video_files = [f for f in output_files if f.suffix in ['.mp4', '.avi', '.mov']]
                
                success = result.get("performance_metrics", {}).get("success_rate", 0) > 0
                has_videos = len(video_files) > 0
                
                results[channel_type] = {
                    "success": success and has_videos,
                    "video_files": len(video_files),
                    "total_files": len(output_files)
                }
                
                print(f"  Result: Success={success}, Videos={len(video_files)}, Files={len(output_files)}")
                
        except Exception as e:
            print(f"  Error testing {channel_type}: {e}")
            results[channel_type] = {"success": False, "error": str(e)}
    
    print(f"\n📊 Channel Pipeline Test Summary:")
    successful_channels = 0
    for channel, result in results.items():
        success = result.get("success", False)
        if success:
            successful_channels += 1
        print(f"  {channel}: {'✅' if success else '❌'} {result}")
    
    overall_success = successful_channels == len(channel_types)
    print(f"\n🎯 Overall Channel Success: {successful_channels}/{len(channel_types)} channels working")
    
    return overall_success

if __name__ == "__main__":
    os.chdir("/home/ubuntu/repos/ai-manager-app")
    sys.path.insert(0, ".")
    
    print("🚀 COMPREHENSIVE ASYNC PIPELINE TESTING")
    print("=" * 50)
    
    async_success = test_async_pipeline_execution()
    channel_success = test_all_channel_pipelines()
    
    print("\n" + "=" * 50)
    if async_success and channel_success:
        print("🎉 ALL TESTS PASSED!")
        print("✅ AsyncPipelineManager fixed and working")
        print("✅ All channel pipelines generate video content")
        print("✅ 0% success rate issue resolved")
    else:
        print("❌ SOME TESTS FAILED!")
        print(f"AsyncPipelineManager: {'✅' if async_success else '❌'}")
        print(f"Channel Pipelines: {'✅' if channel_success else '❌'}")
    print("=" * 50)
    
    sys.exit(0 if (async_success and channel_success) else 1)
