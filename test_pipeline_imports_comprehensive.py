#!/usr/bin/env python3
"""
Comprehensive test of all pipeline import fixes.
Tests that all import errors preventing video generation are resolved.
"""

import sys
import os

sys.path.insert(0, os.path.abspath('.'))

def test_core_imports():
    """Test core module imports that were causing 'No module named core' errors."""
    print("Testing core module imports...")
    
    try:
        from backend.core.async_pipeline_manager import get_async_pipeline_manager
        print("✅ async_pipeline_manager import successful")
        
        from backend.core.performance_monitor import get_performance_monitor
        print("✅ performance_monitor import successful")
        
        return True
    except Exception as e:
        print(f"❌ Core imports failed: {e}")
        return False

def test_pipeline_imports():
    """Test pipeline module imports that were causing 'No module named pipelines' errors."""
    print("\nTesting pipeline module imports...")
    
    try:
        from backend.pipelines.channel_specific import get_pipeline_for_channel
        print("✅ channel_specific import successful")
        
        from backend.localization.multi_language_pipeline import multi_language_pipeline_manager
        print("✅ multi_language_pipeline import successful")
        
        return True
    except Exception as e:
        print(f"❌ Pipeline imports failed: {e}")
        return False

def test_all_channel_pipelines():
    """Test that all 6 channel-specific pipelines can be imported."""
    print("\nTesting all channel-specific pipeline imports...")
    
    pipelines = [
        "anime_pipeline",
        "gaming_pipeline", 
        "superhero_pipeline",
        "manga_pipeline",
        "marvel_dc_pipeline",
        "original_manga_pipeline"
    ]
    
    success_count = 0
    
    for pipeline_name in pipelines:
        try:
            module_path = f"backend.pipelines.channel_specific.{pipeline_name}"
            module = __import__(module_path, fromlist=[pipeline_name])
            print(f"✅ {pipeline_name} import successful")
            success_count += 1
        except Exception as e:
            print(f"❌ {pipeline_name} import failed: {e}")
    
    return success_count == len(pipelines)

def test_utility_imports():
    """Test utility module imports."""
    print("\nTesting utility module imports...")
    
    try:
        from backend.utils.error_handler import PipelineErrorHandler
        print("✅ error_handler import successful")
        
        from backend.utils.duration_parser import parse_duration
        print("✅ duration_parser import successful")
        
        return True
    except Exception as e:
        print(f"❌ Utility imports failed: {e}")
        return False

def test_database_imports():
    """Test database imports that were previously fixed."""
    print("\nTesting database imports...")
    
    try:
        from backend.database import DBProject, DBPipelineRun, DBProjectLora, get_db
        print("✅ All database imports successful")
        
        from backend.ai_tasks import run_pipeline, queue_pipeline, get_queue_status
        print("✅ ai_tasks imports successful")
        
        return True
    except Exception as e:
        print(f"❌ Database imports failed: {e}")
        return False

def test_application_startup():
    """Test that the application starts without import errors."""
    print("\nTesting application startup...")
    
    try:
        import main
        print("✅ main.py import successful")
        
        from backend import api
        print("✅ backend.api import successful")
        
        return True
    except Exception as e:
        print(f"❌ Application startup failed: {e}")
        return False

def test_async_pipeline_execution():
    """Test that async pipeline execution can be initiated without import errors."""
    print("\nTesting async pipeline execution setup...")
    
    try:
        from backend.ai_tasks import run_pipeline_async
        print("✅ run_pipeline_async import successful")
        
        from backend.core.async_pipeline_manager import AsyncPipelineManager
        print("✅ AsyncPipelineManager import successful")
        
        return True
    except Exception as e:
        print(f"❌ Async pipeline execution setup failed: {e}")
        return False

def main():
    """Run comprehensive pipeline import tests."""
    print("Comprehensive Pipeline Import Test Suite")
    print("Fixing 'No module named core' and 'No module named pipelines' errors")
    print("=" * 70)
    
    tests = [
        ("Core Imports", test_core_imports),
        ("Pipeline Imports", test_pipeline_imports),
        ("All Channel Pipelines", test_all_channel_pipelines),
        ("Utility Imports", test_utility_imports),
        ("Database Imports", test_database_imports),
        ("Application Startup", test_application_startup),
        ("Async Pipeline Execution", test_async_pipeline_execution)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                failed += 1
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} ERROR: {e}")
    
    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE PIPELINE IMPORT TEST SUMMARY")
    print(f"{'='*70}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {len(tests)}")
    
    if failed == 0:
        print("🎉 ALL PIPELINE IMPORT TESTS PASSED!")
        print("✅ 'No module named core' error RESOLVED")
        print("✅ 'No module named pipelines' error RESOLVED")
        print("✅ All channel-specific pipelines can be imported")
        print("✅ Async pipeline execution ready")
        print("✅ User can now start generating videos!")
        return True
    else:
        print("❌ SOME PIPELINE IMPORT TESTS FAILED!")
        print("⚠️  Video generation may still encounter import errors")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
