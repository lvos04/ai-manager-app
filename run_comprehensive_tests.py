#!/usr/bin/env python3
"""Run all comprehensive tests for the AI Manager App."""

import asyncio
import sys
import subprocess
from pathlib import Path

async def run_syntax_validation():
    """Run syntax validation test."""
    print("🔍 Running syntax validation...")
    result = subprocess.run([sys.executable, "test_syntax_only.py"], 
                          capture_output=True, text=True, cwd="/home/ubuntu/repos/ai-manager-app")
    
    if result.returncode == 0:
        print("✅ Syntax validation passed")
        return True
    else:
        print("❌ Syntax validation failed")
        print(result.stdout)
        print(result.stderr)
        return False

async def run_pipeline_tests():
    """Run comprehensive pipeline tests."""
    print("🧪 Running comprehensive pipeline tests...")
    result = subprocess.run([sys.executable, "test_comprehensive_syntax.py"], 
                          capture_output=True, text=True, cwd="/home/ubuntu/repos/ai-manager-app")
    
    if result.returncode == 0:
        print("✅ Pipeline tests passed")
        return True
    else:
        print("❌ Pipeline tests failed")
        print(result.stdout)
        print(result.stderr)
        return False

async def run_final_validation():
    """Run final validation tests."""
    print("🎯 Running final validation tests...")
    result = subprocess.run([sys.executable, "test_final_validation.py"], 
                          capture_output=True, text=True, cwd="/home/ubuntu/repos/ai-manager-app")
    
    if result.returncode == 0:
        print("✅ Final validation passed")
        return True
    else:
        print("❌ Final validation failed")
        print(result.stdout)
        print(result.stderr)
        return False

async def main():
    """Run all tests in sequence."""
    print("🚀 Starting Comprehensive Test Suite")
    
    tests = [
        ("Syntax Validation", run_syntax_validation),
        ("Pipeline Tests", run_pipeline_tests),
        ("Final Validation", run_final_validation)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running {test_name}")
        print(f"{'='*50}")
        
        try:
            success = await test_func()
            if success:
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                failed += 1
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} ERROR: {e}")
    
    print(f"\n{'='*50}")
    print(f"TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {len(tests)}")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED!")
        return True
    else:
        print("❌ SOME TESTS FAILED!")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
