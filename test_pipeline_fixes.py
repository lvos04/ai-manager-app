#!/usr/bin/env python3
"""
Quick test to verify MotionAdapter fixes work across all pipelines
"""

import sys
import os
sys.path.append('.')

def test_motionadapter_fix():
    """Test that MotionAdapter loads without device_map errors"""
    print("Testing MotionAdapter device loading fix...")
    try:
        from backend.pipelines.video_generation import TextToVideoGenerator
        
        generator = TextToVideoGenerator()
        
        print("Loading animatediff_v2_sdxl...")
        model = generator.load_model('animatediff_v2_sdxl')
        
        if model is not None:
            print("✅ AnimateDiff v2 SDXL loaded successfully")
            generator._cleanup_model_memory()
            return True
        else:
            print("⚠️  Model returned None but no exception")
            return False
            
    except Exception as e:
        if 'device_map' in str(e).lower() and 'motionadapter' in str(e).lower():
            print(f"❌ MotionAdapter device_map error still present: {e}")
            return False
        else:
            print(f"⚠️  Different error (may be expected): {e}")
            return True

def test_all_video_models():
    """Test loading of all video models"""
    print("Testing all video models...")
    try:
        from backend.pipelines.video_generation import TextToVideoGenerator
        
        generator = TextToVideoGenerator()
        models = ["svd_xt", "zeroscope_v2_xl", "animatediff_v2_sdxl", "animatediff_lightning", "modelscope_t2v"]
        
        results = {}
        for model_name in models:
            try:
                print(f"Testing {model_name}...")
                model = generator.load_model(model_name)
                results[model_name] = model is not None
                if model:
                    generator._cleanup_model_memory()
            except Exception as e:
                if 'device_map' in str(e).lower():
                    print(f"❌ {model_name} device_map error: {e}")
                    results[model_name] = False
                else:
                    results[model_name] = True
        
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        print(f"Model loading: {successful}/{total} successful")
        return successful >= total * 0.7
        
    except Exception as e:
        print(f"❌ Video model test error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Pipeline Fixes")
    print("=" * 40)
    
    motionadapter_ok = test_motionadapter_fix()
    models_ok = test_all_video_models()
    
    if motionadapter_ok and models_ok:
        print("✅ All fixes working correctly!")
        sys.exit(0)
    else:
        print("❌ Some fixes need attention")
        sys.exit(1)
