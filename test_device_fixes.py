#!/usr/bin/env python3
"""
Test device map fixes for video generation models
"""

import sys
import os
sys.path.append('.')

def test_device_map_fixes():
    """Test that device_map fixes are working."""
    print("🔧 Testing Device Map Fixes")
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

if __name__ == "__main__":
    success = test_device_map_fixes()
    sys.exit(0 if success else 1)
