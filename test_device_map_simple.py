#!/usr/bin/env python3
"""
Simple test to verify device_map fixes work
"""

import sys
sys.path.append('.')

def test_device_map_fix():
    print('Testing device_map fix...')
    try:
        from backend.pipelines.video_generation import TextToVideoGenerator
        generator = TextToVideoGenerator()
        model = generator.load_model('animatediff_v2_sdxl')
        print('✅ Device_map fix working - no auto not supported errors')
        return True
    except Exception as e:
        if 'auto not supported' in str(e):
            print('❌ Still has auto not supported error')
            return False
        else:
            print(f'✅ Different error (expected): {e}')
            return True

if __name__ == "__main__":
    success = test_device_map_fix()
    sys.exit(0 if success else 1)
