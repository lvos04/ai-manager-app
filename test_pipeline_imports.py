#!/usr/bin/env python3
"""Quick test to verify pipeline imports and instantiation."""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test all pipeline imports and instantiation."""
    try:
        from backend.pipelines.channel_specific.gaming_pipeline import GamingPipeline
        print('✅ Gaming pipeline import successful')
        
        from backend.pipelines.channel_specific.manga_pipeline import MangaPipeline
        print('✅ Manga pipeline import successful')
        
        g = GamingPipeline()
        print('✅ Gaming pipeline instantiated successfully')
        
        m = MangaPipeline()
        print('✅ Manga pipeline instantiated successfully')
        
        import inspect
        gaming_sig = inspect.signature(g.run)
        manga_sig = inspect.signature(m.run)
        
        print(f"Gaming run signature: {gaming_sig}")
        print(f"Manga run signature: {manga_sig}")
        
        gaming_params = list(gaming_sig.parameters.keys())
        manga_params = list(manga_sig.parameters.keys())
        
        required_params = ['render_fps', 'output_fps', 'frame_interpolation_enabled']
        
        for param in required_params:
            if param in gaming_params:
                print(f"✅ Gaming pipeline has {param}")
            else:
                print(f"❌ Gaming pipeline missing {param}")
                
            if param in manga_params:
                print(f"✅ Manga pipeline has {param}")
            else:
                print(f"❌ Manga pipeline missing {param}")
        
        return True
        
    except Exception as e:
        print(f"❌ Import/instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
