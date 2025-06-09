#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_all_model_configurations():
    print("Testing comprehensive model configurations...")
    
    try:
        from backend.model_manager import BASE_MODELS, BASE_MODEL_PROMPT_TEMPLATES, HF_MODEL_REPOS, CIVITAI_LORA_MODELS
        
        print("\n=== BASE_MODEL_PROMPT_TEMPLATES ===")
        for model_key, template in BASE_MODEL_PROMPT_TEMPLATES.items():
            print(f"{model_key}:")
            print(f"  prefix: {template['prefix'][:50]}...")
            if 'structure' in template:
                print(f"  structure: {template['structure']}")
            print(f"  negative: {template['negative'][:50]}...")
        
        print("\n=== HF_MODEL_REPOS consistency check ===")
        for model_key in BASE_MODELS.keys():
            if model_key in HF_MODEL_REPOS:
                print(f"✓ {model_key}: {HF_MODEL_REPOS[model_key]}")
            else:
                print(f"✗ {model_key}: MISSING HF repo mapping")
        
        print("\n=== CIVITAI_LORA_MODELS ===")
        for model_key, model_info in CIVITAI_LORA_MODELS.items():
            if isinstance(model_info, dict) and 'model_id' in model_info:
                model_id = model_info['model_id']
                version_id = model_info.get('version_id', 'N/A')
                name = model_info.get('name', model_key)
                print(f"{model_key}: ID {model_id}, Version {version_id} - {name}")
            else:
                print(f"{model_key}: {model_info}")
        
        print("\n=== Template coverage check ===")
        for model_key in BASE_MODELS.keys():
            if model_key in BASE_MODEL_PROMPT_TEMPLATES:
                print(f"✓ {model_key} has prompt template")
            else:
                print(f"✗ {model_key} missing prompt template")
        
        print("\n✅ All model configurations loaded successfully")
        
    except Exception as e:
        print(f"❌ Error testing model configurations: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_all_model_configurations()
