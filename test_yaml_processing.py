#!/usr/bin/env python3
"""Test script processing with user's YAML file."""

import asyncio
import sys
import os
sys.path.append('/home/ubuntu/repos/ai-manager-app')

from backend.ai_tasks import extract_scenes_from_pipeline

async def test_yaml_processing():
    """Test processing of user's YAML file."""
    input_path = "/home/leon/Documents/aetherion_aflevering_1.yaml"
    
    if not os.path.exists(input_path):
        print(f"YAML file not found: {input_path}")
        return
    
    print("Testing YAML script processing...")
    
    scenes = await extract_scenes_from_pipeline(input_path, "anime")
    
    print(f"Extracted {len(scenes)} scenes:")
    for i, scene in enumerate(scenes):
        print(f"Scene {i+1}: {scene}")
    
    import yaml
    with open(input_path, 'r', encoding='utf-8') as f:
        script_data = yaml.safe_load(f)
    
    print(f"Characters: {script_data.get('characters', [])}")
    print(f"Original scenes: {script_data.get('scenes', [])}")

if __name__ == "__main__":
    asyncio.run(test_yaml_processing())
