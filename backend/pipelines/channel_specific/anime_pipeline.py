"""
AI Original Anime Series Channel Pipeline
Self-contained anime content generation with complete internal processing.
"""

import os
import sys
import json
import yaml
import time
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from .base_pipeline import BasePipeline

logger = logging.getLogger(__name__)

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Warning: PIL/Pillow not available. Image generation will be limited.")
    Image = ImageDraw = ImageFont = None

class AnimePipeline(BasePipeline):
    """Self-contained anime content generation pipeline."""
    
    def __init__(self):
        super().__init__("anime")
        self.combat_calls_count = 0
        self.max_combat_calls = 3
    
    def run(self, input_path: str, output_path: str, base_model: str = "stable_diffusion_1_5", 
            lora_models: Optional[List[str]] = None, lora_paths: Optional[Dict[str, str]] = None, 
            db_run=None, db=None, render_fps: int = 24, output_fps: int = 24, 
            frame_interpolation_enabled: bool = True, language: str = "en") -> str:
        """
        Run the self-contained anime pipeline.
        
        Args:
            input_path: Path to input script/description
            output_path: Path to output directory
            base_model: Base model to use for generation
            lora_models: List of LoRA models to apply
            lora_paths: Dictionary mapping LoRA model names to their file paths
            db_run: Database run object for progress tracking
            db: Database session
            render_fps: Rendering frame rate
            output_fps: Output frame rate
            frame_interpolation_enabled: Enable frame interpolation
            language: Target language
            
        Returns:
            str: Path to output directory
        """
        
        print("Running self-contained anime pipeline")
        print(f"Using base model: {base_model}")
        print(f"Using LoRA models: {lora_models}")
        print(f"Language: {language}")
        
        try:
            return self._execute_pipeline(
                input_path, output_path, base_model, lora_models, 
                db_run, db, render_fps, output_fps, frame_interpolation_enabled, language
            )
        except Exception as e:
            logger.error(f"Anime pipeline failed: {e}")
            raise
        finally:
            self.cleanup_models()
    
    def _execute_pipeline(self, input_path: str, output_path: str, base_model: str, 
                         lora_models: Optional[List[str]], db_run, db, render_fps: int, 
                         output_fps: int, frame_interpolation_enabled: bool, language: str) -> str:
        
        output_dir = self.ensure_output_dir(output_path)
        
        scenes_dir = output_dir / "scenes"
        scenes_dir.mkdir(exist_ok=True)
        
        characters_dir = output_dir / "characters"
        characters_dir.mkdir(exist_ok=True)
        
        final_dir = output_dir / "final"
        final_dir.mkdir(exist_ok=True)
        
        shorts_dir = output_dir / "shorts"
        shorts_dir.mkdir(exist_ok=True)
        
        print("Step 1: Loading and parsing script...")
        if db_run and db:
            db_run.progress = 5.0
            db.commit()
        
        script_data = self.parse_input_script(input_path)
        scenes = script_data.get('scenes', [])
        characters = script_data.get('characters', [])
        locations = script_data.get('locations', [])
        
        if not scenes:
            scenes = [{"description": "A mysterious anime character appears in a magical forest setting.", "duration": 300}]
        
        if not characters:
            characters = [{"name": "Protagonist", "description": "Main anime character with mysterious powers"}]
        
        if not locations:
            locations = [{"name": "Magical Forest", "description": "Ethereal forest with glowing particles"}]
        
        print("Step 2: Expanding script with LLM...")
        if db_run and db:
            db_run.progress = 10.0
            db.commit()
        
        try:
            script_data['scenes'] = scenes
            script_data['characters'] = characters
            script_data['locations'] = locations
            
            expanded_script = self.expand_script_if_needed(script_data, min_duration=20.0)
            
            scenes = expanded_script.get('scenes', scenes)
            characters = expanded_script.get('characters', characters)
            locations = expanded_script.get('locations', locations)
            
            print(f"Anime script expanded to {len(scenes)} scenes for 20-minute target")
            
        except Exception as e:
            print(f"Error during anime script expansion: {e}")
        
        print("Step 3: Generating anime scenes with combat integration...")
        if db_run and db:
            db_run.progress = 20.0
            db.commit()
        
        scene_files = []
        for i, scene in enumerate(scenes):
            scene_text = scene if isinstance(scene, str) else scene.get('description', f'Scene {i+1}')
            scene_chars = [characters[i % len(characters)], characters[(i + 1) % len(characters)]]
            scene_location = locations[i % len(locations)]
            
            scene_type = self._detect_scene_type(scene_text)
            
            scene_detail = {
                "scene_number": i + 1,
                "description": scene_text,
                "characters": scene_chars,
                "location": scene_location,
                "scene_type": scene_type,
                "duration": scene.get('duration', 10.0) if isinstance(scene, dict) else 10.0
            }
            
            if scene_type == "combat" and self.combat_calls_count < self.max_combat_calls:
                try:
                    combat_data = self.generate_combat_scene(
                        scene_description=scene_text,
                        duration=10.0,
                        characters=scene_chars,
                        style="anime",
                        difficulty="medium"
                    )
                    scene_detail["combat_data"] = combat_data
                    self.combat_calls_count += 1
                    print(f"Generated anime combat scene {i+1} with choreography ({self.combat_calls_count}/{self.max_combat_calls})")
                except Exception as e:
                    print(f"Error generating anime combat scene: {e}")
            
            scene_file = scenes_dir / f"scene_{i+1:03d}.mp4"
            
            print(f"Generating anime scene {i+1}: {scene_text[:50]}...")
            
            try:
                char_names = ", ".join([c.get("name", "character") if isinstance(c, dict) else str(c) for c in scene_chars])
                location_desc = scene_location.get("description", scene_location.get("name", "location")) if isinstance(scene_location, dict) else str(scene_location)
                
                anime_prompt = f"anime scene, {location_desc}, with {char_names}, {scene_text}, detailed anime style, vibrant colors, high quality, 16:9 aspect ratio"
                
                if scene_detail.get("combat_data"):
                    anime_prompt = scene_detail["combat_data"]["video_prompt"]
                
                video_path = self.generate_video(
                    prompt=anime_prompt,
                    duration=scene_detail["duration"],
                    output_path=str(scene_file)
                )
                
                if video_path:
                    scene_files.append(video_path)
                    print(f"Generated anime scene video {i+1}")
                else:
                    print(f"Failed to generate video for scene {i+1}")
                    
            except Exception as e:
                print(f"Error generating scene {i}: {e}")
                fallback_path = self._create_fallback_video(scene_text, scene_detail["duration"], str(scene_file))
                if fallback_path:
                    scene_files.append(fallback_path)
            
            if db_run and db:
                db_run.progress = 20.0 + (i + 1) / len(scenes) * 30.0
                db.commit()
        
        print("Step 4: Generating voice lines...")
        if db_run and db:
            db_run.progress = 50.0
            db.commit()
        
        voice_files = []
        for i, scene in enumerate(scenes):
            scene_text = scene if isinstance(scene, str) else scene.get('description', f'Scene {i+1}')
            dialogue = scene.get('dialogue', scene_text) if isinstance(scene, dict) else scene_text
            
            voice_file = scenes_dir / f"voice_{i+1:03d}.wav"
            
            try:
                voice_path = self.generate_voice(
                    text=dialogue,
                    language=language,
                    output_path=str(voice_file)
                )
                
                if voice_path:
                    voice_files.append(voice_path)
                    print(f"Generated voice for scene {i+1}")
                    
            except Exception as e:
                print(f"Error generating voice for scene {i+1}: {e}")
        
        print("Step 5: Generating background music...")
        if db_run and db:
            db_run.progress = 60.0
            db.commit()
        
        music_file = final_dir / "background_music.wav"
        try:
            music_path = self.generate_background_music(
                prompt="anime background music, epic adventure soundtrack",
                duration=sum(scene.get('duration', 10.0) if isinstance(scene, dict) else 10.0 for scene in scenes),
                output_path=str(music_file)
            )
            print(f"Generated background music: {music_path}")
        except Exception as e:
            print(f"Error generating background music: {e}")
            music_path = None
        
        print("Step 6: Combining scenes into final episode...")
        if db_run and db:
            db_run.progress = 80.0
            db.commit()
        
        final_video = final_dir / "anime_episode.mp4"
        try:
            combined_path = self._combine_scenes_to_episode(
                scene_files=scene_files,
                voice_files=voice_files,
                music_path=music_path,
                output_path=str(final_video),
                render_fps=render_fps,
                output_fps=output_fps,
                frame_interpolation_enabled=frame_interpolation_enabled
            )
            print(f"Final anime episode created: {combined_path}")
        except Exception as e:
            print(f"Error combining scenes: {e}")
            combined_path = str(final_video)
        
        print("Step 7: Creating shorts...")
        if db_run and db:
            db_run.progress = 90.0
            db.commit()
        
        try:
            shorts_paths = self._create_shorts(scene_files, shorts_dir)
            print(f"Created {len(shorts_paths)} shorts")
        except Exception as e:
            print(f"Error creating shorts: {e}")
        
        if db_run and db:
            db_run.progress = 100.0
            db.commit()
        
        self.create_manifest(
            output_dir,
            scenes_generated=len(scene_files),
            combat_scenes=self.combat_calls_count,
            final_video=str(final_video),
            language=language,
            render_fps=render_fps,
            output_fps=output_fps
        )
        
        print(f"Anime pipeline completed successfully: {output_dir}")
        return str(output_dir)
    
    def _detect_scene_type(self, scene_text: str) -> str:
        """Detect scene type from description."""
        scene_lower = scene_text.lower()
        
        if any(word in scene_lower for word in ["fight", "battle", "combat", "attack", "sword", "punch", "kick"]):
            return "combat"
        elif any(word in scene_lower for word in ["dialogue", "talk", "conversation", "speak", "say"]):
            return "dialogue"
        elif any(word in scene_lower for word in ["action", "run", "chase", "escape", "jump"]):
            return "action"
        elif any(word in scene_lower for word in ["emotional", "cry", "sad", "happy", "love", "heart"]):
            return "emotional"
        else:
            return "dialogue"
    
    def _combine_scenes_to_episode(self, scene_files: List[str], voice_files: List[str], 
                                  music_path: Optional[str], output_path: str, 
                                  render_fps: int, output_fps: int, 
                                  frame_interpolation_enabled: bool) -> str:
        """Combine all scenes into final episode."""
        try:
            import cv2
            import numpy as np
            
            if not scene_files:
                return self._create_fallback_video("No scenes generated", 1200, output_path)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, output_fps, (1920, 1080))
            
            total_frames = 0
            for scene_file in scene_files:
                try:
                    cap = cv2.VideoCapture(scene_file)
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        if frame.shape[:2] != (1080, 1920):
                            frame = cv2.resize(frame, (1920, 1080))
                        
                        out.write(frame)
                        total_frames += 1
                    cap.release()
                except Exception as e:
                    print(f"Error processing scene file {scene_file}: {e}")
            
            out.release()
            
            if total_frames > 0:
                print(f"Combined {len(scene_files)} scenes into {total_frames} frames")
                return output_path
            else:
                return self._create_fallback_video("Scene combination failed", 1200, output_path)
                
        except Exception as e:
            print(f"Error in scene combination: {e}")
            return self._create_fallback_video("Scene combination error", 1200, output_path)
    
    def _create_shorts(self, scene_files: List[str], shorts_dir: Path) -> List[str]:
        """Create short clips from scenes."""
        shorts_paths = []
        
        for i, scene_file in enumerate(scene_files[:3]):
            try:
                short_path = shorts_dir / f"short_{i+1:03d}.mp4"
                
                import cv2
                cap = cv2.VideoCapture(scene_file)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(short_path), fourcc, 24, (1080, 1920))
                
                frame_count = 0
                max_frames = 24 * 15
                
                while frame_count < max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame = cv2.resize(frame, (1080, 1920))
                    out.write(frame)
                    frame_count += 1
                
                cap.release()
                out.release()
                
                if frame_count > 0:
                    shorts_paths.append(str(short_path))
                    
            except Exception as e:
                print(f"Error creating short {i+1}: {e}")
        
        return shorts_paths


def run(input_path: str, output_path: str, base_model: str = "stable_diffusion_1_5", 
        lora_models: Optional[List[str]] = None, lora_paths: Optional[Dict[str, str]] = None, 
        db_run=None, db=None, render_fps: int = 24, output_fps: int = 24, 
        frame_interpolation_enabled: bool = True, language: str = "en") -> str:
    """Run anime pipeline with self-contained processing."""
    pipeline = AnimePipeline()
    return pipeline.run(
        input_path=input_path,
        output_path=output_path,
        base_model=base_model,
        lora_models=lora_models,
        lora_paths=lora_paths,
        db_run=db_run,
        db=db,
        render_fps=render_fps,
        output_fps=output_fps,
        frame_interpolation_enabled=frame_interpolation_enabled,
        language=language
    )
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    output_dir = Path(output_path)
    scenes_dir = output_dir / "scenes"
    characters_dir = output_dir / "characters"
    final_dir = output_dir / "final"
    shorts_dir = output_dir / "shorts"
    
    for dir_path in [output_dir, scenes_dir, characters_dir, final_dir, shorts_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    character_memory = None
    project_id = str(output_dir.name)
    
    if db_run and db:
        db_run.progress = 10.0
        db.commit()
    
    print("Step 1: Reading YAML script...")
    
    scenes = []
    characters = []
    locations = []
    
    if input_path and os.path.exists(input_path):
        try:
            if input_path.endswith('.yaml') or input_path.endswith('.yml'):
                with open(input_path, 'r', encoding='utf-8') as f:
                    script_data = yaml.safe_load(f)
                    if isinstance(script_data, dict):
                        scenes = script_data.get('scenes', [])
                        characters = script_data.get('characters', [])
                        locations = script_data.get('locations', [])
            elif input_path.endswith('.json'):
                with open(input_path, 'r', encoding='utf-8') as f:
                    script_data = json.load(f)
                    if isinstance(script_data, dict):
                        scenes = script_data.get('scenes', [])
                        characters = script_data.get('characters', [])
                        locations = script_data.get('locations', [])
            elif input_path.endswith('.txt'):
                with open(input_path, 'r', encoding='utf-8') as f:
                    scenes = [scene.strip() for scene in f.read().split('\n\n') if scene.strip()]
            else:
                print(f"Using {input_path} as single scene description")
                with open(input_path, 'r', encoding='utf-8') as f:
                    scenes = [f.read().strip()]
        except Exception as e:
            print(f"Error parsing input script: {e}")
            scenes = []
    
    if script_data and isinstance(script_data, dict):
        try:
            from ..script_expander import expand_script_if_needed
            from ..ai_models import load_llm
            
            pass
            
            llm_model = load_llm()
            expanded_script = expand_script_if_needed(script_data, min_duration=20.0, llm_model=llm_model)
            
            if expanded_script != script_data:
                print(f"Script expanded from {len(script_data.get('scenes', []))} to {len(expanded_script.get('scenes', []))} scenes")
                scenes = expanded_script.get('scenes', scenes)
                characters = expanded_script.get('characters', characters) 
                locations = expanded_script.get('locations', locations)
        except Exception as e:
            print(f"Error during anime script expansion: {e}")
                
    scene_details = []
    for i, scene in enumerate(scenes):
        if isinstance(scene, str):
            scene_chars = [characters[i % len(characters)], characters[(i + 1) % len(characters)]]
            scene_location = locations[i % len(locations)]
            
            from ..pipeline_utils import detect_scene_type
            scene_type = detect_scene_type(scene)
            
            scene_detail = {
                "scene_text": scene,
                "scene_type": scene_type,
                "characters": scene_chars,
                "location": scene_location
            }
            
            if scene_type == "combat":
                try:
                    from ..combat_scene_generator import generate_combat_scene
                    combat_data = generate_combat_scene(
                        scene_description=scene,
                        duration=10.0,
                        characters=scene_chars,
                        style="anime",
                        difficulty="medium"
                    )
                    scene_detail["combat_data"] = combat_data
                    scene_detail["scene_text"] = combat_data["video_prompt"]
                    print(f"Generated anime combat choreography for scene {i+1}")
                except Exception as e:
                    print(f"Error generating anime combat scene: {e}")
            
            
            from ..pipeline_utils import detect_scene_type
            scene_type = detect_scene_type(scene)
            
            scene_detail = {
                "scene_text": scene,
                "scene_type": scene_type,
                "characters": scene_chars,
                "location": scene_location
            }
            
            if scene_type == "combat":
                try:
                    from ..combat_scene_generator import generate_combat_scene
                    combat_data = generate_combat_scene(
                        scene_description=scene,
                        duration=10.0,
                        characters=scene_chars,
                        style="anime",
                        difficulty="medium"
                    )
                    scene_detail["combat_data"] = combat_data
                    scene_detail["scene_text"] = combat_data["video_prompt"]
                    print(f"Generated anime combat choreography for scene {i+1}")
                except Exception as e:
                    print(f"Error generating anime combat scene: {e}")
            
            scene_details.append(scene_detail)
        else:
            scene_details.append(scene)
    
    print(f"Processing {len(scene_details)} scenes with {len(characters)} characters across {len(locations)} locations")

    if not scenes:
        scenes = [
            "Anime school scene with cherry blossoms and students",
            "Anime battle scene with magical effects and dramatic poses",
            "Anime emotional scene with two characters under night sky",
            "Anime fantasy landscape with magical creatures and vibrant colors",
            "Anime slice of life scene in a cozy cafe with detailed characters"
        ]
    
    if not characters:
        characters = [
            {"name": "Yuki", "description": "Female protagonist with long blue hair and school uniform", "voice": "female_young"},
            {"name": "Hiro", "description": "Male protagonist with spiky black hair and casual outfit", "voice": "male_young"},
            {"name": "Sensei", "description": "Older mentor character with glasses and formal attire", "voice": "male_mature"}
        ]
    
    if not locations:
        locations = [
            "High school campus with cherry blossoms",
            "Magical forest with glowing elements",
            "Futuristic city with neon lights",
            "Traditional Japanese temple with garden"
        ]
    
    scene_details = []
    for i, scene in enumerate(scenes):
        if isinstance(scene, str):
            scene_chars = [characters[i % len(characters)], characters[(i + 1) % len(characters)]]
            scene_location = locations[i % len(locations)]
            
            from ..pipeline_utils import detect_scene_type
            scene_type = detect_scene_type(scene)
            
            scene_detail = {
                "scene_text": scene,
                "scene_type": scene_type,
                "characters": scene_chars,
                "location": scene_location
            }
            
            if scene_type == "combat":
                try:
                    from ..combat_scene_generator import generate_combat_scene
                    combat_data = generate_combat_scene(
                        scene_description=scene,
                        duration=10.0,
                        characters=scene_chars,
                        style="anime",
                        difficulty="medium"
                    )
                    scene_detail["combat_data"] = combat_data
                    scene_detail["scene_text"] = combat_data["video_prompt"]
                    print(f"Generated anime combat choreography for scene {i+1}")
                except Exception as e:
                    print(f"Error generating anime combat scene: {e}")
            
            scene_details.append(scene_detail)
        else:
            scene_details.append(scene)
    
    print(f"Processing {len(scene_details)} scenes with {len(characters)} characters across {len(locations)} locations")
    
    if db_run and db:
        db_run.progress = 20.0
        db.commit()
    
    print("Step 2: Determining characters and locations per scene...")
    
    if db_run and db:
        db_run.progress = 30.0
        db.commit()
    
    print("Step 3: Loading character references and voice profiles...")
    
    try:
        anime_model = None
        print(f"Successfully loaded {base_model} with {lora_models} LoRA(s)")
    except Exception as e:
        print(f"Error loading models: {e}")
        anime_model = None
    
    if db_run and db:
        db_run.progress = 40.0
        db.commit()
    
    character_seeds = {}
    character_ids = {}
    
    for character in characters:
        character_name = character.get("name", "Unknown") if isinstance(character, dict) else str(character)
        character_desc = character.get("description", "") if isinstance(character, dict) else ""
        character_voice = character.get("voice", "default") if isinstance(character, dict) else "default"
        
        print(f"Processing character: {character_name}")
        
        existing_char = character_memory.get_character_by_name(character_name, project_id)
        
        if existing_char:
            print(f"Using existing character design for: {character_name}")
            character_id = existing_char["character_id"]
            seed = character_memory.get_character_seed(character_id)
            if seed is None:
                import hashlib
                seed = int(hashlib.md5(character_name.encode()).hexdigest()[:8], 16) % (2**32)
                character_memory.set_character_seed(character_id, seed)
        else:
            print(f"Creating new character design for: {character_name}")
            character_id = character_memory.register_character(
                name=character_name,
                description=character_desc,
                voice_profile=character_voice,
                project_id=project_id
            )
            
            import hashlib
            seed = int(hashlib.md5(character_name.encode()).hexdigest()[:8], 16) % (2**32)
            character_memory.set_character_seed(character_id, seed)
        
        character_seeds[character_name] = seed
        character_ids[character_name] = character_id
        
        angles = ["front_view", "side_view", "three_quarter_view"]
        
        for angle in angles:
            char_file = characters_dir / f"{character_name.lower().replace(' ', '_')}_{angle}.png"
            
            character_id = character_ids[character_name]
            existing_refs = character_memory.get_character_reference_images(character_id)
            existing_angle = next((ref for ref in existing_refs if ref["angle"] == angle), None)
            
            if existing_angle and Path(existing_angle["path"]).exists():
                print(f"Using existing character reference for {character_name} {angle}")
                import shutil
                shutil.copy2(existing_angle["path"], char_file)
                continue
            
            try:
                if anime_model:
                    angle_desc = angle.replace('_', ' ')
                    char_desc = character.get("description", "") if isinstance(character, dict) else ""
                    
                    generation_params = {
                        "prompt": f"anime character {character_name}, {char_desc}, {angle_desc}, detailed face, consistent design, high quality, masterpiece",
                        "width": 512,
                        "height": 512,
                        "seed": character_seeds[character_name]
                    }
                    
                    generation_params = character_memory.ensure_character_consistency(character_id, generation_params)
                    
                    result = None
                    if result and hasattr(result, "images") and result.images:
                        result.images[0].save(char_file)
                        print(f"Generated character image: {char_file}")
                        
                        character_memory.save_character_reference(character_id, str(char_file), angle)
                    else:
                        print(f"Failed to generate character {character_name} {angle}")
                        create_error_image(str(char_file), f"Character: {character_name}")
                else:
                    print(f"No model available for character {character_name}")
                    create_error_image(str(char_file), f"Character: {character_name}")
                    
            except Exception as e:
                print(f"Error generating character {character_name} {angle}: {e}")
                create_error_image(str(char_file), f"Error: {character_name}")
        
        main_char_file = characters_dir / f"{character_name.lower().replace(' ', '_')}.png"
        front_view_file = characters_dir / f"{character_name.lower().replace(' ', '_')}_front_view.png"
        
        try:
            if os.path.exists(front_view_file):
                import shutil
                shutil.copy2(front_view_file, main_char_file)
                print(f"Created main character file: {main_char_file}")
        except Exception as e:
            print(f"Error creating main character file for {character_name}: {e}")
    
    if db_run and db:
        db_run.progress = 50.0
        db.commit()
    
    print("Step 4: Generating visuals with SD Anime Model + Anime LoRA...")
    
    for i, scene_detail in enumerate(scene_details):
        scene_text = scene_detail.get("scene_text") or scene_detail.get("description", "")
        scene_location = scene_detail.get("location", "")
        scene_chars = scene_detail.get("characters", [])
        
        print(f"Generating scene {i+1}: {scene_text[:50]}...")
        
        char_names = ", ".join([c.get("name", "character") if isinstance(c, dict) else str(c) for c in scene_chars])
        anime_prompt = f"anime scene, {scene_location}, with {char_names}, {scene_text}, detailed style, vibrant colors"
        
        scene_file = scenes_dir / f"scene_{i+1:03d}.png"
        try:
            if anime_model:
                result = None
                if result and hasattr(result, "images") and result.images:
                    result.images[0].save(scene_file)
                    print(f"Successfully generated scene {i+1} image")
                else:
                    print(f"Failed to generate scene {i+1} image: No valid result")
                    create_error_image(str(scene_file), f"Scene {i+1}: {scene_text}")
            else:
                print(f"No model available for scene {i+1}")
                create_error_image(str(scene_file), f"Scene {i+1}: {scene_text}")
        except Exception as e:
            print(f"Error generating scene {i+1}: {e}")
            create_error_image(str(scene_file), f"Scene {i+1}: {scene_text}")
    
    if db_run and db:
        db_run.progress = 60.0
        db.commit()
    
    print("Step 5: Adding animation via text-to-video generation...")
    if db_run and db:
        db_run.progress = 60.0
        db.commit()
    
    for i, scene_detail in enumerate(scene_details):
        scene_text = scene_detail.get("scene_text") or scene_detail.get("description", "")
        scene_location = scene_detail.get("location", "")
        scene_chars = scene_detail.get("characters", [])
        
        print(f"Generating professional anime video for scene {i+1}: {scene_text[:50]}...")
        
        animated_file = scenes_dir / f"scene_{i+1:03d}_anime_hq.mp4"
        voice_file = scenes_dir / f"scene_{i+1:03d}_voice.wav"
        music_file = scenes_dir / f"scene_{i+1:03d}_music.wav"
        final_file = scenes_dir / f"scene_{i+1:03d}_final.mp4"
        
        try:
            from ..pipeline_utils import create_scene_video_with_generation, optimize_video_prompt, generate_voice_lines, generate_background_music, apply_lipsync, create_fallback_video
            from ..video_generation import get_best_model_for_content
            from ..ai_models import AIModelManager
            
            model_manager = AIModelManager()
            vram_tier = model_manager._detect_vram_tier()
            
            optimized_prompt = optimize_video_prompt(scene_text, "anime")
            
            if scene_detail.get("scene_type") == "combat" and scene_detail.get("combat_data"):
                combat_data = scene_detail["combat_data"]
                combat_type = combat_data.get("combat_type", "melee")
                from ..video_generation import get_best_model_for_combat
                best_model = get_best_model_for_combat("anime", vram_tier, combat_type)
                optimized_prompt = combat_data.get("video_prompt", optimized_prompt)
            else:
                best_model = get_best_model_for_content("anime", vram_tier)
            
            success = create_scene_video_with_generation(
                scene_description=optimized_prompt,
                characters=scene_chars,
                output_path=str(animated_file),
                model_name=best_model
            )
            
            if success:
                print(f"Successfully generated high-quality anime video for scene {i+1} using {best_model}")
                
                character_voice = scene_chars[0].get("voice", "default") if scene_chars else "default"
                voice_success = generate_voice_lines(scene_text, character_voice, str(voice_file))
                
                music_success = generate_background_music(scene_text, 10.0, str(music_file))
                
                if voice_success:
                    lipsync_success = apply_lipsync(str(animated_file), str(voice_file), str(final_file), "anime")
                    if lipsync_success:
                        print(f"Applied lipsync for scene {i+1}")
                
            else:
                print(f"Failed to generate video for scene {i+1}, creating professional fallback")
                create_fallback_video(animated_file, scene_text, i+1, (1920, 1080))
                
        except Exception as e:
            print(f"Error generating video for scene {i+1}: {e}")
            create_fallback_video(animated_file, scene_text, i+1, (1920, 1080))
    
    print("Step 6: Generating voice-over via RVC/Bark per character...")
    
    try:
        bark_model = None
        print("Bark model loaded successfully")
    except Exception as e:
        print(f"Error loading Bark model: {e}")
        bark_model = None
    
    for i, scene_detail in enumerate(scene_details):
        scene_chars = scene_detail["characters"]
        
        for j, character in enumerate(scene_chars):
            char_name = character.get("name", f"character_{j}")
            
            print(f"Generating voice-over for {char_name} in scene {i+1}")
            
            voice_file = scenes_dir / f"scene_{i+1:03d}_{char_name.lower().replace(' ', '_')}.wav"
            
            try:
                if bark_model and bark_model.get("loaded", False):
                    try:
                        from bark import generate_audio, SAMPLE_RATE
                        import numpy as np
                        from scipy.io.wavfile import write as write_wav
                        
                        voice_type = character.get("voice", "neutral")
                        speaker_map = {
                            "female_young": "v2/en_speaker_6",
                            "male_young": "v2/en_speaker_9",
                            "female_mature": "v2/en_speaker_5",
                            "male_mature": "v2/en_speaker_0",
                            "neutral": "v2/en_speaker_3"
                        }
                        
                        speaker = speaker_map.get(voice_type, "v2/en_speaker_3")
                        text = f"Voice line for {char_name} in scene {i+1}"
                        
                        audio_array = generate_audio(text, history_prompt=speaker)
                        
                        write_wav(voice_file, SAMPLE_RATE, audio_array)
                        print(f"Generated voice-over for {char_name} using Bark")
                        
                    except Exception as e:
                        print(f"Error generating audio with Bark: {e}")
                        create_silent_audio(str(voice_file))
                else:
                    print(f"Bark model not properly loaded for {char_name}")
                    create_silent_audio(str(voice_file))
                    
            except Exception as e:
                print(f"Error in voice generation for {char_name}: {e}")
                create_silent_audio(str(voice_file))
    
    print("Step 7: Performing lipsync via SadTalker...")
    
    if db_run and db:
        db_run.progress = 70.0
        db.commit()
    
    for i, scene_detail in enumerate(scene_details):
        scene_chars = scene_detail["characters"]
        
        for j, character in enumerate(scene_chars):
            char_name = character.get("name", f"character_{j}")
            
            lipsync_file = scenes_dir / f"scene_{i+1:03d}_{char_name.lower().replace(' ', '_')}_lipsync.mp4"
            print(f"Performing lipsync for {char_name} in scene {i+1}")
            
            try:
                char_img = characters_dir / f"{char_name.lower().replace(' ', '_')}.png"
                voice_file = scenes_dir / f"scene_{i+1:03d}_{char_name.lower().replace(' ', '_')}.wav"
                
                if os.path.exists(char_img) and os.path.exists(voice_file):
                    create_lipsync_video(str(char_img), str(voice_file), str(lipsync_file))
                    print(f"Created lipsync for {char_name} in scene {i+1}")
                else:
                    if not os.path.exists(char_img):
                        print(f"Character image for {char_name} not found")
                    if not os.path.exists(voice_file):
                        print(f"Voice file for {char_name} in scene {i+1} not found")
                    print(f"Skipping lipsync for {char_name} in scene {i+1} - missing required files")
                    
            except Exception as e:
                print(f"Error creating lipsync for {char_name} in scene {i+1}: {e}")
                print(f"Lipsync processing will continue with next character")
    
    print("Step 8: Adding Japanese music elements...")
    
    if db_run and db:
        db_run.progress = 80.0
        db.commit()
    
    try:
        music_file = output_dir / "background_music.wav"
        create_silent_audio(str(music_file), duration=30.0)
        print(f"Background music created: {music_file}")
        
    except Exception as e:
        print(f"Error generating background music: {e}")
    
    print("Step 9: Creating MP4 per scene...")
    
    scene_videos_created = []
    
    for i in range(1, len(scene_details) + 1):
        animated_file = scenes_dir / f"scene_{i:03d}_animated.mp4"
        
        if os.path.exists(animated_file):
            scene_videos_created.append(str(animated_file))
            print(f"Using animated scene MP4: {animated_file}")
        else:
            scene_file = scenes_dir / f"scene_{i:03d}.png"
            scene_mp4 = scenes_dir / f"scene_{i:03d}.mp4"
            
            if os.path.exists(scene_file):
                try:
                    create_scene_video(str(scene_file), str(scene_mp4), duration=8.0)
                    scene_videos_created.append(str(scene_mp4))
                    print(f"Created scene MP4: {scene_mp4}")
                except Exception as e:
                    print(f"Error creating MP4 for scene {i}: {e}")
            else:
                print(f"Scene file {scene_file} not found, skipping MP4 creation")
    
    print(f"Successfully created {len(scene_videos_created)} scene videos")
    
    if db_run and db:
        db_run.progress = 90.0
        db.commit()
    
    print("Step 10: Combining scenes into full episode...")
    
    final_episode = final_dir / "full_episode.mp4"
    try:
        combine_scenes_to_episode(scenes_dir, str(final_episode))
        print(f"Full episode created: {final_episode}")
    except Exception as e:
        print(f"Error creating full episode: {e}")
    
    print("Step 11: Creating shorts and generating metadata...")
    
    try:
        create_shorts(scenes_dir, shorts_dir, num_shorts=5)
        print("Shorts created successfully")
    except Exception as e:
        print(f"Error creating shorts: {e}")
    
    if db_run and db:
        db_run.progress = 100.0
        db_run.status = "completed"
        db.commit()
    
    print("AI Original Anime Series Channel pipeline completed!")
    print(f"Output directory: {output_dir}")
    print(f"Scenes: {scenes_dir}")
    print(f"Characters: {characters_dir}")
    print(f"Final episode: {final_dir}")
    print(f"Shorts: {shorts_dir}")
    
    return str(output_dir)


def create_silent_audio(file_path: str, duration: float = 3.0, sample_rate: int = 22050):
    """Create a silent audio file as a fallback."""
    try:
        import numpy as np
        from scipy.io.wavfile import write as write_wav
        
        audio_array = np.zeros(int(duration * sample_rate))
        write_wav(file_path, sample_rate, audio_array)
        print(f"Created silent audio file: {file_path}")
        
    except Exception as e:
        print(f"Error creating silent audio: {e}")
        with open(file_path, "wb") as f:
            f.write(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00")


def create_error_image(file_path: str, text: str):
    """Create an error placeholder image."""
    try:
        if Image and ImageDraw:
            img = Image.new('RGB', (512, 512), color='red')
            draw = ImageDraw.Draw(img)
            
            try:
                font = ImageFont.load_default() if ImageFont else None
            except:
                font = None
                
            draw.text((50, 250), f"Error: {text}", fill='white', font=font)
            img.save(file_path)
        print(f"Created error image: {file_path}")
        
    except Exception as e:
        print(f"Failed to create error image: {e}")
        with open(file_path, "wb") as f:
            f.write(b"Error")


def create_static_video(image_path: str, video_path: str, duration: float = 5.0):
    """Create a static video from an image with anime-quality settings."""
    try:
        from moviepy.editor import ImageClip
        
        clip = ImageClip(image_path, duration=duration)
        clip.write_videofile(
            video_path, 
            fps=24,
            codec='libx264',
            bitrate='12000k',
            audio_codec='aac',
            verbose=False, 
            logger=None,
            preset='veryslow',
            ffmpeg_params=['-crf', '15', '-profile:v', 'high', '-level', '4.1']
        )
        clip.close()
        
    except Exception as e:
        print(f"Error creating static video: {e}")


def create_lipsync_video(char_img_path: str, voice_path: str, output_path: str):
    """Create a lipsync video with anime-quality settings."""
    try:
        from moviepy.editor import ImageClip, AudioFileClip
        
        img_clip = ImageClip(char_img_path)
        audio_clip = AudioFileClip(voice_path)
        
        img_clip = img_clip.set_duration(audio_clip.duration)
        video_clip = img_clip.set_audio(audio_clip)
        
        video_clip.write_videofile(
            output_path, 
            fps=24,
            codec='libx264',
            bitrate='12000k',
            audio_codec='aac',
            verbose=False, 
            logger=None,
            preset='veryslow',
            ffmpeg_params=['-crf', '15', '-profile:v', 'high', '-level', '4.1']
        )
        
        img_clip.close()
        audio_clip.close()
        video_clip.close()
        
    except Exception as e:
        print(f"Error creating lipsync video: {e}")


def create_scene_video(scene_img_path: str, output_path: str, duration: float = 5.0):
    """Create a scene video with anime-quality settings."""
    try:
        from moviepy.editor import ImageClip
        
        clip = ImageClip(scene_img_path, duration=duration)
        clip.write_videofile(
            output_path, 
            fps=24,
            codec='libx264',
            bitrate='12000k',
            audio_codec='aac',
            verbose=False, 
            logger=None,
            preset='veryslow',
            ffmpeg_params=['-crf', '15', '-profile:v', 'high', '-level', '4.1']
        )
        clip.close()
        
    except Exception as e:
        print(f"Error creating scene video: {e}")


def combine_scenes_to_episode(scenes_dir: Path, output_path: str, frame_interpolation_enabled: bool = True, render_fps: int = 24, output_fps: int = 24):
    """Combine scene videos into a full episode."""
    try:
        from moviepy.editor import VideoFileClip, concatenate_videoclips
        import glob
        
        scene_files = sorted(glob.glob(str(scenes_dir / "scene_*.mp4")))
        
        if scene_files:
            clips = [VideoFileClip(f) for f in scene_files]
            final_video = concatenate_videoclips(clips)
            temp_output = output_path.replace('.mp4', '_temp.mp4') if frame_interpolation_enabled and output_fps > render_fps else output_path
            final_video.write_videofile(
                temp_output, 
                fps=render_fps, 
                codec='libx264',
                bitrate='12000k',
                preset='veryslow',
                ffmpeg_params=['-crf', '15', '-profile:v', 'high', '-level', '4.1'],
                verbose=False, 
                logger=None
            )
            
            for clip in clips:
                clip.close()
            final_video.close()
            
            if frame_interpolation_enabled and output_fps > render_fps:
                from ..frame_interpolation import FrameInterpolator
                from ..ai_models import AIModelManager
                
                model_manager = AIModelManager()
                vram_tier = model_manager._detect_vram_tier()
                
                interpolator = FrameInterpolator(vram_tier)
                if interpolator.interpolate_video(temp_output, output_path, render_fps, output_fps):
                    import os
                    os.remove(temp_output)
                    logger.info(f"Frame interpolation completed: {render_fps}fps -> {output_fps}fps")
                else:
                    import os
                    os.rename(temp_output, output_path)
                    logger.warning("Frame interpolation failed, using original video")
        else:
            print("No scene videos found to combine")
            
    except Exception as e:
        print(f"Error combining scenes: {e}")


def create_shorts(scenes_dir: Path, shorts_dir: Path, num_shorts: int = 5, render_fps: int = 24):
    """Create short clips from scenes."""
    try:
        from moviepy.editor import VideoFileClip
        import glob
        
        scene_files = sorted(glob.glob(str(scenes_dir / "scene_*.mp4")))
        
        for i, scene_file in enumerate(scene_files[:num_shorts]):
            short_path = shorts_dir / f"short_{i+1:02d}.mp4"
            
            clip = VideoFileClip(scene_file)
            short_clip = clip.subclip(0, min(20, clip.duration))
            short_clip.write_videofile(
                str(short_path), 
                fps=render_fps, 
                codec='libx264',
                bitrate='12000k',
                preset='veryslow',
                ffmpeg_params=['-crf', '15', '-profile:v', 'high', '-level', '4.1'],
                verbose=False, 
                logger=None
            )
            
            clip.close()
            short_clip.close()
            
    except Exception as e:
        print(f"Error creating shorts: {e}")
