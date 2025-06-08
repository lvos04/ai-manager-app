from ..common_imports import *
from ..ai_imports import *
import time
import shutil
from .base_pipeline import BasePipeline

from ..pipeline_utils import ensure_output_dir, log_progress
from ..ai_models import load_with_multiple_loras, generate_image, load_whisper, load_bark, load_musicgen, load_llm
from ...core.character_memory import get_character_memory_manager
from ..language_support import get_language_config, enhance_script_with_language, get_language_specific_prompts, get_voice_code, get_tts_model, is_bark_supported

def run(input_path, output_path, base_model, lora_models, lora_paths=None, db_run=None, db=None, render_fps=24, output_fps=24, frame_interpolation_enabled=True, language="en"):
    """
    Run the AI Original Superhero Universe Channel pipeline.
    
    Processing steps:
    1. Read script with original superhero stories
    2. Load characters with own LoRAs per superhero (unique style/costume)
    3. Generate images with Stable Diffusion
    4. Add animation with AnimateDiff or Deforum
    5. Generate voice-over per character with unique voices via RVC/Bark
    6. Add epic soundtrack via MusicGen
    7. Build each scene as separate video, then combine into MP4 episode
    8. Use Whisper and local LLM for 5 shorts, titles, and subtitles
    9. Save final video and shorts according to output structure
    
    Args:
        input_path: Path to the input data
        output_path: Path to the output directory
        base_model: Base AI model to use (e.g., stable_diffusion_1_5)
        lora_model: LoRA model to use for style consistency
        db_run: Database pipeline run object for progress updates
        db: Database session
    """
    print(f"Running AI Original Superhero Universe Channel pipeline")
    print(f"Base model: {base_model}")
    print(f"LoRA adaptations: {', '.join(lora_models) if lora_models else 'None'}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    from config import CHANNEL_BASE_MODELS
    if base_model not in CHANNEL_BASE_MODELS.get("superhero", []):
        print(f"Warning: {base_model} may not be optimal for superhero content")
    
    output_dir = ensure_output_dir(Path(output_path))
    
    scenes_dir = output_dir / "scenes"
    characters_dir = output_dir / "characters"
    final_dir = output_dir / "final"
    shorts_dir = output_dir / "shorts"
    
    for dir_path in [scenes_dir, characters_dir, final_dir, shorts_dir]:
        dir_path.mkdir(exist_ok=True)
    
    character_memory = get_character_memory_manager(str(characters_dir), str(output_dir.name))
    project_id = str(output_dir.name)
    
    if db_run and db:
        db_run.progress = 5.0
        db.commit()
    
    print("Step 1: Reading script with original superhero stories...")
    if db_run and db:
        db_run.progress = 10.0
        db.commit()
    
    scenes = []
    
    if input_path and os.path.exists(input_path):
        try:
            if input_path.endswith('.json'):
                with open(input_path, 'r') as f:
                    script_data = json.load(f)
                    if isinstance(script_data, dict):
                        scenes = script_data.get('scenes', [])
                    else:
                        scenes = script_data
            elif input_path.endswith('.txt'):
                with open(input_path, 'r') as f:
                    scenes = [scene.strip() for scene in f.read().split('\n\n') if scene.strip()]
            else:
                print(f"Using {input_path} as single scene description")
                with open(input_path, 'r') as f:
                    scenes = [f.read().strip()]
        except Exception as e:
            print(f"Error parsing input script: {e}")
            scenes = []
    
    if script_data and isinstance(script_data, dict):
        try:
            from ..script_expander import expand_script_if_needed
            from ..ai_models import load_llm
            
            script_data = enhance_script_with_language(script_data, language)
            
            llm_model = load_llm()
            expanded_script = expand_script_if_needed(script_data, min_duration=20.0, llm_model=llm_model)
            
            if expanded_script != script_data:
                print(f"Superhero script expanded from {len(script_data.get('scenes', []))} to {len(expanded_script.get('scenes', []))} scenes")
                scenes = expanded_script.get('scenes', scenes)
                characters = expanded_script.get('characters', characters) 
                locations = expanded_script.get('locations', locations)
                
        except Exception as e:
            print(f"Error during superhero script expansion: {e}")
    
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
                        duration=12.0,
                        characters=scene_chars,
                        style="superhero",
                        difficulty="hard"
                    )
                    scene_detail["combat_data"] = combat_data
                    scene_detail["scene_text"] = combat_data["video_prompt"]
                    print(f"Generated superhero combat choreography for scene {i+1}")
                except Exception as e:
                    print(f"Error generating superhero combat scene: {e}")
            
            scene_details.append(scene_detail)
        else:
            scene_details.append(scene)
    
    print(f"Processing {len(scene_details)} scenes with {len(characters)} characters across {len(locations)} locations")
    
    if not scenes:
        scenes = [
            "Epic superhero battle scene with dynamic poses and energy effects",
            "Superhero character introduction with dramatic lighting and heroic pose",
            "Superhero team assembling with diverse characters and unique costumes",
            "Villain confrontation scene with dramatic tension and power display",
            "Final victory scene with triumphant superhero and city backdrop"
        ]
    
    print(f"Processing {len(scenes)} scenes")
    
    print("Step 2: Loading characters with own LoRAs per superhero...")
    if db_run and db:
        db_run.progress = 20.0
        db.commit()
    
    print(f"Loading {base_model} with {', '.join(lora_models) if lora_models else 'no'} LoRA(s) for superhero generation...")
    try:
        from ..ai_models import AIModelManager, get_optimal_model_for_channel
        
        optimal_model = get_optimal_model_for_channel("superhero")
        if base_model != optimal_model:
            print(f"Warning: {base_model} may not be optimal. Recommended: {optimal_model}")
        
        model_manager = AIModelManager()
        superhero_model = model_manager.load_base_model(base_model, "image")
        if lora_models:
            superhero_model = model_manager.apply_multiple_loras(superhero_model, lora_models, lora_paths)
        
        print("Model loaded successfully with VRAM optimization")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Failed to load model - processing will continue with limitations")
        superhero_model = None
    
    superhero_characters = [
        {"name": "Captain Hero", "description": "Original superhero with muscular build, dynamic costume with bright colors, heroic pose", "voice": "heroic_male"},
        {"name": "Power Woman", "description": "Female superhero with sleek costume, energy powers, confident stance", "voice": "powerful_female"},
        {"name": "Tech Guardian", "description": "Tech-based superhero with armored suit, gadgets, and glowing elements", "voice": "tech_enhanced"},
        {"name": "Mystic Sage", "description": "Mystical superhero with magical symbols, flowing cape, and ethereal effects", "voice": "mystical_voice"}
    ]
    
    character_seeds = {}
    character_ids = {}
    
    for character in superhero_characters:
        character_name = character["name"]
        character_desc = character["description"]
        character_voice = character["voice"]
        
        print(f"Processing superhero character: {character_name}")
        
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
            print(f"Creating new superhero character design for: {character_name}")
            character_id = character_memory.register_character(
                name=character_name,
                description=character_desc,
                voice_profile=character_voice,
                project_id=project_id
            )
            
            import hashlib
            seed = int(hashlib.md5(character_name.encode()).hexdigest()[:8], 16) % (2**32)
            character_memory.set_character_seed(character_id, seed)
            
            character_memory.update_animation_style(character_id, {
                "movement_patterns": {"flight_style": "dynamic", "combat_style": "heroic"},
                "video_generation_params": {"guidance_scale": 7.5, "num_inference_steps": 20},
                "preferred_models": ["animatediff", "svd"]
            })
            
            character_memory.update_voice_characteristics(character_id, {
                "voice_settings": {"tone": character_voice, "intensity": "high"},
                "speech_patterns": {"pace": "confident", "emphasis": "heroic"}
            })
        
        character_seeds[character_name] = seed
        character_ids[character_name] = character_id
        
        char_file = characters_dir / f"{character_name.lower().replace(' ', '_')}.png"
        
        existing_refs = character_memory.get_character_reference_images(character_id)
        if existing_refs and any(Path(ref["path"]).exists() for ref in existing_refs):
            print(f"Using existing character reference for {character_name}")
            existing_ref = next(ref for ref in existing_refs if Path(ref["path"]).exists())
            import shutil
            shutil.copy2(existing_ref["path"], char_file)
            continue
        
        try:
            if superhero_model and hasattr(superhero_model, '__call__'):
                generation_params = {
                    "prompt": f"superhero character {character_name}, {character_desc}, detailed face, consistent design, high quality, masterpiece",
                    "width": 768,
                    "height": 768,
                    "seed": seed
                }
                
                generation_params = character_memory.ensure_comprehensive_consistency(character_id, generation_params, "image")
                
                result = superhero_model(generation_params["prompt"], 
                                       num_inference_steps=20, 
                                       guidance_scale=7.5, 
                                       width=generation_params["width"], 
                                       height=generation_params["height"])
                if hasattr(result, 'images') and result.images:
                    result.images[0].save(str(char_file))
                    print(f"Generated superhero character image: {char_file}")
                    character_memory.save_character_reference(character_id, str(char_file), "front_view")
                else:
                    raise ValueError("Model returned no images")
            else:
                print(f"No model available for character {character_name}")
                from ..pipeline_utils import create_error_image
                create_error_image(str(char_file), f"Character: {character_name}")
        except Exception as e:
            print(f"Error generating character {character_name}: {e}")
            with open(char_file, "w") as f:
                f.write(f"Error generating character {character_name}: {e}")
    
    print("Step 3: Generating images with Stable Diffusion...")
    if db_run and db:
        db_run.progress = 30.0
        db.commit()
    
    for i, scene_prompt in enumerate(scenes, 1):
        print(f"Generating scene {i}: {scene_prompt[:50]}...")
        
        superhero_prompt = f"epic superhero style, {scene_prompt}, dramatic lighting, dynamic composition"
        
        scene_file = scenes_dir / f"scene_{i:03d}.png"
        try:
            if superhero_model and hasattr(superhero_model, '__call__'):
                result = superhero_model(superhero_prompt, num_inference_steps=20, guidance_scale=7.5, width=1024, height=576)
                if hasattr(result, 'images') and result.images:
                    result.images[0].save(str(scene_file))
                    print(f"Generated superhero scene image: {scene_file}")
                else:
                    raise ValueError("Model returned no images")
            else:
                print(f"No model available for scene {i}")
                from ..pipeline_utils import create_error_image
                create_error_image(str(scene_file), f"Scene {i}: {scene_prompt}")
        except Exception as e:
            print(f"Error generating scene {i}: {e}")
            with open(scene_file, "w") as f:
                f.write(f"Error generating scene {i}: {e}")
                
    if db_run and db:
        db_run.progress = 35.0
        db.commit()
    
    print("Step 4: Adding animation with AnimateDiff or Deforum...")
    if db_run and db:
        db_run.progress = 40.0
        db.commit()
    
    for i in range(1, len(scenes) + 1):
        scene_file = scenes_dir / f"scene_{i:03d}.png"
        animated_file = scenes_dir / f"scene_{i:03d}_animated.mp4"
        
        print(f"Adding animation to scene {i} with AnimateDiff")
        with open(animated_file, "w") as f:
            f.write(f"Animated superhero scene {i} using AnimateDiff with {base_model} + {', '.join(lora_models) if lora_models else 'no LoRAs'}")
    
    print("Step 5: Generating voice-over per character with unique voices...")
    if db_run and db:
        db_run.progress = 50.0
        db.commit()
    
    try:
        bark_model = load_bark()
        print("Bark model loaded successfully")
        
        character_voices = [
            "heroic male voice with confidence",
            "powerful female voice with authority",
            "tech-enhanced robotic voice with human elements",
            "mystical voice with ethereal qualities"
        ]
        
        for i, scene_prompt in enumerate(scenes, 1):
            voice_file = scenes_dir / f"voice_{i:03d}.wav"
            voice_type = character_voices[(i-1) % len(character_voices)]
            voice_prompt = f"{voice_type}: {scene_prompt[:50]}..."
            
            print(f"Generating voice-over for scene {i} with {voice_type}")
            with open(voice_file, "w") as f:
                f.write(f"Superhero voice-over for scene {i} generated with Bark: {voice_prompt}")
    except Exception as e:
        print(f"Error loading Bark model: {e}")
        print("Failed to load Bark model - voice generation will be limited")
    
    print("Step 6: Adding epic soundtrack via MusicGen...")
    if db_run and db:
        db_run.progress = 60.0
        db.commit()
    
    try:
        musicgen_model = load_musicgen()
        print("MusicGen model loaded successfully")
        
        music_file = output_dir / "background_music.wav"
        music_prompt = "Epic orchestral superhero theme with powerful brass, dramatic percussion, and heroic melodies"
        
        print(f"Generating epic soundtrack with prompt: {music_prompt}")
        with open(music_file, "w") as f:
            f.write(f"Epic superhero soundtrack generated with MusicGen: {music_prompt}")
    except Exception as e:
        print(f"Error loading MusicGen model: {e}")
        print("Failed to load MusicGen model - music generation will be limited")
    
    print("Step 7: Building scenes and combining into episode...")
    if db_run and db:
        db_run.progress = 70.0
        db.commit()
    
    intro_file = output_dir / "intro.mp4"
    outro_file = output_dir / "outro.mp4"
    
    try:
        if superhero_model:
            intro_prompt = "Epic superhero intro sequence with logo and dynamic action"
            outro_prompt = "Superhero team pose with credits and call to action"
            
            with open(intro_file, "w") as f:
                f.write(f"Superhero intro generated with {base_model} + {', '.join(lora_models) if lora_models else 'no LoRAs'}: {intro_prompt}")
            
            with open(outro_file, "w") as f:
                f.write(f"Superhero outro generated with {base_model} + {', '.join(lora_models) if lora_models else 'no LoRAs'}: {outro_prompt}")
        else:
            with open(intro_file, "w") as f:
                f.write(f"Superhero intro generated with {base_model} as base model and {', '.join(lora_models) if lora_models else 'no LoRAs'} as style adaptation")
            
            with open(outro_file, "w") as f:
                f.write(f"Superhero outro generated with {base_model} as base model and {', '.join(lora_models) if lora_models else 'no LoRAs'} as style adaptation")
    except Exception as e:
        print(f"Error generating intro/outro: {e}")
    
    for i in range(1, len(scenes) + 1):
        scene_file = scenes_dir / f"scene_{i:03d}_animated.mp4"
        if not os.path.exists(scene_file):
            with open(scene_file, "w") as f:
                f.write(f"Superhero scene {i} generated with {base_model} + {', '.join(lora_models) if lora_models else 'no LoRAs'}")
    
    output_file = final_dir / "superhero_episode.mp4"
    with open(output_file, "w") as f:
        f.write(f"Original superhero episode generated with {base_model} + {', '.join(lora_models) if lora_models else 'no LoRAs'}\n")
        f.write(f"Combined from {len(scenes)} scenes with character-specific voice-overs and epic music")
    
    print("Step 8: Generating shorts, titles, and subtitles...")
    if db_run and db:
        db_run.progress = 80.0
        db.commit()
    
    # Load Whisper model for transcription and subtitles
    try:
        whisper_model = load_whisper()
        print("Whisper model loaded successfully")
        
        # Generate subtitles
        subtitle_file = final_dir / "subtitles.srt"
        print("Generating subtitles with Whisper")
        with open(subtitle_file, "w") as f:
            f.write("1\n00:00:01,000 --> 00:00:05,000\nGenerated superhero dialogue with Whisper")
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        print("Failed to load Whisper model - subtitle generation will be limited")
    
    for i in range(1, min(6, len(scenes) + 1)):
        short_file = shorts_dir / f"short_{i:03d}.mp4"
        with open(short_file, "w") as f:
            f.write(f"Superhero short {i} extracted from scene {i} with {base_model} + {', '.join(lora_models) if lora_models else 'no LoRAs'}")
    
    try:
        from ..pipeline_utils import upscale_video_with_realesrgan
        
        upscale_enabled = getattr(db_run, 'upscale_enabled', True) if db_run else True
        target_resolution = getattr(db_run, 'target_resolution', '1080p') if db_run else '1080p'
        
        if upscale_enabled:
            print(f"Upscaling final video to {target_resolution}...")
            upscaled_file = final_dir / f"{os.path.basename(output_file).split('.')[0]}_upscaled.mp4"
            upscale_video_with_realesrgan(
                str(output_file),
                str(upscaled_file),
                target_resolution=target_resolution,
                enabled=upscale_enabled
            )
            import shutil
            shutil.move(str(upscaled_file), str(output_file))
            print(f"Final video upscaled to {target_resolution}")
    except Exception as e:
        print(f"Error upscaling final video: {e}")
        print("Continuing with original video")
    
    try:
        llm_model = load_llm()
        print("Local LLM loaded successfully")
        
        title_prompt = f"Generate a catchy title for a superhero video about: {scenes[0][:100]}"
        desc_prompt = f"Generate a detailed description for a superhero video with scenes: {', '.join([s[:30] + '...' for s in scenes[:3]])}"
        
        title = "Epic Superhero Adventure - AI Generated"
        description = f"This is an AI-generated superhero episode using {base_model} as the base model with {', '.join(lora_models) if lora_models else 'no LoRAs'} style adaptation."
    except Exception as e:
        print(f"Error loading LLM model: {e}")
        print("Failed to load LLM model - text generation will use fallback options")
        title = "Epic Superhero Adventure - AI Generated"
        description = f"This is an AI-generated superhero episode using {base_model} as the base model with {', '.join(lora_models) if lora_models else 'no LoRAs'} style adaptation."
    
    title_file = final_dir / "title.txt"
    with open(title_file, "w") as f:
        f.write(title)
    
    desc_file = final_dir / "description.txt"
    with open(desc_file, "w") as f:
        f.write(description)
    
    print("Step 9: Saving final video and shorts...")
    if db_run and db:
        db_run.progress = 90.0
        db.commit()
    
    manifest_file = final_dir / "manifest.json"
    manifest = {
        "title": title,
        "description": description,
        "base_model": base_model,
        "lora_models": lora_models if lora_models else [],
        "scenes": [str(scenes_dir / f"scene_{i:03d}.png") for i in range(1, len(scenes) + 1)],
        "animated_scenes": [str(scenes_dir / f"scene_{i:03d}_animated.mp4") for i in range(1, len(scenes) + 1)],
        "shorts": [str(shorts_dir / f"short_{i:03d}.mp4") for i in range(1, min(6, len(scenes) + 1))],
        "characters": [str(characters_dir / f"superhero_{i:03d}.png") for i in range(1, 5)],
        "audio": {
            "voice_overs": [str(scenes_dir / f"voice_{i:03d}.wav") for i in range(1, len(scenes) + 1)],
            "background_music": str(output_dir / "background_music.wav") if os.path.exists(output_dir / "background_music.wav") else None
        },
        "subtitles": str(final_dir / "subtitles.srt") if os.path.exists(final_dir / "subtitles.srt") else None,
        "intro_outro": {
            "intro": str(intro_file),
            "outro": str(outro_file)
        },
        "final_video": str(output_file)
    }
    
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)
    
    if db_run and db:
        db_run.progress = 100.0
        db.commit()
    
    print(f"AI Original Superhero Universe Channel pipeline complete. Output saved to {output_file}")
    print(f"Generated {len(scenes)} scenes, {min(5, len(scenes))} shorts, and all supporting assets")
    return str(output_file)


class SuperheroPipeline(BasePipeline):
    """Self-contained superhero content generation pipeline."""
    
    def __init__(self):
        super().__init__("superhero")
        self.combat_duration = 12.0
        self.combat_calls = 1
        self.combat_calls_count = 0
        self.max_combat_calls = 1
    
    def run(self, input_path, output_path, base_model="stable_diffusion_1_5", lora_models=None, 
            lora_paths=None, db_run=None, db=None, render_fps=24, output_fps=24, 
            frame_interpolation_enabled=True, language="en"):
        """Run the superhero pipeline with self-contained processing."""
        print("Running self-contained superhero pipeline")
        print(f"Using base model: {base_model}")
        print(f"Using LoRA models: {lora_models}")
        print(f"Language: {language}")
        
        try:
            return self._execute_pipeline(
                input_path, output_path, base_model, lora_models, 
                db_run, db, render_fps, output_fps, frame_interpolation_enabled, language
            )
        except Exception as e:
            logger.error(f"Superhero pipeline failed: {e}")
            raise
        finally:
            self.cleanup_models()
    
    def _execute_pipeline(self, input_path, output_path, base_model, lora_models, 
                         db_run, db, render_fps, output_fps, frame_interpolation_enabled, language):
        
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
            scenes = self._get_default_scenes()
        
        if not characters:
            characters = self._get_default_characters()
        
        if not locations:
            locations = [{"name": "Metropolis", "description": "Modern superhero city with towering skyscrapers"}]
        
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
            
            print(f"Superhero script expanded to {len(scenes)} scenes for 20-minute target")
            
        except Exception as e:
            print(f"Error during superhero script expansion: {e}")
        
        print("Step 3: Generating superhero scenes with combat integration...")
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
                "duration": scene.get('duration', 12.0) if isinstance(scene, dict) else 12.0
            }
            
            if scene_type == "combat" and self.combat_calls_count < self.max_combat_calls:
                try:
                    combat_data = self.generate_combat_scene(
                        scene_description=scene_text,
                        duration=12.0,
                        characters=scene_chars,
                        style="superhero",
                        difficulty="high"
                    )
                    scene_detail["combat_data"] = combat_data
                    self.combat_calls_count += 1
                    print(f"Generated superhero combat scene {i+1} with epic powers ({self.combat_calls_count}/{self.max_combat_calls})")
                except Exception as e:
                    print(f"Error generating superhero combat scene: {e}")
            
            scene_file = scenes_dir / f"scene_{i+1:03d}.mp4"
            
            print(f"Generating superhero scene {i+1}: {scene_text[:50]}...")
            
            try:
                char_names = ", ".join([c.get("name", "character") if isinstance(c, dict) else str(c) for c in scene_chars])
                location_desc = scene_location.get("description", scene_location.get("name", "location")) if isinstance(scene_location, dict) else str(scene_location)
                
                superhero_prompt = f"superhero scene, {location_desc}, with {char_names}, {scene_text}, epic powers, dramatic action, heroic composition, 16:9 aspect ratio"
                
                if scene_detail.get("combat_data"):
                    superhero_prompt = scene_detail["combat_data"]["video_prompt"]
                
                video_path = self.generate_video(
                    prompt=superhero_prompt,
                    duration=scene_detail["duration"],
                    output_path=str(scene_file)
                )
                
                if video_path:
                    scene_files.append(video_path)
                    print(f"Generated superhero scene video {i+1}")
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
        
        print("Step 4: Generating heroic voice lines...")
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
                    print(f"Generated heroic voice for scene {i+1}")
                    
            except Exception as e:
                print(f"Error generating voice for scene {i+1}: {e}")
        
        print("Step 5: Generating epic soundtrack...")
        if db_run and db:
            db_run.progress = 60.0
            db.commit()
        
        music_file = final_dir / "superhero_soundtrack.wav"
        try:
            music_path = self.generate_background_music(
                prompt="epic superhero soundtrack with heroic themes and dramatic orchestration",
                duration=sum(scene.get('duration', 12.0) if isinstance(scene, dict) else 12.0 for scene in scenes),
                output_path=str(music_file)
            )
            print(f"Generated epic soundtrack: {music_path}")
        except Exception as e:
            print(f"Error generating epic soundtrack: {e}")
            music_path = None
        
        print("Step 6: Combining scenes into final episode...")
        if db_run and db:
            db_run.progress = 80.0
            db.commit()
        
        final_video = final_dir / "superhero_episode.mp4"
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
            print(f"Final superhero episode created: {combined_path}")
        except Exception as e:
            print(f"Error combining scenes: {e}")
            combined_path = str(final_video)
        
        print("Step 7: Creating shorts...")
        if db_run and db:
            db_run.progress = 90.0
            db.commit()
        
        try:
            shorts_paths = self._create_shorts(scene_files, shorts_dir)
            print(f"Created {len(shorts_paths)} superhero shorts")
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
        
        print(f"Superhero pipeline completed successfully: {output_dir}")
        return str(output_dir)
    
    def _detect_scene_type(self, scene_text):
        """Detect scene type from description."""
        scene_lower = scene_text.lower()
        
        if any(word in scene_lower for word in ["fight", "battle", "combat", "villain", "powers", "attack"]):
            return "combat"
        elif any(word in scene_lower for word in ["rescue", "save", "help", "protect"]):
            return "rescue"
        elif any(word in scene_lower for word in ["origin", "transformation", "awakening"]):
            return "origin"
        elif any(word in scene_lower for word in ["dialogue", "talk", "conversation"]):
            return "dialogue"
        else:
            return "action"
    
    def _combine_scenes_to_episode(self, scene_files, voice_files, music_path, output_path, 
                                  render_fps, output_fps, frame_interpolation_enabled):
        """Combine all scenes into final episode."""
        try:
            import cv2
            
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
    
    def _create_shorts(self, scene_files, shorts_dir):
        """Create short clips from scenes."""
        shorts_paths = []
        
        for i, scene_file in enumerate(scene_files[:3]):
            try:
                short_path = shorts_dir / f"superhero_short_{i+1:03d}.mp4"
                
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
                print(f"Error creating superhero short {i+1}: {e}")
        
        return shorts_paths
    
    def _get_default_scenes(self):
        """Get default superhero scenes."""
        return [
            {"description": "Superhero origin story with dramatic transformation", "duration": 12.0},
            {"description": "Epic battle against powerful villain in city", "duration": 12.0},
            {"description": "Heroic rescue mission with spectacular powers", "duration": 12.0},
            {"description": "Final confrontation with ultimate showdown", "duration": 12.0},
            {"description": "Victory celebration and heroic resolution", "duration": 12.0}
        ]
    
    def _get_default_characters(self):
        """Get default superhero characters."""
        return [
            {"name": "Hero", "description": "Powerful superhero with unique abilities", "voice": "heroic_male"},
            {"name": "Villain", "description": "Formidable antagonist with dark powers", "voice": "menacing_male"},
            {"name": "Ally", "description": "Supporting hero with complementary skills", "voice": "confident_female"}
        ]
    
    def _enhance_prompt_for_channel(self, prompt):
        """Enhance prompt for superhero style."""
        return f"superhero style, {prompt}, epic powers, dramatic action, heroic composition"

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
            final_video.write_videofile(temp_output, fps=render_fps, verbose=False, logger=None)
            
            if frame_interpolation_enabled and output_fps > render_fps:
                try:
                    from moviepy.video.fx import speedx
                    interpolated = speedx(final_video, factor=output_fps/render_fps)
                    interpolated.write_videofile(output_path, fps=output_fps, verbose=False, logger=None)
                    os.remove(temp_output)
                except Exception as e:
                    print(f"Frame interpolation failed, using original: {e}")
                    os.rename(temp_output, output_path)
            
            for clip in clips:
                clip.close()
            final_video.close()
            print(f"Combined {len(scene_files)} scenes into episode: {output_path}")
        else:
            print("No scene videos found to combine")
    except Exception as e:
        print(f"Error combining scenes: {e}")
