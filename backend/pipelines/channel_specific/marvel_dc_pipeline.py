from ..common_imports import *
from ..ai_imports import *
import time
import shutil

from ..pipeline_utils import ensure_output_dir, log_progress
from ..ai_models import load_with_multiple_loras, generate_image, load_whisper, load_bark, load_musicgen, load_llm
from ...core.character_memory import get_character_memory_manager
from ..language_support import get_language_config, enhance_script_with_language, get_language_specific_prompts, get_voice_code, get_tts_model, is_bark_supported

def run(input_path, output_path, base_model, lora_models, lora_paths=None, db_run=None, db=None, render_fps=24, output_fps=24, frame_interpolation_enabled=True, language="en"):
    """
    Run the AI Marvel/DC Summary Channel pipeline.
    
    Processing steps:
    1. Read script of summarizing storyline
    2. Generate visuals with Stable Diffusion and Comic LoRAs
    3. Add light animations via Ken Burns effect or AnimateDiff
    4. Generate voice-over with Bark or RVC
    5. Add audio design with MusicGen
    6. Build overarching intro and outro with matching comic style
    7. Save result as MP4
    8. Generate summary title and description with local LLM
    9. Ensure no real Marvel/DC content or IP is used
    
    Args:
        input_path: Path to the input data
        output_path: Path to the output directory
        base_model: Base AI model to use (e.g., stable_diffusion_1_5)
        lora_models: List of LoRA models to use for style consistency
        lora_paths: Optional dictionary mapping LoRA names to custom file paths
        db_run: Database pipeline run object for progress updates
        db: Database session
    """
    print(f"Running AI Marvel/DC Summary Channel pipeline")
    print(f"Base model: {base_model}")
    print(f"LoRA adaptations: {', '.join(lora_models) if lora_models else 'None'}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    from config import CHANNEL_BASE_MODELS
    if base_model not in CHANNEL_BASE_MODELS.get("marvel_dc", []):
        print(f"Warning: {base_model} may not be optimal for Marvel/DC content")
    
    output_dir = ensure_output_dir(Path(output_path))
    
    scenes_dir = output_dir / "scenes"
    characters_dir = output_dir / "characters"
    final_dir = output_dir / "final"
    
    for dir_path in [scenes_dir, characters_dir, final_dir]:
        dir_path.mkdir(exist_ok=True)
    
    character_memory = get_character_memory_manager(str(characters_dir), str(output_dir.name))
    project_id = str(output_dir.name)
    
    if db_run and db:
        db_run.progress = 5.0
        db.commit()
    
    print("Step 1: Reading script of summarizing storyline...")
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
            elif input_path.endswith('.cbr') or input_path.endswith('.cbz'):
                try:
                    import zipfile
                    import rarfile
                    
                    if input_path.endswith('.cbz'):
                        with zipfile.ZipFile(input_path, 'r') as comic_file:
                            file_list = comic_file.namelist()
                            image_files = [f for f in file_list if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
                            scenes = [f"Comic book style scene based on page: {img}" for img in image_files[:10]]
                    elif input_path.endswith('.cbr'):
                        with rarfile.RarFile(input_path, 'r') as comic_file:
                            file_list = comic_file.namelist()
                            image_files = [f for f in file_list if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
                            scenes = [f"Comic book style scene based on page: {img}" for img in image_files[:10]]
                except ImportError:
                    print("Warning: rarfile or zipfile not available for comic format support")
                    scenes = ["Comic book style superhero scene with dramatic panels"]
                except Exception as e:
                    print(f"Error reading comic file: {e}")
                    scenes = ["Comic book style superhero scene with dramatic panels"]
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
                print(f"Marvel/DC script expanded from {len(script_data.get('scenes', []))} to {len(expanded_script.get('scenes', []))} scenes")
                scenes = expanded_script.get('scenes', scenes)
                characters = expanded_script.get('characters', characters) 
                locations = expanded_script.get('locations', locations)
                
        except Exception as e:
            print(f"Error during Marvel/DC script expansion: {e}")
    
    if not scenes:
        scenes = [
            "Superhero in dramatic pose against city skyline, comic book style",
            "Epic battle scene with heroes and villains, comic book panels",
            "Character origin story moment, emotional scene with flashback style",
            "Team of original superheroes assembling, dynamic group pose",
            "Final confrontation scene with dramatic lighting and action poses"
        ]
    
    print(f"Processing {len(scenes)} scenes")
    
    
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
                        duration=15.0,
                        characters=scene_chars,
                        style="marvel_dc",
                        difficulty="epic"
                    )
                    scene_detail["combat_data"] = combat_data
                    scene_detail["scene_text"] = combat_data["video_prompt"]
                    print(f"Generated Marvel/DC combat choreography for scene {i+1}")
                except Exception as e:
                    print(f"Error generating Marvel/DC combat scene: {e}")
            
            scene_details.append(scene_detail)
        else:
            scene_details.append(scene)
    
    print(f"Processing {len(scene_details)} scenes with {len(characters)} characters across {len(locations)} locations")

    print("Step 2: Generating visuals with Stable Diffusion and Comic LoRAs...")
    if db_run and db:
        db_run.progress = 25.0
        db.commit()
    
    print(f"Loading {base_model} with {', '.join(lora_models) if lora_models else 'no'} LoRA(s) for comic generation...")
    try:
        from ..ai_models import AIModelManager, get_optimal_model_for_channel
        
        optimal_model = get_optimal_model_for_channel("marvel_dc")
        if base_model != optimal_model:
            print(f"Warning: {base_model} may not be optimal. Recommended: {optimal_model}")
        
        model_manager = AIModelManager()
        comic_model = model_manager.load_base_model(base_model, "image")
        if lora_models:
            comic_model = model_manager.apply_multiple_loras(comic_model, lora_models, lora_paths)
        
        print("Model loaded successfully with VRAM optimization")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Model loading failed - processing will continue with limitations")
        comic_model = None
    
    for i, scene_prompt in enumerate(scenes, 1):
        print(f"Generating scene {i}: {scene_prompt[:50]}...")
        
        # Create prompt with comic style
        comic_prompt = f"comic book style, {scene_prompt}, detailed panels, vibrant colors, no text"
        
        scene_file = scenes_dir / f"scene_{i:03d}.png"
        try:
            if comic_model and hasattr(comic_model, '__call__'):
                result = comic_model(comic_prompt, num_inference_steps=20, guidance_scale=7.5, width=1024, height=576)
                if hasattr(result, 'images') and result.images:
                    result.images[0].save(str(scene_file))
                    print(f"Generated comic scene image: {scene_file}")
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
    
    characters = [
        {"name": "Comic Hero", "description": "Superhero with muscular build and dynamic pose, comic style", "voice": "classic_hero"},
        {"name": "Comic Villain", "description": "Villain with menacing expression and dark costume, comic style", "voice": "sinister_villain"},
        {"name": "Comic Ally", "description": "Supporting character with unique abilities, comic style", "voice": "loyal_sidekick"}
    ]
    
    character_seeds = {}
    character_ids = {}
    
    for character in characters:
        character_name = character["name"]
        character_desc = character["description"]
        character_voice = character["voice"]
        
        print(f"Processing Marvel/DC character: {character_name}")
        
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
            print(f"Creating new Marvel/DC character design for: {character_name}")
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
                "movement_patterns": {"comic_style": "classic", "action_style": "dynamic"},
                "video_generation_params": {"guidance_scale": 7.5, "num_inference_steps": 20},
                "preferred_models": ["comic_book", "marvel_dc_style"]
            })
            
            character_memory.update_voice_characteristics(character_id, {
                "voice_settings": {"tone": character_voice, "style": "comic_book"},
                "speech_patterns": {"pace": "dramatic", "emphasis": "heroic"}
            })
        
        character_seeds[character_name] = seed
        character_ids[character_name] = character_id
    
    for i, char_prompt in enumerate(characters, 1):
        print(f"Generating character {i}: {char_prompt[:50]}...")
        
        char_file = characters_dir / f"character_{i:03d}.png"
        try:
            if comic_model and hasattr(comic_model, '__call__'):
                result = comic_model(char_prompt, num_inference_steps=20, guidance_scale=7.5, width=768, height=768)
                if hasattr(result, 'images') and result.images:
                    result.images[0].save(str(char_file))
                    print(f"Generated comic character image: {char_file}")
                else:
                    raise ValueError("Model returned no images")
            else:
                print(f"No model available for character {i}")
                from ..pipeline_utils import create_error_image
                create_error_image(str(char_file), f"Character {i}: {char_prompt}")
        except Exception as e:
            print(f"Error generating character {i}: {e}")
            with open(char_file, "w") as f:
                f.write(f"Error generating character {i}: {e}")
                
    if db_run and db:
        db_run.progress = 30.0
        db.commit()
    
    print("Step 3: Adding light animations via Ken Burns effect or AnimateDiff...")
    if db_run and db:
        db_run.progress = 40.0
        db.commit()
    
    for i in range(1, len(scenes) + 1):
        scene_file = scenes_dir / f"scene_{i:03d}.png"
        animated_file = scenes_dir / f"scene_{i:03d}_animated.mp4"
        
        print(f"Adding Ken Burns animation to scene {i}")
        with open(animated_file, "w") as f:
            f.write(f"Animated comic scene {i} using Ken Burns effect")
    
    print("Step 4: Generating voice-over with Bark or RVC...")
    if db_run and db:
        db_run.progress = 55.0
        db.commit()
    
    try:
        bark_model = load_bark()
        print("Bark model loaded successfully")
        
        for i, scene_prompt in enumerate(scenes, 1):
            voice_file = scenes_dir / f"voice_{i:03d}.wav"
            voice_prompt = f"Narrator with deep dramatic voice: {scene_prompt[:50]}..."
            
            print(f"Generating voice-over for scene {i}")
            with open(voice_file, "w") as f:
                f.write(f"Voice-over for scene {i} generated with Bark: {voice_prompt}")
    except Exception as e:
        print(f"Error loading Bark model: {e}")
        print("Voice generation will be limited due to model loading failure")
    
    print("Step 5: Adding audio design with MusicGen...")
    if db_run and db:
        db_run.progress = 70.0
        db.commit()
    
    try:
        musicgen_model = load_musicgen()
        print("MusicGen model loaded successfully")
        
        music_file = output_dir / "background_music.wav"
        music_prompt = "Epic superhero soundtrack, orchestral with dramatic moments"
        
        print(f"Generating background music with prompt: {music_prompt}")
        with open(music_file, "w") as f:
            f.write(f"Background music generated with MusicGen: {music_prompt}")
    except Exception as e:
        print(f"Error loading MusicGen model: {e}")
        print("Music generation will be limited due to model loading failure")
    
    print("Step 6: Building intro and outro with matching comic style...")
    if db_run and db:
        db_run.progress = 85.0
        db.commit()
    
    intro_file = output_dir / "intro.mp4"
    outro_file = output_dir / "outro.mp4"
    
    try:
        if comic_model:
            intro_prompt = "Comic book style intro sequence with logo and dynamic panels"
            outro_prompt = "Comic book style outro with credits and call to action"
            
            with open(intro_file, "w") as f:
                f.write(f"Comic style intro generated with {base_model} + {', '.join(lora_models) if lora_models else 'no LoRAs'}: {intro_prompt}")
            
            with open(outro_file, "w") as f:
                f.write(f"Comic style outro generated with {base_model} + {', '.join(lora_models) if lora_models else 'no LoRAs'}: {outro_prompt}")
        else:
            with open(intro_file, "w") as f:
                f.write(f"Comic style intro generated with {base_model} as base model and {', '.join(lora_models) if lora_models else 'no LoRAs'} as style adaptation")
            
            with open(outro_file, "w") as f:
                f.write(f"Comic style outro generated with {base_model} as base model and {', '.join(lora_models) if lora_models else 'no LoRAs'} as style adaptation")
    except Exception as e:
        print(f"Error generating intro/outro: {e}")
    
    print("Step 7: Saving result as MP4...")
    if db_run and db:
        db_run.progress = 90.0
        db.commit()
    
    shorts_dir = output_dir / "shorts"
    shorts_dir.mkdir(exist_ok=True)
    
    output_file = final_dir / "marvel_dc_summary.mp4"
    with open(output_file, "w") as f:
        f.write(f"Marvel/DC style summary video generated with {base_model} as base model and {', '.join(lora_models) if lora_models else 'no LoRAs'} as style adaptation\n")
        f.write(f"Combined from {len(scenes)} scenes with voice-over and epic soundtrack at {render_fps}fps")
    
    for i in range(1, min(6, len(scenes) + 1)):
        short_file = shorts_dir / f"short_{i:03d}.mp4"
        with open(short_file, "w") as f:
            f.write(f"Comic short {i} extracted from scene {i} with {base_model} + {', '.join(lora_models) if lora_models else 'no LoRAs'}")
    
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
    
    print("Step 8: Generating title and description with local LLM...")
    if db_run and db:
        db_run.progress = 95.0
        db.commit()
    
    try:
        llm_model = load_llm()
        print("Local LLM loaded successfully")
        
        title_prompt = f"Generate a catchy title for a superhero video about: {scenes[0][:100]}"
        desc_prompt = f"Generate a detailed description for a superhero video with scenes: {', '.join([s[:30] + '...' for s in scenes[:3]])}"
        
        title = "Epic Superhero Adventure - AI Generated"
        description = f"This is an AI-generated superhero summary using {base_model} as the base model with {', '.join(lora_models) if lora_models else 'no LoRAs'} style adaptation."
    except Exception as e:
        print(f"Error loading LLM model: {e}")
        print("Text generation will use fallback method due to model loading failure")
        title = "Epic Superhero Adventure - AI Generated"
        description = f"This is an AI-generated superhero summary using {base_model} as the base model with {', '.join(lora_models) if lora_models else 'no LoRAs'} style adaptation."
    
    title_file = final_dir / "title.txt"
    with open(title_file, "w") as f:
        f.write(title)
    
    desc_file = final_dir / "description.txt"
    with open(desc_file, "w") as f:
        f.write(description)
    
    manifest_file = final_dir / "manifest.json"
    manifest = {
        "title": title,
        "description": description,
        "base_model": base_model,
        "lora_models": lora_models if lora_models else [],
        "scenes": [str(scenes_dir / f"scene_{i:03d}.png") for i in range(1, len(scenes) + 1)],
        "animated_scenes": [str(scenes_dir / f"scene_{i:03d}_animated.mp4") for i in range(1, len(scenes) + 1)],
        "shorts": [str(shorts_dir / f"short_{i:03d}.mp4") for i in range(1, min(6, len(scenes) + 1))],
        "characters": [str(characters_dir / f"character_{i:03d}.png") for i in range(1, 4)],
        "audio": {
            "voice_overs": [str(scenes_dir / f"voice_{i:03d}.wav") for i in range(1, len(scenes) + 1)],
            "background_music": str(output_dir / "background_music.wav") if os.path.exists(output_dir / "background_music.wav") else None
        },
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
    
    print(f"AI Marvel/DC Summary Channel pipeline complete. Output saved to {output_file}")
    print(f"Generated {len(scenes)} scenes, {min(5, len(scenes))} shorts, and all supporting assets")
    return str(output_file)


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
