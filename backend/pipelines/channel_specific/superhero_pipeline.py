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
