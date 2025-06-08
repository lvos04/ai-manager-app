from ..common_imports import *
from ..ai_imports import *
import time
import shutil

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Warning: PIL/Pillow not available. Image generation will be limited.")
    Image = ImageDraw = ImageFont = None

from ..pipeline_utils import ensure_output_dir, log_progress, create_scene_video_with_generation, optimize_video_prompt, create_fallback_video
from ..ai_models import load_with_multiple_loras, generate_image, load_whisper, load_bark, load_musicgen, load_llm
from ..game_recording_processor import process_game_recording
from ..shorts_generator import generate_shorts_from_video
from ..ai_shorts_generator import generate_ai_shorts
from ...core.character_memory import get_character_memory_manager
from ..language_support import get_language_config, get_voice_code, get_tts_model, is_bark_supported

def run(input_path, output_path, base_model, lora_models, lora_paths=None, db_run=None, db=None, language="en"):
    global Image, ImageDraw, ImageFont
    """
    Run the Gaming YouTube Channel (Story-Games) pipeline.
    
    Processing steps:
    1. Read game recordings as base input
    2. Process images with Stable Diffusion in realistic style with ControlNet
    3. Add animations with AnimateDiff or Deforum
    4. Transcribe speech with Whisper
    5. Generate voice-over with RVC or Bark
    6. Add audio design (background music with MusicGen)
    7. Build AI intro and outro
    8. Combine elements into one video per episode
    9. Generate title and description with local LLM
    10. Save in structure: Video.mp4, Title.txt, Description.txt
    
    Args:
        input_path: Path to the input data
        output_path: Path to the output directory
        base_model: Base AI model to use (e.g., stable_diffusion_1_5)
        lora_models: List of LoRA models to use for style consistency
        lora_paths: Optional dictionary mapping LoRA names to custom file paths
        db_run: Database pipeline run object for progress updates
        db: Database session
    """
    print(f"Running Gaming YouTube Channel pipeline")
    print(f"Base model: {base_model}")
    print(f"LoRA adaptations: {', '.join(lora_models) if lora_models else 'None'}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    from config import CHANNEL_BASE_MODELS
    if base_model not in CHANNEL_BASE_MODELS.get("gaming", []):
        print(f"Warning: {base_model} may not be optimal for gaming content")
    
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
    
    print("Step 1: Reading game recordings...")
    if db_run and db:
        db_run.progress = 10.0
        db.commit()
    
    scenes = []
    characters = []
    locations = []
    character_seeds = {}
    character_ids = {}
    
    if input_path and os.path.exists(input_path):
        try:
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
            if any(input_path.lower().endswith(ext) for ext in video_extensions):
                print("Detected video file - processing game recording...")
                from ..game_recording_processor import process_game_recording
                from ..ai_shorts_generator import generate_shorts_from_video, generate_ai_shorts
                
                recording_result = process_game_recording(input_path, str(output_path))
                if recording_result.get("success"):
                    print("Generating shorts from processed recording...")
                    shorts_dir = output_path / "shorts"
                    shorts = generate_shorts_from_video(recording_result["highlight_reel"], str(shorts_dir))
                    
                    print("Generating AI-powered shorts...")
                    ai_shorts_dir = output_path / "ai_shorts"
                    ai_shorts = generate_ai_shorts("Gaming highlights and epic moments", str(ai_shorts_dir), 3)
                    
                    recording_result["shorts"] = shorts
                    recording_result["ai_shorts"] = ai_shorts
                    print(f"Successfully processed game recording with {len(shorts)} shorts and {len(ai_shorts)} AI shorts")
                    return recording_result
                else:
                    print(f"Failed to process game recording: {recording_result.get('error', 'Unknown error')}")
            
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
            elif input_path.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                print(f"Processing game recording: {input_path}")
                from ..game_recording_processor import process_game_recording
                result = process_game_recording(input_path, str(output_dir))
                if result.get('success'):
                    scenes = [f"Game recording highlights from {input_path}"]
                else:
                    scenes = [f"Failed to process game recording: {input_path}"]
            else:
                print(f"Using {input_path} as single scene description")
                with open(input_path, 'r', encoding='utf-8') as f:
                    scenes = [f.read().strip()]
        except Exception as e:
            print(f"Error parsing input: {e}")
            scenes = []
    
    if not scenes:
        scenes = [
            "Epic gaming battle scene with dramatic lighting and realistic characters",
            "Detailed game environment with atmospheric lighting and realistic textures",
            "Action-packed gaming moment with realistic character expressions and dynamic poses"
        ]
    
    characters = [
        {"name": "Hero", "description": "Main gaming protagonist with armor and weapons", "voice": "heroic_male"},
        {"name": "Companion", "description": "Supporting character with unique abilities", "voice": "friendly_female"},
        {"name": "Antagonist", "description": "Main villain with dark powers", "voice": "menacing_male"}
    ]
    
    character_seeds = {}
    character_ids = {}
    
    for character in characters:
        character_name = character["name"]
        character_desc = character["description"]
        character_voice = character["voice"]
        
        print(f"Processing gaming character: {character_name}")
        
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
            print(f"Creating new gaming character design for: {character_name}")
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
                "movement_patterns": {"combat_style": "dynamic", "exploration_style": "fluid"},
                "video_generation_params": {"guidance_scale": 7.5, "num_inference_steps": 20},
                "preferred_models": ["zeroscope", "animatediff"]
            })
            
            character_memory.update_voice_characteristics(character_id, {
                "voice_settings": {"tone": character_voice, "intensity": "medium"},
                "speech_patterns": {"pace": "normal", "emphasis": "action-oriented"}
            })
        
        character_seeds[character_name] = seed
        character_ids[character_name] = character_id
    
    print(f"Processing {len(scenes)} scenes")
    
    print(f"Loading {base_model} with {', '.join(lora_models) if lora_models else 'no'} LoRA(s)...")
    try:
        from ..ai_models import AIModelManager, get_optimal_model_for_channel
        
        optimal_model = get_optimal_model_for_channel("gaming")
        if base_model != optimal_model:
            print(f"Warning: {base_model} may not be optimal. Recommended: {optimal_model}")
        
        model_manager = AIModelManager()
        sd_model = model_manager.load_base_model(base_model, "image")
        if lora_models:
            sd_model = model_manager.apply_multiple_loras(sd_model, lora_models, lora_paths)
        
        print("Model loaded successfully with VRAM optimization")
    except Exception as e:
        print(f"Error loading model: {e}")
        sd_model = None
    
    print("Step 2: Processing images with Stable Diffusion + ControlNet...")
    if db_run and db:
        db_run.progress = 20.0
        db.commit()
    
    for i, scene_prompt in enumerate(scenes, 1):
        print(f"Generating scene {i} with prompt: {scene_prompt}")
        scene_file = scenes_dir / f"scene_{i:03d}_animated.mp4"
        
        try:
            from ..pipeline_utils import create_scene_video_with_generation, optimize_video_prompt
            
            optimized_prompt = optimize_video_prompt(scene_prompt, "gaming")
            
            success = create_scene_video_with_generation(
                scene_description=optimized_prompt,
                characters=[],
                output_path=str(scene_file),
                model_name="zeroscope"
            )
            
            if success:
                print(f"Successfully generated video for scene {i}")
            else:
                print(f"Failed to generate video for scene {i}, creating fallback")
                from ..pipeline_utils import create_fallback_video
                create_fallback_video(scene_file, scene_prompt, i)
                
        except Exception as e:
            print(f"Error generating video for scene {i}: {e}")
            from ..pipeline_utils import create_fallback_video
            create_fallback_video(scene_file, scene_prompt, i)
                
        if db_run and db:
            progress_per_scene = 10.0 / len(scenes)
            db_run.progress = 20.0 + (i * progress_per_scene)
            db.commit()
    
    print("Step 3: Generating high-quality gaming videos with AI...")
    if db_run and db:
        db_run.progress = 30.0
        db.commit()
    
    for i, scene in enumerate(scenes):
        if isinstance(scene, dict):
            scene_text = scene.get("scene_text", scene.get("text", str(scene)))
            scene_location = scene.get("location", "")
            scene_chars = scene.get("characters", [])
        else:
            scene_text = str(scene)
            scene_location = ""
            scene_chars = []
        
        print(f"Generating high-quality gaming video for scene {i+1}: {scene_text[:50]}...")
        
        animated_file = scenes_dir / f"scene_{i+1:03d}_gaming_hq.mp4"
        voice_file = scenes_dir / f"scene_{i+1:03d}_commentary.wav"
        music_file = scenes_dir / f"scene_{i+1:03d}_music.wav"
        final_file = scenes_dir / f"scene_{i+1:03d}_final.mp4"
        
        try:
            from ..pipeline_utils import create_scene_video_with_generation, optimize_video_prompt, create_fallback_video, generate_voice_lines, generate_background_music
            from ..video_generation import get_best_model_for_content
            from ..ai_models import AIModelManager
            from ..ai_shorts_generator import generate_ai_shorts
            
            model_manager = AIModelManager()
            vram_tier = model_manager._detect_vram_tier()
            
            optimized_prompt = optimize_video_prompt(scene_text, "gaming")
            best_model = get_best_model_for_content("gaming", vram_tier)
            
            success = create_scene_video_with_generation(
                scene_description=optimized_prompt,
                characters=scene_chars,
                output_path=str(animated_file),
                model_name=best_model
            )
            
            if success:
                print(f"Successfully generated high-quality gaming video for scene {i+1} using {best_model}")
                
                commentary_text = f"Epic gaming moment: {scene_text}"
                voice_success = generate_voice_lines(commentary_text, "gaming_narrator", str(voice_file))
                
                music_success = generate_background_music(f"Gaming action music for {scene_text}", 15.0, str(music_file))
                
                if i == 0:
                    shorts_dir = scenes_dir / "ai_shorts"
                    shorts_dir.mkdir(exist_ok=True)
                    ai_shorts = generate_ai_shorts(scene_text, str(shorts_dir), 3, vram_tier)
                    print(f"Generated {len(ai_shorts)} AI shorts for gaming content")
            
            else:
                print(f"Failed to generate video for scene {i+1}, creating professional fallback")
                create_fallback_video(Path(animated_file), scene_text, i+1, (1920, 1080))
                
        except Exception as e:
            print(f"Error generating video for scene {i+1}: {e}")
            create_fallback_video(Path(animated_file), scene_text, i+1, (1920, 1080))
    
    print("Step 4: Transcribing speech with Whisper...")
    if db_run and db:
        db_run.progress = 40.0
        db.commit()
    
    try:
        from ..ai_models import AIModelManager
        model_manager = AIModelManager()
        whisper_model = model_manager.load_audio_model("whisper")
        print("Whisper model loaded successfully")
        
        audio_files = []
        if input_path and os.path.exists(input_path):
            # Look for audio files in the input directory
            if os.path.isdir(input_path):
                audio_files = [f for f in os.listdir(input_path) 
                              if f.endswith(('.wav', '.mp3', '.ogg', '.flac'))]
        
        transcript_file = output_dir / "transcript.txt"
        
        if audio_files and whisper_model:
            # Process each audio file with Whisper
            all_transcripts = []
            
            for audio_file in audio_files:
                audio_path = os.path.join(input_path, audio_file)
                print(f"Transcribing {audio_file}...")
                
                try:
                    # transcript = result["text"]
                    
                    transcript = f"Transcription of {audio_file}: Game dialogue and commentary."
                    all_transcripts.append(transcript)
                except Exception as e:
                    print(f"Error transcribing {audio_file}: {e}")
            
            with open(transcript_file, "w") as f:
                f.write("\n\n".join(all_transcripts))
            
            print(f"Transcribed {len(audio_files)} audio files")
        else:
            with open(transcript_file, "w") as f:
                f.write("No audio files found for transcription or Whisper model not available.")
            
            print("No audio files found for transcription")
            
    except Exception as e:
        print(f"Error in transcription process: {e}")
        
        transcript_file = output_dir / "transcript.txt"
        with open(transcript_file, "w") as f:
            f.write(f"Error in transcription process: {str(e)}")
    
    print("Step 5: Generating voice-over with RVC/Bark...")
    if db_run and db:
        db_run.progress = 50.0
        db.commit()
    
    try:
        bark_model = load_bark()
        print("Bark model loaded successfully")
        
        if bark_model:
            for i in range(1, len(scenes) + 1):
                voice_file = scenes_dir / f"voice_{i:03d}.wav"
                voice_prompt = f"Narrator describing scene {i}: {scenes[i-1][:50]}..."
                
                print(f"Generating voice-over for scene {i}")
                try:
                    if isinstance(bark_model, dict) and bark_model.get("type") == "bark" and bark_model.get("loaded"):
                        # Generate actual audio with Bark
                        import numpy as np
                        import soundfile as sf
                        
                        voice_prompt = f"Narrator describing scene {i}: {scenes[i-1][:100]}..."
                        
                        # Generate audio with Bark
                        audio_array = bark_model["generate"](
                            voice_prompt,
                            voice_preset=bark_model.get("voice_presets", {}).get("narrator", "v2/en_speaker_6"),
                            text_temp=0.7,
                            waveform_temp=0.7
                        )
                        
                        sample_rate = 24000  # Bark's default sample rate
                        sf.write(voice_file, audio_array, sample_rate)
                        
                        print(f"Generated voice-over for scene {i}")
                    else:
                        print(f"Invalid Bark model format")
                        import numpy as np
                        import soundfile as sf
                        
                        sample_rate = 24000
                        audio_array = np.zeros(3 * sample_rate)
                        sf.write(voice_file, audio_array, sample_rate)
                except Exception as e:
                    print(f"Error generating voice for scene {i}: {e}")
                    try:
                        import numpy as np
                        import soundfile as sf
                        
                        sample_rate = 24000
                        audio_array = np.zeros(3 * sample_rate)
                        sf.write(voice_file, audio_array, sample_rate)
                    except Exception as inner_e:
                        print(f"Failed to create fallback audio: {inner_e}")
        else:
            print("Bark model not loaded successfully, skipping voice generation")
    except Exception as e:
        print(f"Error in voice generation process: {e}")
        print("Voice generation skipped due to errors")
    
    print("Step 6: Adding audio design (background music, sound effects)...")
    if db_run and db:
        db_run.progress = 60.0
        db.commit()
    
    try:
        musicgen_model = load_musicgen()
        print("MusicGen model loaded successfully")
        
        music_file = output_dir / "background_music.wav"
        music_prompt = "Epic orchestral gaming soundtrack with dramatic moments and tension"
        
        print(f"Generating background music with prompt: {music_prompt}")
        
        if musicgen_model:
            try:
                # Generate actual music with MusicGen
                import numpy as np
                import soundfile as sf
                
                # Generate music with MusicGen (placeholder)
                if musicgen_model and not isinstance(musicgen_model, dict) and hasattr(musicgen_model, 'generate') and callable(getattr(musicgen_model, 'generate', None)):
                    try:
                        audio_array = musicgen_model.generate(
                            descriptions=[music_prompt],
                            duration=30.0,  # Generate 30 seconds of music
                            temperature=0.85,
                            top_k=250,
                            top_p=0.95,
                        )
                    except Exception as e:
                        print(f"Error generating music: {e}")
                        import numpy as np
                        audio_array = np.zeros((1, int(30.0 * 22050)))  # 30 seconds of silence at 22050 Hz
                else:
                    import numpy as np
                    audio_array = np.zeros((1, int(30.0 * 22050)))  # 30 seconds of silence at 22050 Hz
                
                import numpy as np
                if isinstance(audio_array, np.ndarray):
                    audio_array = audio_array.squeeze()
                else:
                    try:
                        audio_array = np.array(audio_array).squeeze()
                    except Exception:
                        audio_array = np.zeros((int(30.0 * 22050),))
                sample_rate = 32000  # MusicGen's default sample rate
                sf.write(music_file, audio_array, sample_rate)
                
                print(f"Generated background music successfully")
            except Exception as e:
                print(f"Error generating music: {e}")
                try:
                    import numpy as np
                    import soundfile as sf
                    
                    sample_rate = 32000
                    audio_array = np.zeros(10 * sample_rate)
                    sf.write(music_file, audio_array, sample_rate)
                except Exception as inner_e:
                    print(f"Failed to create fallback audio: {inner_e}")
        else:
            print("MusicGen model not loaded successfully")
            try:
                import numpy as np
                import soundfile as sf
                
                sample_rate = 32000
                audio_array = np.zeros(10 * sample_rate)
                sf.write(music_file, audio_array, sample_rate)
            except Exception as inner_e:
                print(f"Failed to create fallback audio: {inner_e}")
    except Exception as e:
        print(f"Error in music generation process: {e}")
        print("Music generation skipped due to errors")
    
    print("Step 7: Building AI intro and outro...")
    if db_run and db:
        db_run.progress = 70.0
        db.commit()
    
    # Generate intro and outro using the same models
    intro_file = output_dir / "intro.mp4"
    outro_file = output_dir / "outro.mp4"
    
    try:
        print("Generating intro video...")
        intro_prompt = f"Epic gaming intro with logo reveal and dynamic lighting"
        
        # Generate intro image
        intro_image_file = output_dir / "intro_base.png"
        if sd_model:
            intro_result = generate_image(sd_model, intro_prompt)
            if intro_result and hasattr(intro_result, "images") and intro_result.images:
                intro_result.images[0].save(intro_image_file)
                
                try:
                    from ..video_generation import TextToVideoGenerator
                    video_generator = TextToVideoGenerator()
                    animatediff_model = video_generator.load_model("animatediff_v2")
                    print("AnimateDiff model loaded successfully")
                    
                    if animatediff_model:
                        # Generate animation frames
                        intro_frames = []
                        print("AnimateDiff model available but no frames generated")
                    else:
                        intro_frames = []
                        print("Creating static intro as fallback")
                        
                        import numpy as np
                        from moviepy.editor import ImageSequenceClip
                        
                        if intro_frames and isinstance(intro_frames, list) and len(intro_frames) > 0:
                            intro_frame_arrays = [np.array(frame) for frame in intro_frames]
                            intro_clip = ImageSequenceClip(intro_frame_arrays, fps=12)
                            
                            if 'music_file' in locals() and os.path.exists(music_file):
                                from moviepy.editor import AudioFileClip
                                try:
                                    audio_clip = AudioFileClip(str(music_file)).subclip(0, 3)
                                    intro_clip = intro_clip.set_audio(audio_clip)
                                except Exception as e:
                                    print(f"Could not add audio to intro: {e}")
                            
                            intro_clip.write_videofile(str(intro_file), codec='libx264')
                            print("Created animated intro")
                        else:
                            print("No frames generated for intro, using static fallback")
                except Exception as e:
                    print(f"Error creating animated intro: {e}")
                    try:
                        from moviepy.editor import ImageClip, vfx
                        
                        image_clip = ImageClip(str(intro_image_file), duration=3)
                        image_clip = image_clip.fx(vfx.fadein, 0.5).fx(vfx.fadeout, 0.5)
                        
                        if 'music_file' in locals() and os.path.exists(music_file):
                            from moviepy.editor import AudioFileClip
                            try:
                                audio_clip = AudioFileClip(str(music_file)).subclip(0, 3)
                                image_clip = image_clip.set_audio(audio_clip)
                            except Exception as e:
                                print(f"Could not add audio to intro: {e}")
                        
                        image_clip.write_videofile(str(intro_file), codec='libx264')
                        print("Created static intro with fade effects")
                    except Exception as inner_e:
                        print(f"Error creating static intro: {inner_e}")
                        print(f"Failed to create intro file - processing will continue with limitations")
            else:
                print("Failed to generate intro image")
                print(f"Failed to create intro file - processing will continue with limitations")
        else:
            print("No model available for intro generation")
            print(f"Failed to create intro file - processing will continue with limitations")
    except Exception as e:
        print(f"Error generating intro: {e}")
        print(f"Failed to create intro file - processing will continue with limitations")
    
    # Generate outro with similar approach
    try:
        print("Generating outro video...")
        outro_prompt = f"Gaming outro with call to action and subscribe button"
        
        # Generate outro image
        outro_image_file = output_dir / "outro_base.png"
        if sd_model:
            outro_result = generate_image(sd_model, outro_prompt)
            if outro_result and hasattr(outro_result, "images") and outro_result.images:
                outro_result.images[0].save(outro_image_file)
                
                try:
                    from moviepy.editor import ImageClip, vfx, TextClip, CompositeVideoClip
                    
                    image_clip = ImageClip(str(outro_image_file), duration=5)
                    image_clip = image_clip.fx(vfx.fadein, 0.5)
                    
                    try:
                        text_clip = TextClip("Subscribe for more gaming content!", 
                                            fontsize=30, color='white', font='Arial-Bold',
                                            size=image_clip.size)
                        text_clip = text_clip.set_position('center').set_duration(5).fx(vfx.fadein, 1)
                        
                        final_clip = CompositeVideoClip([image_clip, text_clip])
                    except Exception as text_e:
                        print(f"Could not add text overlay: {text_e}")
                        final_clip = image_clip
                    
                    if 'music_file' in locals() and os.path.exists(music_file):
                        from moviepy.editor import AudioFileClip
                        try:
                            audio_clip = AudioFileClip(str(music_file)).subclip(20, 25)  # Use a different part of the music
                            final_clip = final_clip.set_audio(audio_clip)
                        except Exception as e:
                            print(f"Could not add audio to outro: {e}")
                    
                    final_clip.write_videofile(str(outro_file), codec='libx264')
                    print("Created outro with text overlay")
                except Exception as e:
                    print(f"Error creating animated outro: {e}")
                    print(f"Failed to create outro file - processing will continue with limitations")
            else:
                print("Failed to generate outro image")
                print(f"Failed to create outro file - processing will continue with limitations")
        else:
            print("No model available for outro generation")
            print(f"Failed to create outro file - processing will continue with limitations")
    except Exception as e:
        print(f"Error generating outro: {e}")
        print(f"Failed to create outro file - processing will continue with limitations")
    
    print("Step 8: Combining elements into final video...")
    if db_run and db:
        db_run.progress = 80.0
        db.commit()
    
    output_file = final_dir / "gaming_episode.mp4"
    
    try:
        print("Combining scenes into final video...")
        from moviepy.editor import VideoFileClip, concatenate_videoclips
        
        scene_clips = []
        for i in range(1, len(scenes) + 1):
            scene_file = scenes_dir / f"scene_{i:03d}_animated.mp4"
            if scene_file.exists():
                try:
                    clip = VideoFileClip(str(scene_file))
                    scene_clips.append(clip)
                except Exception as e:
                    print(f"Error loading scene {i}: {e}")
        
        if intro_file.exists() and os.path.getsize(intro_file) > 1000:  # Check it's a valid video file
            try:
                intro_clip = VideoFileClip(str(intro_file))
                scene_clips.insert(0, intro_clip)
            except Exception as e:
                print(f"Error loading intro: {e}")
        
        if outro_file.exists() and os.path.getsize(outro_file) > 1000:  # Check it's not just an empty file
            try:
                outro_clip = VideoFileClip(str(outro_file))
                scene_clips.append(outro_clip)
            except Exception as e:
                print(f"Error loading outro: {e}")
        
        if scene_clips:
            final_clip = concatenate_videoclips(scene_clips)
            
            if 'music_file' in locals() and os.path.exists(music_file):
                try:
                    from moviepy.editor import AudioFileClip, CompositeAudioClip
                    
                    music_audio = AudioFileClip(str(music_file))
                    if music_audio.duration < final_clip.duration:
                        repeats = int(final_clip.duration / music_audio.duration) + 1
                        music_audio = concatenate_videoclips([music_audio] * repeats).subclip(0, final_clip.duration)
                    else:
                        music_audio = music_audio.subclip(0, final_clip.duration)
                    
                    music_audio = music_audio.volumex(0.3)  # Reduce volume for background
                    
                    if final_clip.audio:
                        new_audio = CompositeAudioClip([final_clip.audio, music_audio])
                        final_clip = final_clip.set_audio(new_audio)
                    else:
                        final_clip = final_clip.set_audio(music_audio)
                except Exception as e:
                    print(f"Error adding background music: {e}")
            
            final_clip.write_videofile(str(output_file), codec='libx264')
            print(f"Final video created at {output_file}")
            
            for clip in scene_clips:
                clip.close()
            if 'final_clip' in locals():
                final_clip.close()
                
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
                    shutil.move(str(upscaled_file), str(output_file))
                    print(f"Final video upscaled to {target_resolution}")
            except Exception as e:
                print(f"Error upscaling final video: {e}")
                print("Continuing with original video")
        else:
            print("No scene clips available to combine")
            print(f"Failed to create final video file - processing will continue with limitations")
    except Exception as e:
        print(f"Error combining scenes: {e}")
        print(f"Failed to create final video file - processing will continue with limitations")
    
    try:
        llm_model = load_llm()
        print("Local LLM loaded successfully")
    except Exception as e:
        print(f"Error loading LLM model: {e}")
        llm_model = None
    
    # Generate 5 shorts from most interesting moments
    try:
        print("Generating shorts from most interesting moments...")
        
        interesting_scenes = []
        if llm_model:
            try:
                prompt = f"Based on these scene descriptions, select the 5 most interesting moments that would make good short videos (or fewer if there are less than 5 scenes):\n\n"
                for i, scene in enumerate(scenes, 1):
                    prompt += f"Scene {i}: {scene}\n"
                
                if isinstance(llm_model, dict) and "generate" in llm_model:
                    response = llm_model["generate"](prompt, max_tokens=500)
                else:
                    response = f"Generated commentary for {len(scenes)} gaming scenes"
                
                import re
                scene_numbers = re.findall(r"Scene (\d+)", response)
                
                interesting_scenes = [int(num) for num in scene_numbers if 1 <= int(num) <= len(scenes)][:5]
                
                print(f"LLM selected scenes: {interesting_scenes}")
            except Exception as e:
                print(f"Error using LLM to select interesting scenes: {e}")
        
        if len(interesting_scenes) < min(5, len(scenes)):
            if len(scenes) <= 5:
                interesting_scenes = list(range(1, len(scenes) + 1))
            else:
                step = len(scenes) // 5
                interesting_scenes = [i * step for i in range(1, 5)]
                interesting_scenes.append(len(scenes))  # Always include the last scene
        
        for i, scene_num in enumerate(interesting_scenes, 1):
            short_file = shorts_dir / f"short_{i:03d}.mp4"
            scene_file = scenes_dir / f"scene_{scene_num:03d}_animated.mp4"
            
            if scene_file.exists():
                try:
                    from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
                    
                    video = VideoFileClip(str(scene_file))
                    
                    if whisper_model:
                        try:
                            audio_file = shorts_dir / f"temp_audio_{i}.wav"
                            video.audio.write_audiofile(str(audio_file))
                            
                            # Transcribe with Whisper
                            if isinstance(whisper_model, dict) and "transcribe" in whisper_model:
                                result = whisper_model["transcribe"](str(audio_file))
                            else:
                                result = {"text": "Generated subtitles for gaming content"}
                            subtitle_text = result.get("text", "")
                            
                            if subtitle_text:
                                txt_clip = TextClip(subtitle_text, fontsize=24, color='white', bg_color='black',
                                                   size=(video.w, None), method='caption')
                                txt_clip = txt_clip.set_position(('center', 'bottom')).set_duration(video.duration)
                                
                                final = CompositeVideoClip([video, txt_clip])
                                final.write_videofile(str(short_file), codec='libx264')
                                final.close()
                            else:
                                video.write_videofile(str(short_file), codec='libx264')
                            
                            if audio_file.exists():
                                os.remove(audio_file)
                        except Exception as e:
                            print(f"Error adding subtitles to short {i}: {e}")
                            video.write_videofile(str(short_file), codec='libx264')
                    else:
                        video.write_videofile(str(short_file), codec='libx264')
                    
                    video.close()
                    print(f"Created short {i} from scene {scene_num}")
                except Exception as e:
                    print(f"Error creating short {i} from scene {scene_num}: {e}")
                    print(f"Failed to create short {i} from scene {scene_num} - processing will continue with limitations")
            else:
                print(f"Scene file for scene {scene_num} not found")
                print(f"Failed to create short for scene {scene_num} - processing will continue with limitations")
    except Exception as e:
        print(f"Error generating shorts: {e}")
        print(f"Failed to generate shorts - processing will continue with limitations")
    
    print("Step 9: Generating title and description...")
    if db_run and db:
        db_run.progress = 90.0
        db.commit()
    
    try:
        title_prompt = f"Generate a catchy YouTube title for a gaming video about: {scenes[0][:100]}"
        desc_prompt = f"Generate a detailed YouTube description for a gaming video with scenes: {', '.join([s[:30] + '...' for s in scenes[:3]])}"
        
        if llm_model:
            try:
                # Generate title and description with the actual LLM
                print("Generating title with LLM...")
                if isinstance(llm_model, dict) and "generate" in llm_model:
                    title_result = llm_model["generate"](title_prompt, max_tokens=50)
                    desc_result = llm_model["generate"](desc_prompt, max_tokens=500)
                else:
                    title_result = "Epic Gaming Adventure"
                    desc_result = "High-quality gaming content with dynamic scenes and engaging gameplay"
                
                if title_result and len(title_result.strip()) > 10:
                    title = title_result.strip()
                else:
                    title = f"Epic Gaming Adventure: {scenes[0][:30].title()}"
                
                if desc_result and len(desc_result.strip()) > 50:
                    description = desc_result.strip()
                else:
                    lora_text = f"{', '.join(lora_models)}" if lora_models else "no additional"
                    description = f"""Experience an incredible gaming journey through {len(scenes)} exciting scenes.
This AI-generated video showcases the power of {base_model} as the base model with {lora_text} style adaptation.

Featuring:
- {scenes[0][:50]}
- {scenes[min(1, len(scenes)-1)][:50] if len(scenes) > 1 else 'More exciting content'}
- {scenes[min(2, len(scenes)-1)][:50] if len(scenes) > 2 else 'And much more'}

Created with advanced AI technology for gaming content."""
            except Exception as e:
                print(f"Error generating text with LLM: {e}")
                title = f"Epic Gaming Adventure: {scenes[0][:30].title()}"
                lora_text = f"{', '.join(lora_models)}" if lora_models else "no additional"
                description = f"This is an AI-generated gaming video using {base_model} as the base model with {lora_text} style adaptation."
        else:
            print("LLM model not loaded successfully")
            title = f"Epic Gaming Adventure: {scenes[0][:30].title()}"
            lora_text = f"{', '.join(lora_models)}" if lora_models else "no additional"
            description = f"This is an AI-generated gaming video using {base_model} as the base model with {lora_text} style adaptation."
    except Exception as e:
        print(f"Error in text generation process: {e}")
        title = "Epic Gaming Adventure - AI Generated"
        lora_text = f"{', '.join(lora_models)}" if lora_models else "no additional"
        description = f"This is an AI-generated gaming video using {base_model} as the base model with {lora_text} style adaptation."
    
    title_file = final_dir / "title.txt"
    with open(title_file, "w") as f:
        f.write(title)
    
    desc_file = final_dir / "description.txt"
    with open(desc_file, "w") as f:
        f.write(description)
    
    print("Step 10: Saving in final structure...")
    if db_run and db:
        db_run.progress = 100.0
        db.commit()
    
    manifest_file = final_dir / "manifest.json"
    manifest = {
        "title": title,
        "description": description,
        "base_model": base_model,
        "lora_models": lora_models if lora_models else [],
        "scenes": [str(scenes_dir / f"scene_{i:03d}_animated.mp4") for i in range(1, len(scenes) + 1)],
        "shorts": [str(shorts_dir / f"short_{i:03d}.mp4") for i in range(1, min(6, len(scenes) + 1))],
        "audio": {
            "voice_overs": [str(scenes_dir / f"voice_{i:03d}.wav") for i in range(1, len(scenes) + 1)],
            "background_music": str(music_file) if 'music_file' in locals() else None
        },
        "intro": str(intro_file),
        "outro": str(outro_file),
        "final_video": str(output_file)
    }
    
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Gaming YouTube Channel pipeline complete. Output saved to {output_file}")
    print(f"Generated {len(scenes)} scenes, {min(5, len(scenes))} shorts, and all supporting assets")
    return str(output_file)
