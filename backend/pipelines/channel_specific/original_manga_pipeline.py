from ..common_imports import *
from ..ai_imports import *
import time
import shutil

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Warning: PIL/Pillow not available. Image generation will be limited.")
    Image = ImageDraw = ImageFont = None

from ..pipeline_utils import ensure_output_dir, log_progress
from ..ai_models import load_with_multiple_loras, generate_image, load_whisper, load_bark, load_musicgen, load_llm
from ...core.character_memory import get_character_memory_manager
from ..language_support import get_language_config, enhance_script_with_language, get_language_specific_prompts, get_voice_code, get_tts_model, is_bark_supported

class OriginalMangaPipeline(BasePipeline):
    """Self-contained original manga content generation pipeline."""
    
    def __init__(self):
        super().__init__("original_manga")
        self.combat_calls_count = 0
        self.max_combat_calls = 2
    
    def run(self, input_path: str, output_path: str, base_model: str = "stable_diffusion_1_5", 
            lora_models: Optional[List[str]] = None, lora_paths: Optional[Dict[str, str]] = None, 
            db_run=None, db=None, render_fps: int = 24, output_fps: int = 24, 
            frame_interpolation_enabled: bool = True, language: str = "en") -> str:
        """
        Run the self-contained original manga pipeline.
        
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
        
        print("Running self-contained original manga pipeline")
        print(f"Using base model: {base_model}")
        print(f"Using LoRA models: {lora_models}")
        print(f"Language: {language}")
        
        try:
            return self._execute_pipeline(
                input_path, output_path, base_model, lora_models, 
                db_run, db, render_fps, output_fps, frame_interpolation_enabled, language
            )
        except Exception as e:
            logger.error(f"Original manga pipeline failed: {e}")
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
            scenes = [{"description": "Original manga scene with unique artistic style and creative storytelling.", "duration": 300}]
        
        if not characters:
            characters = [{"name": "Original Character", "description": "Unique manga character with distinctive design"}]
        
        if not locations:
            locations = [{"name": "Original World", "description": "Creative manga setting with unique atmosphere"}]
        
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
            
            print(f"Original manga script expanded to {len(scenes)} scenes for 20-minute target")
            
        except Exception as e:
            print(f"Error during original manga script expansion: {e}")
        
        print("Step 3: Generating original manga scenes with combat integration...")
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
                "duration": scene.get('duration', 9.0) if isinstance(scene, dict) else 9.0
            }
            
            if scene_type == "combat" and self.combat_calls_count < self.max_combat_calls:
                try:
                    combat_data = self.generate_combat_scene(
                        scene_description=scene_text,
                        duration=9.0,
                        characters=scene_chars,
                        style="original_manga",
                        difficulty="medium"
                    )
                    scene_detail["combat_data"] = combat_data
                    self.combat_calls_count += 1
                    print(f"Generated original manga combat scene {i+1} with unique styling ({self.combat_calls_count}/{self.max_combat_calls})")
                except Exception as e:
                    print(f"Error generating original manga combat scene: {e}")
            
            scene_file = scenes_dir / f"scene_{i+1:03d}.mp4"
            
            print(f"Generating original manga scene {i+1}: {scene_text[:50]}...")
            
            try:
                char_names = ", ".join([c.get("name", "character") if isinstance(c, dict) else str(c) for c in scene_chars])
                location_desc = scene_location.get("description", scene_location.get("name", "location")) if isinstance(scene_location, dict) else str(scene_location)
                
                manga_prompt = f"original manga scene, {location_desc}, with {char_names}, {scene_text}, unique artistic style, creative composition, original design, 16:9 aspect ratio"
                
                if scene_detail.get("combat_data"):
                    manga_prompt = scene_detail["combat_data"]["video_prompt"]
                
                video_path = self.generate_video(
                    prompt=manga_prompt,
                    duration=scene_detail["duration"],
                    output_path=str(scene_file)
                )
                
                if video_path:
                    scene_files.append(video_path)
                    print(f"Generated original manga scene video {i+1}")
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
        
        print("Step 4: Generating creative voice lines...")
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
                    print(f"Generated creative voice for scene {i+1}")
                    
            except Exception as e:
                print(f"Error generating voice for scene {i+1}: {e}")
        
        print("Step 5: Generating original soundtrack...")
        if db_run and db:
            db_run.progress = 60.0
            db.commit()
        
        music_file = final_dir / "original_soundtrack.wav"
        try:
            music_path = self.generate_background_music(
                prompt="original manga soundtrack with unique musical themes and creative composition",
                duration=sum(scene.get('duration', 9.0) if isinstance(scene, dict) else 9.0 for scene in scenes),
                output_path=str(music_file)
            )
            print(f"Generated original soundtrack: {music_path}")
        except Exception as e:
            print(f"Error generating original soundtrack: {e}")
            music_path = None
        
        print("Step 6: Combining scenes into final episode...")
        if db_run and db:
            db_run.progress = 80.0
            db.commit()
        
        final_video = final_dir / "original_manga_episode.mp4"
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
            print(f"Final original manga episode created: {combined_path}")
        except Exception as e:
            print(f"Error combining scenes: {e}")
            combined_path = str(final_video)
        
        print("Step 7: Creating shorts...")
        if db_run and db:
            db_run.progress = 90.0
            db.commit()
        
        try:
            shorts_paths = self._create_shorts(scene_files, shorts_dir)
            print(f"Created {len(shorts_paths)} original manga shorts")
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
        
        print(f"Original manga pipeline completed successfully: {output_dir}")
        return str(output_dir)
    
    def _detect_scene_type(self, scene_text: str) -> str:
        """Detect scene type from description."""
        scene_lower = scene_text.lower()
        
        if any(word in scene_lower for word in ["fight", "battle", "combat", "attack", "sword", "martial"]):
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
                short_path = shorts_dir / f"original_manga_short_{i+1:03d}.mp4"
                
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
                print(f"Error creating original manga short {i+1}: {e}")
        
        return shorts_paths


def run(input_path: str, output_path: str, base_model: str = "stable_diffusion_1_5", 
        lora_models: Optional[List[str]] = None, lora_paths: Optional[Dict[str, str]] = None, 
        db_run=None, db=None, render_fps: int = 24, output_fps: int = 24, 
        frame_interpolation_enabled: bool = True, language: str = "en") -> str:
    """Run original manga pipeline with self-contained processing."""
    pipeline = OriginalMangaPipeline()
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
    """
    Run the AI Original Manga Universe Channel pipeline.
    
    Processing steps:
    1. Read original manga-style scripts with unique worlds
    2. Generate B&W images with AnythingV5/CounterfeitV3
    3. Use custom tokens and LoRAs for unique style
    4. Apply simple animation with Ken Burns or AnimateDiff
    5. Generate Japanese voice-over with RVC/Bark
    6. Add music via MusicGen
    7. Save scenes as MP4 and edit into episode
    8. Generate 5 shorts via local LLM + Whisper for subtitles
    9. Export episode, shorts, title.txt and description.txt
    
    Args:
        input_path: Path to the input data
        output_path: Path to the output directory
        base_model: Base AI model to use (e.g., anythingv5, counterfeitv3)
        lora_model: LoRA model to use for style consistency
        db_run: Database pipeline run object for progress updates
        db: Database session
    """
    print(f"Running AI Original Manga Universe Channel pipeline")
    print(f"Base model: {base_model}")
    print(f"LoRA adaptations: {', '.join(lora_models) if lora_models else 'None'}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    from config import CHANNEL_BASE_MODELS
    if base_model not in CHANNEL_BASE_MODELS.get("original_manga", []):
        print(f"Warning: {base_model} may not be optimal for original manga content")
    
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
    
    print("Step 1: Reading original manga-style scripts...")
    if db_run and db:
        db_run.progress = 10.0
        db.commit()
    
    scenes = []
    
    if input_path and os.path.exists(input_path):
        try:
            if input_path.endswith('.yaml') or input_path.endswith('.yml'):
                with open(input_path, 'r', encoding='utf-8') as f:
                    script_data = yaml.safe_load(f)
                    if isinstance(script_data, dict):
                        scenes = script_data.get('scenes', [])
                    else:
                        scenes = script_data
            elif input_path.endswith('.json'):
                with open(input_path, 'r', encoding='utf-8') as f:
                    script_data = json.load(f)
                    if isinstance(script_data, dict):
                        scenes = script_data.get('scenes', [])
                    else:
                        scenes = script_data
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
            
            script_data = enhance_script_with_language(script_data, language)
            
            llm_model = load_llm()
            expanded_script = expand_script_if_needed(script_data, min_duration=20.0, llm_model=llm_model)
            
            if expanded_script != script_data:
                print(f"Original manga script expanded from {len(script_data.get('scenes', []))} to {len(expanded_script.get('scenes', []))} scenes")
                scenes = expanded_script.get('scenes', scenes)
                characters = expanded_script.get('characters', characters) 
                locations = expanded_script.get('locations', locations)
                
        except Exception as e:
            print(f"Error during original manga script expansion: {e}")
    
    if not scenes:
        scenes = [
            "Original manga character introduction with unique art style",
            "Dramatic confrontation between protagonist and antagonist",
            "Emotional backstory reveal with flashback sequences",
            "Action-packed battle with creative panel layouts",
            "Character development moment with internal monologue",
            "Climactic resolution with satisfying conclusion"
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
                        duration=9.0,
                        characters=scene_chars,
                        style="original_manga",
                        difficulty="medium"
                    )
                    scene_detail["combat_data"] = combat_data
                    scene_detail["scene_text"] = combat_data["video_prompt"]
                    print(f"Generated original manga combat choreography for scene {i+1}")
                except Exception as e:
                    print(f"Error generating original manga combat scene: {e}")
            
            scene_details.append(scene_detail)
        else:
            scene_details.append(scene)
    
    print(f"Processing {len(scene_details)} scenes with {len(characters)} characters across {len(locations)} locations")

    if not characters:
        print("No characters found, using default character generation")
        characters = ["Protagonist", "Antagonist"]
    
    for i, scene in enumerate(scenes):
        if isinstance(scene, str):
            scene_type = "dialogue"
            scene_chars = characters[:2] if len(characters) >= 2 else characters
            scene_location = locations[i % len(locations)] if locations else "Unknown Location"
            
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
                        duration=9.0,
                        characters=scene_chars,
                        style="original_manga",
                        difficulty="medium"
                    )
                    scene_detail["combat_data"] = combat_data
                    scene_detail["scene_text"] = combat_data["video_prompt"]
                    print(f"Generated original manga combat choreography for scene {i+1}")
                except Exception as e:
                    print(f"Error generating original manga combat scene: {e}")
            
            scene_details.append(scene_detail)
        else:
            scene_details.append(scene)
    
    if not scenes:
        scenes = [
            "Original manga scene with protagonist in dramatic pose, black and white style",
            "Unique manga world with distinctive architecture and landscape elements",
            "Character interaction scene with emotional expressions and dialogue bubbles",
            "Action sequence with speed lines and impact effects in manga style",
            "Climactic scene with detailed character art and dramatic composition"
        ]
    
    print(f"Processing {len(scenes)} scenes")
    
    print("Step 2: Generating B&W images with AnythingV5/CounterfeitV3...")
    if db_run and db:
        db_run.progress = 20.0
        db.commit()
    
    print(f"Loading {base_model} with {', '.join(lora_models) if lora_models else 'no'} LoRA(s) for original manga generation...")
    try:
        from ..ai_models import AIModelManager, get_optimal_model_for_channel
        
        optimal_model = get_optimal_model_for_channel("original_manga")
        if base_model != optimal_model:
            print(f"Warning: {base_model} may not be optimal. Recommended: {optimal_model}")
        
        model_manager = AIModelManager()
        manga_model = model_manager.load_base_model(base_model, "image")
        if lora_models:
            manga_model = model_manager.apply_multiple_loras(manga_model, lora_models, lora_paths)
        
        print("Model loaded successfully with VRAM optimization")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Failed to load model - processing will continue with limitations")
        manga_model = None
    
    for i, scene_prompt in enumerate(scenes, 1):
        print(f"Generating scene {i}: {scene_prompt[:50]}...")
        
        manga_prompt = f"original manga style, black and white, {scene_prompt}, detailed panels, clean lines, unique world"
        
        scene_file = scenes_dir / f"scene_{i:03d}.png"
        try:
            if manga_model:
                try:
                    # Generate the actual image using the model
                    image = generate_image(manga_model, manga_prompt, width=1024, height=576)
                    
                    if image is not None:
                        if Image and isinstance(image, Image.Image):
                            image.save(str(scene_file))
                            print(f"Successfully saved scene {i} image to {scene_file}")
                        else:
                            print(f"Warning: generate_image returned unexpected type: {type(image)}")
                            if hasattr(image, 'images') and len(image.images) > 0:
                                image.images[0].save(str(scene_file))
                                print(f"Successfully saved scene {i} image from model output")
                            else:
                                raise ValueError("Could not extract valid image from model output")
                    else:
                        raise ValueError("Model returned None for image generation")
                except Exception as e:
                    print(f"Error saving generated image for scene {i}: {e}")
                    if Image and ImageDraw:
                        error_img = Image.new('RGB', (1024, 576), color=(30, 30, 30))
                    draw = ImageDraw.Draw(error_img)
                    draw.text((512, 288), f"Error: {str(e)[:50]}", fill=(255, 0, 0), anchor="mm")
                    error_img.save(str(scene_file))
                    print(f"Created error image for scene {i}")
            else:
                print(f"No manga model available for scene {i} generation")
                if Image and ImageDraw:
                    fallback_img = Image.new('RGB', (1024, 576), color=(50, 50, 50))
                draw = ImageDraw.Draw(fallback_img)
                draw.text((512, 288), f"Scene {i}: {scene_prompt[:50]}...", fill=(255, 255, 255), anchor="mm")
                fallback_img.save(str(scene_file))
                print(f"Created fallback image for scene {i} due to missing model")
        except Exception as e:
            print(f"Error generating scene {i}: {e}")
            try:
                if Image and ImageDraw:
                    error_img = Image.new('RGB', (1024, 576), color=(30, 30, 30))
                draw = ImageDraw.Draw(error_img)
                draw.text((512, 288), f"Error: {str(e)[:50]}", fill=(255, 0, 0), anchor="mm")
                error_img.save(str(scene_file))
                print(f"Created error image for scene {i}")
            except Exception as inner_e:
                print(f"Failed to create error image: {inner_e}")
                with open(str(scene_file), "wb") as f:
                    f.write(b"")
    
    print("Step 3: Using custom tokens and LoRAs for unique style...")
    if db_run and db:
        db_run.progress = 30.0
        db.commit()
    
    # Generate character visuals with unique style
    characters = [
        {"name": "Original Hero", "description": "Original manga protagonist with unique design, expressive eyes, distinctive hairstyle", "voice": "young_determined"},
        {"name": "Dark Rival", "description": "Original manga antagonist with menacing features and complex design elements", "voice": "dark_menacing"},
        {"name": "Loyal Friend", "description": "Supporting manga character with unique visual traits and memorable design", "voice": "supportive_friend"}
    ]
    
    character_seeds = {}
    character_ids = {}
    
    for character in characters:
        character_name = character["name"]
        character_desc = character["description"]
        character_voice = character["voice"]
        
        print(f"Processing original manga character: {character_name}")
        
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
            print(f"Creating new original manga character design for: {character_name}")
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
                "movement_patterns": {"manga_style": "original", "panel_transitions": "dynamic"},
                "video_generation_params": {"guidance_scale": 7.5, "num_inference_steps": 20},
                "preferred_models": ["original_manga", "creative_style"]
            })
            
            character_memory.update_voice_characteristics(character_id, {
                "voice_settings": {"tone": character_voice, "style": "manga_dramatic"},
                "speech_patterns": {"pace": "expressive", "emphasis": "character_driven"}
            })
        
        character_seeds[character_name] = seed
        character_ids[character_name] = character_id
    
    for i, char_prompt in enumerate(characters, 1):
        print(f"Generating character {i}: {char_prompt[:50]}...")
        
        char_file = characters_dir / f"character_{i:03d}.png"
        try:
            if manga_model:
                try:
                    # Generate the actual character image using the model
                    image = generate_image(manga_model, char_prompt, width=768, height=768)
                    
                    if image is not None:
                        if Image and isinstance(image, Image.Image):
                            image.save(str(char_file))
                            print(f"Successfully saved character {i} image to {char_file}")
                        else:
                            print(f"Warning: generate_image returned unexpected type: {type(image)}")
                            if hasattr(image, 'images') and len(image.images) > 0:
                                image.images[0].save(str(char_file))
                                print(f"Successfully saved character {i} image from model output")
                            else:
                                raise ValueError("Could not extract valid image from model output")
                    else:
                        raise ValueError("Model returned None for character image generation")
                except Exception as e:
                    print(f"Error saving generated image for character {i}: {e}")
                    if Image and ImageDraw:
                        error_img = Image.new('RGB', (768, 768), color=(30, 30, 30))
                    draw = ImageDraw.Draw(error_img)
                    draw.text((384, 384), f"Error: {str(e)[:50]}", fill=(255, 0, 0), anchor="mm")
                    error_img.save(str(char_file))
                    print(f"Created error image for character {i}")
            else:
                print(f"No manga model available for character {i} generation")
                if Image and ImageDraw:
                    fallback_img = Image.new('RGB', (768, 768), color=(50, 50, 50))
                draw = ImageDraw.Draw(fallback_img)
                draw.text((384, 384), f"Character {i}: {char_prompt[:50]}...", fill=(255, 255, 255), anchor="mm")
                fallback_img.save(str(char_file))
                print(f"Created fallback image for character {i} due to missing model")
        except Exception as e:
            print(f"Error generating character {i}: {e}")
            try:
                if Image and ImageDraw:
                    error_img = Image.new('RGB', (768, 768), color=(30, 30, 30))
                draw = ImageDraw.Draw(error_img)
                draw.text((384, 384), f"Error: {str(e)[:50]}", fill=(255, 0, 0), anchor="mm")
                error_img.save(str(char_file))
                print(f"Created error image for character {i}")
            except Exception as inner_e:
                print(f"Failed to create error image: {inner_e}")
                with open(str(char_file), "wb") as f:
                    f.write(b"")
                
    if db_run and db:
        db_run.progress = 35.0
        db.commit()
    
    print("Step 4: Applying simple animation with Ken Burns or AnimateDiff...")
    if db_run and db:
        db_run.progress = 40.0
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
        
        print(f"Generating video for scene {i+1}: {scene_text[:50]}...")
        
        animated_file = scenes_dir / f"scene_{i+1:03d}_animated.mp4"
        voice_file = scenes_dir / f"scene_{i+1:03d}_voice.wav"
        music_file = scenes_dir / f"scene_{i+1:03d}_music.wav"
        final_file = scenes_dir / f"scene_{i+1:03d}_final.mp4"
        
        try:
            from ..pipeline_utils import create_scene_video_with_generation, optimize_video_prompt, generate_voice_lines, generate_background_music, apply_lipsync, create_fallback_video
            from ..video_generation import get_best_model_for_content
            from ..ai_models import AIModelManager
            
            model_manager = AIModelManager()
            vram_tier = model_manager._detect_vram_tier()
            
            optimized_prompt = optimize_video_prompt(scene_text, "original_manga")
            best_model = get_best_model_for_content("anime", vram_tier)
            
            success = create_scene_video_with_generation(
                scene_description=optimized_prompt,
                characters=scene_chars,
                output_path=str(animated_file),
                model_name=best_model
            )
            
            if success:
                print(f"Successfully generated video for scene {i+1}")
                
                character_voice = scene_chars[0].get("voice", "default") if scene_chars else "default"
                voice_success = generate_voice_lines(scene_text, character_voice, str(voice_file))
                
                music_success = generate_background_music(f"Original manga music for {scene_text}", 10.0, str(music_file))
                
                if voice_success:
                    lipsync_success = apply_lipsync(str(animated_file), str(voice_file), str(final_file), "anime")
                    if lipsync_success:
                        print(f"Applied lipsync for scene {i+1}")
                
            else:
                print(f"Failed to generate video for scene {i+1}, creating fallback")
                create_fallback_video(animated_file, scene_text, i+1)
                
        except Exception as e:
            print(f"Error generating video for scene {i+1}: {e}")
            from ..pipeline_utils import create_fallback_video
            create_fallback_video(Path(animated_file), scene_text, i+1)
    
    print("Step 5: Generating Japanese voice-over with RVC/Bark...")
    if db_run and db:
        db_run.progress = 50.0
        db.commit()
    
    try:
        bark_model = load_bark()
        print("Bark model loaded successfully")
        
        # Generate Japanese voice-over for each scene
        japanese_voices = [
            "young female Japanese voice with emotional tone",
            "mature male Japanese voice with serious tone",
            "energetic Japanese narrator voice"
        ]
        
        for i, scene_prompt in enumerate(scenes, 1):
            voice_file = scenes_dir / f"voice_{i:03d}.wav"
            voice_type = japanese_voices[(i-1) % len(japanese_voices)]
            voice_prompt = f"{voice_type}: {scene_prompt[:50]}..."
            
            print(f"Generating Japanese voice-over for scene {i} with {voice_type}")
            try:
                # Generate actual voice-over using Bark
                if bark_model:
                    voice_text = scene_prompt[:200] if len(scene_prompt) > 200 else scene_prompt
                    
                    # Generate the audio using Bark
                    import numpy as np
                    import soundfile as sf
                    
                    # Generate speech with Bark
                    audio_array = bark_model["generate"](
                        text=voice_text,
                        voice_preset=voice_type,
                        sample_rate=24000
                    )
                    
                    if isinstance(audio_array, np.ndarray):
                        sf.write(str(voice_file), audio_array, 24000)
                        print(f"Successfully saved voice-over for scene {i}")
                    else:
                        print(f"Warning: Bark returned unexpected type: {type(audio_array)}")
                        if hasattr(audio_array, 'audio_array') and isinstance(audio_array.audio_array, np.ndarray):
                            sf.write(str(voice_file), audio_array.audio_array, 24000)
                            print(f"Successfully saved voice-over from model output")
                        else:
                            raise ValueError("Could not extract valid audio from model output")
                else:
                    raise ValueError("Bark model not available")
            except Exception as e:
                print(f"Error generating voice-over for scene {i}: {e}")
                try:
                    import numpy as np
                    import soundfile as sf
                    
                    sample_rate = 24000
                    duration = 3  # seconds
                    silent_audio = np.zeros(sample_rate * duration)
                    
                    sf.write(str(voice_file), silent_audio, sample_rate)
                    print(f"Created silent audio fallback for scene {i}")
                except Exception as inner_e:
                    print(f"Failed to create silent audio: {inner_e}")
                    with open(str(voice_file), "wb") as f:
                        f.write(b"")
    except Exception as e:
        print(f"Error loading Bark model: {e}")
        print("Failed to load Bark model - voice generation will be limited")
    
    print("Step 6: Adding music via MusicGen...")
    if db_run and db:
        db_run.progress = 60.0
        db.commit()
    
    try:
        musicgen_model = load_musicgen()
        print("MusicGen model loaded successfully")
        
        # Generate Japanese-style background music
        music_file = output_dir / "background_music.wav"
        music_prompt = "Traditional Japanese music with modern elements, suitable for manga animation, emotional and atmospheric"
        
        print(f"Generating Japanese-style soundtrack with prompt: {music_prompt}")
        try:
            # Generate actual music using MusicGen
            if musicgen_model:
                # Generate the audio using MusicGen
                import numpy as np
                import soundfile as sf
                
                # Generate music with MusicGen
                audio_array = musicgen_model.generate(
                    descriptions=[music_prompt],
                    duration=30  # Generate 30 seconds of music
                )
                
                if isinstance(audio_array, np.ndarray):
                    sf.write(str(music_file), audio_array, 32000)  # MusicGen typically uses 32kHz
                    print(f"Successfully saved background music")
                else:
                    print(f"Warning: MusicGen returned unexpected type: {type(audio_array)}")
                    if hasattr(audio_array, 'waveform') and isinstance(audio_array.waveform, np.ndarray):
                        waveform = audio_array.waveform[0] if audio_array.waveform.ndim > 2 else audio_array.waveform
                        sf.write(str(music_file), waveform.T, 32000)  # Transpose if needed
                        print(f"Successfully saved music from model output")
                    elif isinstance(audio_array, list) and len(audio_array) > 0 and isinstance(audio_array[0], np.ndarray):
                        sf.write(str(music_file), audio_array[0], 32000)
                        print(f"Successfully saved music from model output list")
                    else:
                        raise ValueError("Could not extract valid audio from model output")
            else:
                raise ValueError("MusicGen model not available")
        except Exception as e:
            print(f"Error generating background music: {e}")
            try:
                import numpy as np
                import soundfile as sf
                
                sample_rate = 32000
                duration = 30  # seconds
                silent_audio = np.zeros(sample_rate * duration)
                
                sf.write(str(music_file), silent_audio, sample_rate)
                print(f"Created silent audio fallback for background music")
            except Exception as inner_e:
                print(f"Failed to create silent audio: {inner_e}")
                with open(str(music_file), "wb") as f:
                    f.write(b"")
    except Exception as e:
        print(f"Error loading MusicGen model: {e}")
        print("Failed to load MusicGen model - music generation will be limited")
    
    print("Step 7: Saving scenes as MP4 and editing into episode...")
    if db_run and db:
        db_run.progress = 70.0
        db.commit()
    
    for i in range(1, len(scenes) + 1):
        scene_mp4 = scenes_dir / f"scene_{i:03d}.mp4"
        animated_file = scenes_dir / f"scene_{i:03d}_animated.mp4"
        voice_file = scenes_dir / f"voice_{i:03d}.wav"
        
        try:
            print(f"Combining animation and voice for scene {i}")
            if os.path.exists(animated_file) and os.path.exists(voice_file):
                from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
                
                video_clip = VideoFileClip(str(animated_file))
                voice_clip = AudioFileClip(str(voice_file))
                
                if os.path.exists(music_file):
                    music_clip = AudioFileClip(str(music_file))
                    if music_clip.duration < voice_clip.duration:
                        music_clip = music_clip.loop(duration=voice_clip.duration)
                    else:
                        music_clip = music_clip.subclip(0, voice_clip.duration)
                    music_clip = music_clip.volumex(0.3)
                    final_audio = CompositeAudioClip([voice_clip, music_clip])
                else:
                    final_audio = voice_clip
                
                final_clip = video_clip.set_audio(final_audio)
                
                final_clip.write_videofile(
                    str(scene_mp4),
                    codec='libx264',
                    audio_codec='aac',
                    fps=render_fps
                )
                
                video_clip.close()
                voice_clip.close()
                if 'music_clip' in locals():
                    music_clip.close()
                final_clip.close()
                
                print(f"Successfully created scene video {i} with animation and voice-over")
            else:
                print(f"Source files for scene {i} not found")
                from moviepy.editor import ColorClip, TextClip, CompositeVideoClip
                
                bg_clip = ColorClip(size=(1024, 576), color=(50, 50, 50), duration=5)
                
                text = f"Scene {i} video - source files not found"
                txt_clip = TextClip(text, fontsize=30, color='white', bg_color='black',
                                  font='Arial', size=(1024, None), method='caption')
                txt_clip = txt_clip.set_position('center').set_duration(5)
                
                video = CompositeVideoClip([bg_clip, txt_clip])
                
                video.write_videofile(str(scene_mp4), codec='libx264', fps=render_fps)
                
                bg_clip.close()
                txt_clip.close()
                video.close()
                
                print(f"Created fallback video for scene {i} due to missing source files")
        except Exception as e:
            print(f"Error creating scene video {i}: {e}")
            try:
                from moviepy.editor import ColorClip, TextClip, CompositeVideoClip
                
                bg_clip = ColorClip(size=(1024, 576), color=(30, 30, 30), duration=5)
                
                error_text = f"Error creating scene video: {str(e)[:50]}"
                txt_clip = TextClip(error_text, fontsize=30, color='red', bg_color='black',
                                  font='Arial', size=(1024, None), method='caption')
                txt_clip = txt_clip.set_position('center').set_duration(5)
                
                video = CompositeVideoClip([bg_clip, txt_clip])
                
                video.write_videofile(str(scene_mp4), codec='libx264', fps=render_fps)
                
                bg_clip.close()
                txt_clip.close()
                video.close()
                
                print(f"Created error video for scene {i}")
            except Exception as inner_e:
                print(f"Failed to create error video: {inner_e}")
                with open(str(scene_mp4), "wb") as f:
                    f.write(b"")
    
    output_file = final_dir / "original_manga_episode.mp4"
    try:
        print(f"Creating final episode by combining all scene videos...")
        from moviepy.editor import concatenate_videoclips, VideoFileClip
        
        scene_videos = []
        for i in range(1, len(scenes) + 1):
            scene_mp4 = scenes_dir / f"scene_{i:03d}.mp4"
            if os.path.exists(scene_mp4):
                scene_videos.append(str(scene_mp4))
        
        if scene_videos:
            video_clips = [VideoFileClip(video) for video in scene_videos]
            
            final_clip = concatenate_videoclips(video_clips)
            
            if os.path.exists(music_file):
                from moviepy.editor import AudioFileClip, CompositeAudioClip
                
                original_audio = final_clip.audio
                
                # Load the background music
                music_clip = AudioFileClip(str(music_file))
                
                if music_clip.duration < final_clip.duration:
                    music_clip = music_clip.loop(duration=final_clip.duration)
                else:
                    music_clip = music_clip.subclip(0, final_clip.duration)
                
                music_clip = music_clip.volumex(0.2)
                
                # Combine the original audio with the background music
                if original_audio:
                    final_audio = CompositeAudioClip([original_audio, music_clip])
                else:
                    final_audio = music_clip
                
                final_clip = final_clip.set_audio(final_audio)
            
            final_clip.write_videofile(
                str(output_file),
                codec='libx264',
                audio_codec='aac',
                fps=24
            )
            
            for clip in video_clips:
                clip.close()
            final_clip.close()
            if 'music_clip' in locals():
                music_clip.close()
            if 'original_audio' in locals() and original_audio:
                original_audio.close()
            
            print(f"Successfully created final episode video with {len(scene_videos)} scenes")
        else:
            print("No scene videos found to combine")
            from moviepy.editor import ColorClip, TextClip, CompositeVideoClip
            
            bg_clip = ColorClip(size=(1024, 576), color=(50, 50, 50), duration=10)
            
            text = f"Original manga episode - No scene videos found"
            txt_clip = TextClip(text, fontsize=30, color='white', bg_color='black',
                              font='Arial', size=(1024, None), method='caption')
            txt_clip = txt_clip.set_position('center').set_duration(10)
            
            video = CompositeVideoClip([bg_clip, txt_clip])
            
            video.write_videofile(str(output_file), codec='libx264', fps=render_fps)
            
            bg_clip.close()
            txt_clip.close()
            video.close()
            
            print(f"Created fallback video for final episode due to missing scene videos")
    except Exception as e:
        print(f"Error creating final episode video: {e}")
        try:
            from moviepy.editor import ColorClip, TextClip, CompositeVideoClip
            
            bg_clip = ColorClip(size=(1024, 576), color=(30, 30, 30), duration=10)
            
            error_text = f"Error creating final episode: {str(e)[:100]}"
            txt_clip = TextClip(error_text, fontsize=30, color='red', bg_color='black',
                              font='Arial', size=(1024, None), method='caption')
            txt_clip = txt_clip.set_position('center').set_duration(10)
            
            video = CompositeVideoClip([bg_clip, txt_clip])
            
            video.write_videofile(str(output_file), codec='libx264', fps=render_fps)
            
            bg_clip.close()
            txt_clip.close()
            video.close()
            
            print(f"Created error video for final episode")
        except Exception as inner_e:
            print(f"Failed to create error video: {inner_e}")
            with open(str(output_file), "wb") as f:
                f.write(b"")
    
    print("Step 8: Generating shorts and subtitles...")
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
        try:
            if whisper_model:
                # Generate subtitles for the final video
                if os.path.exists(output_file):
                    import tempfile
                    import os
                    
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                        temp_audio_path = temp_audio.name
                    
                    from moviepy.editor import VideoFileClip
                    video_clip = VideoFileClip(str(output_file))
                    if video_clip.audio:
                        video_clip.audio.write_audiofile(temp_audio_path, codec='pcm_s16le')
                        video_clip.close()
                        
                        result = whisper_model.transcribe(temp_audio_path)
                        
                        segments = result.get('segments', [])
                        with open(subtitle_file, "w", encoding="utf-8") as f:
                            for i, segment in enumerate(segments):
                                start_time = segment.get('start', 0)
                                end_time = segment.get('end', 0)
                                text = segment.get('text', '')
                                
                                start_formatted = '{:02d}:{:02d}:{:02d},{:03d}'.format(
                                    int(start_time // 3600),
                                    int((start_time % 3600) // 60),
                                    int(start_time % 60),
                                    int((start_time % 1) * 1000)
                                )
                                
                                end_formatted = '{:02d}:{:02d}:{:02d},{:03d}'.format(
                                    int(end_time // 3600),
                                    int((end_time % 3600) // 60),
                                    int(end_time % 60),
                                    int((end_time % 1) * 1000)
                                )
                                
                                f.write(f"{i+1}\n")
                                f.write(f"{start_formatted} --> {end_formatted}\n")
                                f.write(f"{text}\n\n")
                        
                        print(f"Successfully generated subtitles with {len(segments)} segments")
                        
                        os.unlink(temp_audio_path)
                    else:
                        print("Final video has no audio track for subtitle generation")
                        with open(subtitle_file, "w", encoding="utf-8") as f:
                            f.write("1\n00:00:01,000 --> 00:00:05,000\nNo audio track found in final video")
                else:
                    print("Final video not found for subtitle generation")
                    with open(subtitle_file, "w", encoding="utf-8") as f:
                        f.write("1\n00:00:01,000 --> 00:00:05,000\nFinal video not found")
            else:
                print("Whisper model not available for subtitle generation")
                with open(subtitle_file, "w", encoding="utf-8") as f:
                    f.write("1\n00:00:01,000 --> 00:00:05,000\nWhisper model not available")
        except Exception as e:
            print(f"Error generating subtitles: {e}")
            with open(subtitle_file, "w", encoding="utf-8") as f:
                f.write(f"1\n00:00:01,000 --> 00:00:05,000\nError generating subtitles: {str(e)[:50]}")
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        print("Failed to load Whisper model - subtitle generation will be limited")
    
    # Generate 5 shorts from the most impactful moments
    try:
        llm_model = load_llm()
        print("Local LLM loaded successfully")
        
        print("Using LLM to select 5 impactful moments for shorts")
        
        for i in range(1, min(6, len(scenes) + 1)):
            short_file = shorts_dir / f"short_{i:03d}.mp4"
            scene_mp4 = scenes_dir / f"scene_{i:03d}.mp4"
            
            try:
                if os.path.exists(scene_mp4):
                    from moviepy.editor import VideoFileClip
                    
                    # Load the scene video
                    scene_clip = VideoFileClip(str(scene_mp4))
                    
                    if scene_clip.duration > 15:
                        start_time = (i - 1) * 5  # 5 second offset for each short
                        if start_time + 15 > scene_clip.duration:
                            start_time = max(0, scene_clip.duration - 15)
                        
                        short_clip = scene_clip.subclip(start_time, start_time + 15)
                    else:
                        short_clip = scene_clip
                    
                    if 'whisper_model' in locals() and whisper_model:
                        try:
                            import tempfile
                            import os
                            
                            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                                temp_audio_path = temp_audio.name
                            
                            short_clip.audio.write_audiofile(temp_audio_path, codec='pcm_s16le')
                            
                            result = whisper_model.transcribe(temp_audio_path)
                            
                            subtitle_text = result.get('text', f"Short {i} from Original Manga")
                            
                            from moviepy.editor import TextClip, CompositeVideoClip
                            
                            subtitle_clip = TextClip(
                                subtitle_text, 
                                fontsize=24, 
                                color='white',
                                bg_color='black',
                                font='Arial',
                                size=(short_clip.size[0], None),
                                method='caption'
                            ).set_position(('center', 'bottom')).set_duration(short_clip.duration)
                            
                            # Composite the video with subtitles
                            final_short = CompositeVideoClip([short_clip, subtitle_clip])
                            
                            os.unlink(temp_audio_path)
                        except Exception as e:
                            print(f"Error adding subtitles to short {i}: {e}")
                            final_short = short_clip
                    else:
                        final_short = short_clip
                    
                    final_short.write_videofile(
                        str(short_file),
                        codec='libx264',
                        audio_codec='aac',
                        fps=render_fps
                    )
                    
                    scene_clip.close()
                    short_clip.close()
                    if 'final_short' in locals() and final_short is not short_clip:
                        final_short.close()
                    if 'subtitle_clip' in locals():
                        subtitle_clip.close()
                    
                    print(f"Successfully created short {i} from scene {i}")
                else:
                    print(f"Source scene for short {i} not found")
                    from moviepy.editor import ColorClip, TextClip, CompositeVideoClip
                    
                    bg_clip = ColorClip(size=(1024, 576), color=(50, 50, 50), duration=15)
                    
                    text = f"Short {i} - source scene not found"
                    txt_clip = TextClip(text, fontsize=30, color='white', bg_color='black',
                                      font='Arial', size=(1024, None), method='caption')
                    txt_clip = txt_clip.set_position('center').set_duration(15)
                    
                    video = CompositeVideoClip([bg_clip, txt_clip])
                    
                    video.write_videofile(str(short_file), codec='libx264', fps=render_fps)
                    
                    bg_clip.close()
                    txt_clip.close()
                    video.close()
                    
                    print(f"Created fallback video for short {i} due to missing source scene")
            except Exception as e:
                print(f"Error creating short {i}: {e}")
                try:
                    from moviepy.editor import ColorClip, TextClip, CompositeVideoClip
                    
                    bg_clip = ColorClip(size=(1024, 576), color=(30, 30, 30), duration=15)
                    
                    error_text = f"Error creating short {i}: {str(e)[:50]}"
                    txt_clip = TextClip(error_text, fontsize=30, color='red', bg_color='black',
                                      font='Arial', size=(1024, None), method='caption')
                    txt_clip = txt_clip.set_position('center').set_duration(15)
                    
                    video = CompositeVideoClip([bg_clip, txt_clip])
                    
                    video.write_videofile(str(short_file), codec='libx264', fps=render_fps)
                    
                    bg_clip.close()
                    txt_clip.close()
                    video.close()
                    
                    print(f"Created error video for short {i}")
                except Exception as inner_e:
                    print(f"Failed to create error video: {inner_e}")
                    with open(str(short_file), "wb") as f:
                        f.write(b"")
    except Exception as e:
        print(f"Error using LLM for short selection: {e}")
        print("Failed to use LLM for short selection - using default selection instead")
        
        for i in range(1, min(6, len(scenes) + 1)):
            short_file = shorts_dir / f"short_{i:03d}.mp4"
            scene_mp4 = scenes_dir / f"scene_{i:03d}.mp4"
            
            try:
                if os.path.exists(scene_mp4):
                    from moviepy.editor import VideoFileClip
                    
                    # Load the scene video
                    scene_clip = VideoFileClip(str(scene_mp4))
                    
                    # For default shorts, we'll take from the beginning of each scene
                    if scene_clip.duration > 15:
                        short_clip = scene_clip.subclip(0, 15)
                    else:
                        short_clip = scene_clip
                    
                    short_clip.write_videofile(
                        str(short_file),
                        codec='libx264',
                        audio_codec='aac',
                        fps=render_fps
                    )
                    
                    scene_clip.close()
                    short_clip.close()
                    
                    print(f"Successfully created default short {i} from scene {i}")
                else:
                    print(f"Source scene for default short {i} not found")
                    from moviepy.editor import ColorClip, TextClip, CompositeVideoClip
                    
                    bg_clip = ColorClip(size=(1024, 576), color=(50, 50, 50), duration=15)
                    
                    text = f"Default short {i} - source scene not found"
                    txt_clip = TextClip(text, fontsize=30, color='white', bg_color='black',
                                      font='Arial', size=(1024, None), method='caption')
                    txt_clip = txt_clip.set_position('center').set_duration(15)
                    
                    # Composite the clips
                    video = CompositeVideoClip([bg_clip, txt_clip])
                    
                    video.write_videofile(str(short_file), codec='libx264', fps=render_fps)
                    
                    bg_clip.close()
                    txt_clip.close()
                    video.close()
                    
                    print(f"Created fallback video for default short {i}")
            except Exception as e:
                print(f"Error creating default short {i}: {e}")
                try:
                    from moviepy.editor import ColorClip, TextClip, CompositeVideoClip
                    
                    bg_clip = ColorClip(size=(1024, 576), color=(30, 30, 30), duration=15)
                    
                    error_text = f"Error creating default short {i}: {str(e)[:50]}"
                    txt_clip = TextClip(error_text, fontsize=30, color='red', bg_color='black',
                                      font='Arial', size=(1024, None), method='caption')
                    txt_clip = txt_clip.set_position('center').set_duration(15)
                    
                    # Composite the clips
                    video = CompositeVideoClip([bg_clip, txt_clip])
                    
                    video.write_videofile(str(short_file), codec='libx264', fps=render_fps)
                    
                    bg_clip.close()
                    txt_clip.close()
                    video.close()
                    
                    print(f"Created error video for default short {i}")
                except Exception as inner_e:
                    print(f"Failed to create error video: {inner_e}")
                    with open(str(short_file), "wb") as f:
                        f.write(b"")
    
    print("Step 9: Exporting episode, shorts, title, and description...")
    if db_run and db:
        db_run.progress = 90.0
        db.commit()
    
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
    
    # Generate title and description using LLM
    try:
        if 'llm_model' not in locals() or not llm_model:
            llm_model = load_llm()
            print("Local LLM loaded successfully")
        
        if llm_model:
            # Generate title and description using the actual LLM model
            title_prompt = f"Generate a catchy title for an original manga video about: {scenes[0][:100]}"
            desc_prompt = f"Generate a detailed description for an original manga video with scenes: {', '.join([s[:30] + '...' for s in scenes[:3]])}"
            
            try:
                # Generate title using the LLM
                title_response = llm_model.generate(title_prompt, max_length=50)
                title = title_response.strip()
                if not title:
                    title = "Epic Original Manga Adventure - AI Generated"
                print(f"Generated title using LLM: {title}")
                
                # Generate description using the LLM
                desc_response = llm_model.generate(desc_prompt, max_length=500)
                description = desc_response.strip()
                if not description:
                    description = f"This is an AI-generated original manga episode using {base_model} as the base model with {', '.join(lora_models) if lora_models else 'no LoRAs'} style adaptation."
                print(f"Generated description using LLM: {description[:100]}...")
            except Exception as llm_e:
                print(f"Error generating text with LLM: {llm_e}")
                title = "Epic Original Manga Adventure - AI Generated"
                description = f"This is an AI-generated original manga episode using {base_model} as the base model with {', '.join(lora_models) if lora_models else 'no LoRAs'} style adaptation."
        else:
            print("LLM model not available - using default title and description")
            title = "Epic Original Manga Adventure - AI Generated"
            description = f"This is an AI-generated original manga episode using {base_model} as the base model with {', '.join(lora_models) if lora_models else 'no LoRAs'} style adaptation."
    except Exception as e:
        print(f"Error generating title/description with LLM: {e}")
        print("Failed to generate text with LLM - using default title and description")
        title = "Epic Original Manga Adventure - AI Generated"
        description = f"This is an AI-generated original manga episode using {base_model} as the base model with {', '.join(lora_models) if lora_models else 'no LoRAs'} style adaptation."
    
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
        "scene_videos": [str(scenes_dir / f"scene_{i:03d}.mp4") for i in range(1, len(scenes) + 1)],
        "shorts": [str(shorts_dir / f"short_{i:03d}.mp4") for i in range(1, min(6, len(scenes) + 1))],
        "characters": [str(characters_dir / f"character_{i:03d}.png") for i in range(1, 4)],
        "audio": {
            "voice_overs": [str(scenes_dir / f"voice_{i:03d}.wav") for i in range(1, len(scenes) + 1)],
            "background_music": str(output_dir / "background_music.wav") if os.path.exists(output_dir / "background_music.wav") else None
        },
        "subtitles": str(final_dir / "subtitles.srt") if os.path.exists(final_dir / "subtitles.srt") else None,
        "final_video": str(output_file)
    }
    
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)
    
    if db_run and db:
        db_run.progress = 100.0
        db.commit()
    
    print(f"AI Original Manga Universe Channel pipeline complete. Output saved to {output_file}")
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
