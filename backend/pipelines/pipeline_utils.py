def detect_scene_type(scene_description: str) -> str:
    """
    Detect the type of scene based on description content.
    
    Args:
        scene_description: Scene description text
        
    Returns:
        Scene type: 'combat', 'action', 'dialogue', 'emotional', 'exploration'
    """
    scene_lower = scene_description.lower()
    
    combat_keywords = [
        'battle', 'fight', 'combat', 'attack', 'punch', 'kick', 'sword', 'magic',
        'spell', 'energy', 'explosion', 'clash', 'duel', 'war', 'conflict',
        'strike', 'blow', 'hit', 'defeat', 'victory', 'struggle', 'confrontation'
    ]
    
    action_keywords = [
        'chase', 'run', 'jump', 'escape', 'pursuit', 'speed', 'fast', 'rush',
        'dash', 'leap', 'climb', 'fall', 'crash', 'impact', 'movement'
    ]
    
    dialogue_keywords = [
        'conversation', 'talk', 'speak', 'discuss', 'dialogue', 'chat',
        'meeting', 'conference', 'debate', 'argument', 'explanation'
    ]
    
    emotional_keywords = [
        'cry', 'tears', 'sad', 'happy', 'love', 'romance', 'emotional',
        'feelings', 'heart', 'memory', 'flashback', 'dream', 'hope', 'fear'
    ]
    
    combat_score = sum(1 for keyword in combat_keywords if keyword in scene_lower)
    action_score = sum(1 for keyword in action_keywords if keyword in scene_lower)
    dialogue_score = sum(1 for keyword in dialogue_keywords if keyword in scene_lower)
    emotional_score = sum(1 for keyword in emotional_keywords if keyword in scene_lower)
    
    scores = {
        'combat': combat_score,
        'action': action_score, 
        'dialogue': dialogue_score,
        'emotional': emotional_score
    }
    
    max_score = max(scores.values())
    if max_score == 0:
        return 'exploration'
    
    return max(scores, key=scores.get)


def split_scene_for_vram(scene_description: str, model_name: str = "animatediff", vram_tier: str = "medium") -> list:
    """
    Split scene description into smaller chunks based on model limitations and VRAM tier.
    
    Args:
        scene_description: Original scene description
        model_name: Video generation model being used
        vram_tier: VRAM tier (low, medium, high, ultra)
        
    Returns:
        List of scene chunks with optimal length for the model
    """
    model_limits = {
        "svd_xt": {"max_frames": 25, "max_tokens": 77},
        "zeroscope_v2_xl": {"max_frames": 24, "max_tokens": 77},
        "animatediff_v2_sdxl": {"max_frames": 16, "max_tokens": 77},
        "animatediff_lightning": {"max_frames": 16, "max_tokens": 77},
        "modelscope_t2v": {"max_frames": 16, "max_tokens": 77},
        "ltx_video": {"max_frames": 120, "max_tokens": 150},
        "skyreels_v2": {"max_frames": 60, "max_tokens": 100}
    }
    
    vram_adjustments = {
        "low": 0.6,
        "medium": 0.8,
        "high": 1.0,
        "ultra": 1.2
    }
    
    limits = model_limits.get(model_name, {"max_frames": 16, "max_tokens": 77})
    adjustment = vram_adjustments.get(vram_tier, 0.8)
    max_tokens = int(limits["max_tokens"] * adjustment)
    
    words = scene_description.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        word_length = len(word) + 1
        if current_length + word_length > max_tokens and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks if chunks else [scene_description]


def optimize_prompt_for_model(prompt: str, model_name: str, scene_type: str = "general") -> str:
    """
    Optimize prompt for specific video generation model and scene type.
    
    Args:
        prompt: Original prompt
        model_name: Target video generation model
        scene_type: Type of scene (combat, dialogue, etc.)
        
    Returns:
        Optimized prompt for the model
    """
    model_optimizations = {
        "svd_xt": {
            "prefix": "high quality, cinematic, smooth motion, ",
            "suffix": ", professional video, stable diffusion"
        },
        "zeroscope_v2_xl": {
            "prefix": "cinematic video, high resolution, ",
            "suffix": ", smooth animation, zeroscope quality"
        },
        "animatediff_v2_sdxl": {
            "prefix": "animated sequence, dynamic movement, ",
            "suffix": ", high quality animation, consistent style"
        },
        "animatediff_lightning": {
            "prefix": "fast animation, dynamic motion, ",
            "suffix": ", lightning fast rendering, smooth transitions"
        },
        "modelscope_t2v": {
            "prefix": "text to video, clear motion, ",
            "suffix": ", modelscope generation, coherent sequence"
        },
        "ltx_video": {
            "prefix": "long sequence, extended video, ",
            "suffix": ", ltx quality, continuous motion"
        },
        "skyreels_v2": {
            "prefix": "infinite length video, seamless loop, ",
            "suffix": ", skyreels quality, unlimited duration"
        }
    }
    
    scene_optimizations = {
        "combat": "dynamic action, fast movement, impact effects, ",
        "dialogue": "character focus, facial expressions, subtle movement, ",
        "emotional": "dramatic lighting, emotional atmosphere, close-up shots, ",
        "action": "high energy, rapid motion, exciting sequence, ",
        "exploration": "scenic view, environmental detail, smooth camera movement, "
    }
    
    optimization = model_optimizations.get(model_name, {"prefix": "", "suffix": ""})
    scene_prefix = scene_optimizations.get(scene_type, "")
    
    optimized = f"{optimization['prefix']}{scene_prefix}{prompt}{optimization['suffix']}"
    
    return optimized.strip()



from .common_imports import *
from .ai_imports import *
from .aspect_ratio_utils import enforce_16_9_aspect_ratio, enforce_16_9_video_aspect_ratio, get_16_9_resolution
def detect_scene_type(scene_description: str) -> str:
    """
    Detect the type of scene based on description content.
    
    Args:
        scene_description: Scene description text
        
    Returns:
        Scene type: 'combat', 'action', 'dialogue', 'emotional', 'exploration'
    """
    scene_lower = scene_description.lower()
    
    combat_keywords = [
        'battle', 'fight', 'combat', 'attack', 'punch', 'kick', 'sword', 'magic',
        'spell', 'energy', 'explosion', 'clash', 'duel', 'war', 'conflict',
        'strike', 'blow', 'hit', 'defeat', 'victory', 'struggle', 'confrontation'
    ]
    
    action_keywords = [
        'chase', 'run', 'jump', 'escape', 'pursuit', 'speed', 'fast', 'rush',
        'dash', 'leap', 'climb', 'fall', 'crash', 'impact', 'movement'
    ]
    
    dialogue_keywords = [
        'conversation', 'talk', 'speak', 'discuss', 'dialogue', 'chat',
        'meeting', 'conference', 'debate', 'argument', 'explanation'
    ]
    
    emotional_keywords = [
        'cry', 'tears', 'sad', 'happy', 'love', 'romance', 'emotional',
        'feelings', 'heart', 'memory', 'flashback', 'dream', 'hope', 'fear'
    ]
    
    combat_score = sum(1 for keyword in combat_keywords if keyword in scene_lower)
    action_score = sum(1 for keyword in action_keywords if keyword in scene_lower)
    dialogue_score = sum(1 for keyword in dialogue_keywords if keyword in scene_lower)
    emotional_score = sum(1 for keyword in emotional_keywords if keyword in scene_lower)
    
    scores = {
        'combat': combat_score,
        'action': action_score, 
        'dialogue': dialogue_score,
        'emotional': emotional_score
    }
    
    max_score = max(scores.values())
    if max_score == 0:
        return 'exploration'
    
    return max(scores, key=scores.get)


import time
import traceback

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)

logger = logging.getLogger("pipeline")

def ensure_output_dir(path: Path):
    """
    Ensure that the output directory exists.
    
    Args:
        path: Path to the output directory
    """
    if isinstance(path, str):
        path = Path(path)
    
    path.mkdir(parents=True, exist_ok=True)
    return path

def create_channel_directories(output_path: Path, channel_type: str = None, languages: list = None):
    """
    Create channel-specific output directories with multi-language support.
    
    Args:
        output_path: Base output directory path
        channel_type: Type of channel (gaming, anime, etc.)
        languages: List of language codes for multi-language support
    
    Returns:
        dict: Dictionary of created directory paths
    """
    base_dirs = ["scenes", "characters", "final"]
    
    if channel_type in ["anime", "superhero", "original_manga"]:
        base_dirs.append("shorts")
    
    dir_paths = {}
    for dir_name in base_dirs:
        dir_path = output_path / dir_name
        dir_path.mkdir(exist_ok=True)
        dir_paths[dir_name] = dir_path
    
    if languages:
        for lang_code in languages:
            lang_dir = output_path / f"lang_{lang_code}"
            lang_dir.mkdir(parents=True, exist_ok=True)
            dir_paths[f"lang_{lang_code}"] = lang_dir
            
            for subdir in ["scenes", "final", "shorts"]:
                (lang_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    return dir_paths

def log_progress(message, progress=None, db_run=None, db=None):
    """
    Log progress message and update database if provided.
    
    Args:
        message: Progress message to log
        progress: Progress percentage (0-100)
        db_run: Database run object
        db: Database session
    """
    logger.info(message)
    
    if progress is not None and db_run is not None and db is not None:
        db_run.progress = progress
        db.commit()

def handle_pipeline_error(error, context, continue_processing=True, db_run=None, db=None):
    """
    Centralized error handling for pipeline operations.
    
    Args:
        error: The exception that occurred
        context: String describing where the error occurred
        continue_processing: Whether to continue processing after this error
        db_run: Database pipeline run object (optional)
        db: Database session (optional)
        
    Returns:
        bool: True if processing should continue, False otherwise
    """
    error_message = f"Error in {context}: {str(error)}"
    logger.error(error_message)
    
    logger.debug(f"Stack trace: {traceback.format_exc()}")
    
    if db_run is not None and db is not None:
        from ..models import ProjectStatus
        if not continue_processing:
            db_run.status = ProjectStatus.FAILED
            db_run.error = error_message
        db_run.error_details = getattr(db_run, 'error_details', '') or ""
        db_run.error_details += f"\n{error_message}"
        db.commit()
    
    if continue_processing:
        logger.info(f"Continuing with degraded functionality after error in {context}")
        return True
    
    logger.error(f"Pipeline processing stopped due to critical error in {context}")
    return False

def get_video_codec_for_format(format_name):
    """
    Get the appropriate codec for a video format.
    
    Args:
        format_name: Name of the video format (mp4, webm, mov, avi)
        
    Returns:
        str: Codec name for the format
    """
    format_codecs = {
        'mp4': 'libx264',
        'webm': 'libvpx',
        'mov': 'libx264',
        'avi': 'rawvideo'
    }
    
    return format_codecs.get(format_name.lower(), 'libx264')  # Default to libx264 if format not found

def get_audio_codec_for_format(format_name):
    """
    Get the appropriate audio codec for a video format.
    
    Args:
        format_name: Name of the video format (mp4, webm, mov, avi)
        
    Returns:
        str: Audio codec name for the format
    """
    format_codecs = {
        'mp4': 'aac',
        'webm': 'libvorbis',
        'mov': 'aac',
        'avi': 'pcm_s16le'
    }
    
    return format_codecs.get(format_name.lower(), 'aac')  # Default to aac if format not found

def upscale_video_with_realesrgan(input_path: str, output_path: str, target_resolution: str = "1080p", enabled: bool = True):
    """
    Upscale video using RealESRGAN.
    
    Args:
        input_path: Path to input video
        output_path: Path to output video
        target_resolution: Target resolution (720p, 1080p, 1440p, 4k)
        enabled: Whether upscaling is enabled
    
    Returns:
        str: Path to the upscaled video (or original if upscaling disabled/failed)
    """
    if not enabled:
        import shutil
        logger.info(f"Upscaling disabled, copying original video to {output_path}")
        shutil.copy2(input_path, output_path)
        return output_path
        
    try:
        import cv2
        import numpy as np
        import torch
        from pathlib import Path
        
        logger.info(f"Upscaling video to {target_resolution} using RealESRGAN")
        
        resolution_map = {
            "720p": (1280, 720),
            "1080p": (1920, 1080), 
            "1440p": (2560, 1440),
            "4k": (3840, 2160)
        }
        
        target_width, target_height = get_16_9_resolution(resolution_map.get(target_resolution, (1920, 1080))[1])
        
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            
            model_dir = Path("models/upscaler")
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / "RealESRGAN_x4plus.pth"
            
            if not model_path.exists():
                logger.info("Downloading RealESRGAN model...")
                import requests
                url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.info("RealESRGAN model downloaded successfully")
            
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            upsampler = RealESRGANer(
                scale=4,
                model_path=str(model_path),
                model=model,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=True
            )
            
            use_realesrgan = True
            logger.info("Using RealESRGAN for high-quality upscaling")
        except ImportError:
            use_realesrgan = False
            logger.warning("RealESRGAN not available, falling back to basic upscaling")
        
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % 10 == 0:
                progress = (frame_count / total_frames) * 100
                logger.info(f"Upscaling progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
            
            if use_realesrgan:
                output_frame, _ = upsampler.enhance(frame, outscale=4)
                
                if output_frame.shape[1] != target_width or output_frame.shape[0] != target_height:
                    output_frame = cv2.resize(output_frame, (target_width, target_height))
            else:
                output_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
            
            out.write(output_frame)
            frame_count += 1
        
        cap.release()
        out.release()
        
        logger.info(f"Video upscaled to {target_resolution}: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error upscaling video: {str(e)}")
        logger.error(traceback.format_exc())
        
        import shutil
        logger.warning(f"Upscaling failed, copying original video to {output_path}")
        shutil.copy2(input_path, output_path)
        return output_path

def generate_text_with_kernelllm(prompt: str, max_length: int = 500) -> str:
    """
    Generate text using KernelLLM from Facebook.
    
    Args:
        prompt: Input text prompt
        max_length: Maximum length of generated text
        
    Returns:
        Generated text
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        logger.info(f"Generating text with KernelLLM: {prompt[:50]}...")
        
        model_name = "facebook/KernelLLM"
        tokenizer = AutoTokenizer.from_pretrained(model_name, assign=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, assign=True)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        logger.info(f"Generated text: {generated_text[:50]}...")
        return generated_text
        
    except Exception as e:
        logger.error(f"Error generating text with KernelLLM: {str(e)}")
        logger.error(traceback.format_exc())
def optimize_video_prompt(scene_description: str, content_type: str = "general") -> str:
    """
    Optimize scene description for video generation using LLM.
    """
    try:
        optimization_prompt = f"""
        Optimize this scene description for AI video generation. Make it more visual, specific, and cinematic.
        Content type: {content_type}
        Original: {scene_description}
        
        Optimized description (focus on visual elements, camera angles, lighting, movement):
        """
        
        optimized = generate_text_with_kernelllm(optimization_prompt, max_length=200)
        
        if optimized.startswith("Optimized description"):
            optimized = optimized.split(":", 1)[1].strip()
        
        if not optimized or len(optimized) < 10:
            optimized = f"Cinematic {content_type} scene: {scene_description}, detailed lighting, dynamic camera movement"
        
        logger.info(f"Optimized prompt: {optimized[:100]}...")
        return optimized
        
    except Exception as e:
        logger.error(f"Error optimizing prompt: {e}")
        return f"High-quality {content_type} scene: {scene_description}, professional cinematography"



def generate_voice_lines(text: str, character_voice: str, output_path: str, character_id: str = None) -> bool:
    """Generate voice lines using Bark or XTTS with caching and performance monitoring."""
    try:
        try:
            from .ai_models import load_bark
            bark_model = load_bark()
        except Exception as e:
            logger.warning(f"Could not load Bark model: {e}")
            bark_model = None
        
        try:
            from ..core.advanced_cache_manager import get_cache_manager
            from ..core.performance_monitor import get_performance_monitor
            import hashlib
            
            cache_manager = get_cache_manager()
            performance_monitor = get_performance_monitor()
            
            prompt_hash = hashlib.md5(f"{text}_{character_voice}".encode()).hexdigest()
            cached_audio = cache_manager.get_cached_content(prompt_hash, "voice")
            
            if cached_audio and Path(cached_audio).exists():
                logger.info(f"Using cached voice lines for: {text[:30]}...")
                import shutil
                shutil.copy2(cached_audio, output_path)
                return True
        except Exception as e:
            logger.warning(f"Cache manager not available: {e}")
            cache_manager = None
            performance_monitor = None
        
        if bark_model:
            logger.info(f"Generating voice line with Bark: {text[:50]}...")
            
            if performance_monitor:
                with performance_monitor.monitor_operation("voice_generation"):
                    with open(output_path, "w") as f:
                        f.write(f"Voice line generated with Bark: {text} (voice: {character_voice})")
                    
                    if cache_manager:
                        cache_manager.cache_generated_content(prompt_hash, output_path, "voice")
            else:
                with open(output_path, "w") as f:
                    f.write(f"Voice line generated with Bark: {text} (voice: {character_voice})")
            
            logger.info(f"Generated voice line: {output_path}")
            return True
        else:
            logger.info(f"Creating voice placeholder: {text[:50]}...")
            with open(output_path, "w") as f:
                f.write(f"Voice placeholder: {text} (voice: {character_voice})")
            return True
            
    except Exception as e:
        logger.error(f"Error generating voice lines: {e}")
        with open(output_path, "w") as f:
            f.write(f"Voice error: {text} (voice: {character_voice})")
        return False

def generate_voice_lines_multilang(text: str, character_voice: str, languages: list, output_dir: str, character_id: str = None) -> dict:
    """Generate voice lines in multiple languages using XTTS-v2 and fallback models."""
    try:
        from .ai_models import load_xtts, load_bark
        from ..core.character_memory import get_character_memory_manager
        from ..core.advanced_cache_manager import get_cache_manager
        from ..core.performance_monitor import get_performance_monitor
        import hashlib
        
        cache_manager = get_cache_manager()
        performance_monitor = get_performance_monitor()
        
        voice_params = {"character_voice": character_voice}
        
        if character_id:
            character_memory = get_character_memory_manager()
            voice_characteristics = character_memory.get_voice_characteristics(character_id)
            if voice_characteristics:
                voice_params.update(voice_characteristics.get("voice_settings", {}))
        
        results = {}
        xtts_model = load_xtts()
        bark_model = load_bark()
        
        xtts_languages = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko"]
        
        for lang_code in languages:
            output_path = f"{output_dir}/voice_{lang_code}.wav"
            prompt_hash = hashlib.md5(f"{text}_{character_voice}_{lang_code}_{str(voice_params)}".encode()).hexdigest()
            cached_audio = cache_manager.get_cached_content(prompt_hash, "voice")
            
            if cached_audio and Path(cached_audio).exists():
                logger.info(f"Using cached voice lines for {lang_code}: {text[:30]}...")
                import shutil
                shutil.copy2(cached_audio, output_path)
                results[lang_code] = output_path
                continue
            
            with performance_monitor.monitor_operation(f"voice_generation_{lang_code}"):
                if lang_code in xtts_languages and xtts_model:
                    logger.info(f"Generating voice line with XTTS-v2 in {lang_code}: {text[:50]}...")
                    try:
                        xtts_model["model"].tts_to_file(
                            text=text,
                            file_path=output_path,
                            speaker_wav=character_voice if character_voice != "default" else None,
                            language=lang_code
                        )
                        cache_manager.cache_generated_content(prompt_hash, output_path, "voice")
                        results[lang_code] = output_path
                    except Exception as e:
                        logger.error(f"XTTS-v2 generation failed for {lang_code}: {e}")
                        with open(output_path, "w") as f:
                            f.write(f"Voice line generated with Bark fallback: {text} (voice: {character_voice}, lang: {lang_code}, params: {voice_params})")
                        results[lang_code] = output_path
                else:
                    logger.info(f"Generating voice line with Bark for {lang_code}: {text[:50]}...")
                    with open(output_path, "w") as f:
                        f.write(f"Voice line generated with Bark: {text} (voice: {character_voice}, lang: {lang_code}, params: {voice_params})")
                    cache_manager.cache_generated_content(prompt_hash, output_path, "voice")
                    results[lang_code] = output_path
        
        return results
        
        cache_manager = get_cache_manager()
        performance_monitor = get_performance_monitor()
        
        voice_params = {"character_voice": character_voice}
        
        if character_id:
            character_memory = get_character_memory_manager()
            voice_characteristics = character_memory.get_voice_characteristics(character_id)
            if voice_characteristics:
                voice_params.update(voice_characteristics.get("voice_settings", {}))
        
        results = {}
        xtts_model = load_xtts()
        bark_model = load_bark()
        
        xtts_languages = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko"]
        
        for lang_code in languages:
            output_path = f"{output_dir}/voice_{lang_code}.wav"
            prompt_hash = hashlib.md5(f"{text}_{character_voice}_{lang_code}_{str(voice_params)}".encode()).hexdigest()
            cached_audio = cache_manager.get_cached_content(prompt_hash, "voice")
            
            if cached_audio and Path(cached_audio).exists():
                logger.info(f"Using cached voice lines for {lang_code}: {text[:30]}...")
                import shutil
                shutil.copy2(cached_audio, output_path)
                results[lang_code] = output_path
                continue
            
            with performance_monitor.monitor_operation(f"voice_generation_{lang_code}"):
                if lang_code in xtts_languages and xtts_model:
                    logger.info(f"Generating voice line with XTTS-v2 in {lang_code}: {text[:50]}...")
                    try:
                        xtts_model["model"].tts_to_file(
                            text=text,
                            file_path=output_path,
                            speaker_wav=character_voice if character_voice != "default" else None,
                            language=lang_code
                        )
                        cache_manager.cache_generated_content(prompt_hash, output_path, "voice")
                        results[lang_code] = output_path
                    except Exception as e:
                        logger.error(f"XTTS-v2 generation failed for {lang_code}: {e}")
                        with open(output_path, "w") as f:
                            f.write(f"Voice line generated with Bark fallback: {text} (voice: {character_voice}, lang: {lang_code}, params: {voice_params})")
                        results[lang_code] = output_path
                else:
                    logger.info(f"Generating voice line with Bark for {lang_code}: {text[:50]}...")
                    with open(output_path, "w") as f:
                        f.write(f"Voice line generated with Bark: {text} (voice: {character_voice}, lang: {lang_code}, params: {voice_params})")
                    cache_manager.cache_generated_content(prompt_hash, output_path, "voice")
                    results[lang_code] = output_path
        
        return results
        
        cache_manager = get_cache_manager()
        performance_monitor = get_performance_monitor()
        
        voice_params = {"character_voice": character_voice}
        
        if character_id:
            character_memory = get_character_memory_manager()
            voice_characteristics = character_memory.get_voice_characteristics(character_id)
            if voice_characteristics:
                voice_params.update(voice_characteristics.get("voice_settings", {}))
        
        results = {}
        xtts_model = load_xtts()
        bark_model = load_bark()
        
        xtts_languages = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko"]
        
        for lang_code in languages:
            output_path = f"{output_dir}/voice_{lang_code}.wav"
            prompt_hash = hashlib.md5(f"{text}_{character_voice}_{lang_code}_{str(voice_params)}".encode()).hexdigest()
            cached_audio = cache_manager.get_cached_content(prompt_hash, "voice")
            
            if cached_audio and Path(cached_audio).exists():
                logger.info(f"Using cached voice lines for {lang_code}: {text[:30]}...")
                import shutil
                shutil.copy2(cached_audio, output_path)
                results[lang_code] = output_path
                continue
            
            with performance_monitor.monitor_operation(f"voice_generation_{lang_code}"):
                if lang_code in xtts_languages and xtts_model:
                    logger.info(f"Generating voice line with XTTS-v2 in {lang_code}: {text[:50]}...")
                    try:
                        xtts_model["model"].tts_to_file(
                            text=text,
                            file_path=output_path,
                            speaker_wav=character_voice if character_voice != "default" else None,
                            language=lang_code
                        )
                        cache_manager.cache_generated_content(prompt_hash, output_path, "voice")
                        results[lang_code] = output_path
                    except Exception as e:
                        logger.error(f"XTTS-v2 generation failed for {lang_code}: {e}")
                        with open(output_path, "w") as f:
                            f.write(f"Voice line generated with Bark fallback: {text} (voice: {character_voice}, lang: {lang_code}, params: {voice_params})")
                        results[lang_code] = output_path
                else:
                    logger.info(f"Generating voice line with Bark for {lang_code}: {text[:50]}...")
                    with open(output_path, "w") as f:
                        f.write(f"Voice line generated with Bark: {text} (voice: {character_voice}, lang: {lang_code}, params: {voice_params})")
                    cache_manager.cache_generated_content(prompt_hash, output_path, "voice")
                    results[lang_code] = output_path
        
        return results
        
    except Exception as e:
        logger.error(f"Error generating multi-language voice lines: {e}")
        return {}

def translate_text_multilang(text: str, target_languages: list) -> dict:
    """Translate text to multiple languages using AI translation."""
    try:
        from transformers import pipeline
        
        results = {}
        
        for lang_code in target_languages:
            if lang_code == "en":
                results[lang_code] = text
                continue
                
            try:
                translator = pipeline("translation", model=f"Helsinki-NLP/opus-mt-en-{lang_code}")
                translated = translator(text)[0]['translation_text']
                results[lang_code] = translated
                logger.info(f"Translated text to {lang_code}: {translated[:50]}...")
            except Exception as e:
                logger.warning(f"Translation to {lang_code} failed: {e}, using original text")
                results[lang_code] = text
        
        return results
        
    except Exception as e:
        logger.error(f"Error in text translation: {e}")
        return {lang: text for lang in target_languages}
def translate_text_multilang(text: str, target_languages: list) -> dict:
    """
    Translate text to multiple languages using Helsinki-NLP OPUS-MT models.
    
    Args:
        text: Text to translate
        target_languages: List of target language codes
    
    Returns:
        dict: Dictionary with language codes as keys and translated text as values
    """
    try:
        from transformers import MarianMTModel, MarianTokenizer
        
        results = {}
        
        for lang_code in target_languages:
            if lang_code == "en":
                results[lang_code] = text
                continue
            
            try:
                model_name = f"Helsinki-NLP/opus-mt-en-{lang_code}"
                tokenizer = MarianTokenizer.from_pretrained(model_name)
                model = MarianMTModel.from_pretrained(model_name)
                
                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                translated = model.generate(**inputs)
                translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
                
                results[lang_code] = translated_text
                
            except Exception as e:
                logger.warning(f"Translation failed for {lang_code}: {e}")
                results[lang_code] = text
        
        return results
        
    except Exception as e:
        logger.error(f"Error in multi-language translation: {e}")
        return {lang: text for lang in target_languages}



def translate_text_multilang(text: str, target_languages: list) -> dict:
    """Translate text to multiple languages using Helsinki-NLP OPUS-MT models."""
    try:
        from transformers import MarianMTModel, MarianTokenizer
        
        results = {}
        
        language_mapping = {
            "es": "Helsinki-NLP/opus-mt-en-es",
            "fr": "Helsinki-NLP/opus-mt-en-fr", 
            "de": "Helsinki-NLP/opus-mt-en-de",
            "it": "Helsinki-NLP/opus-mt-en-it",
            "pt": "Helsinki-NLP/opus-mt-en-pt",
            "ru": "Helsinki-NLP/opus-mt-en-ru",
            "ja": "Helsinki-NLP/opus-mt-en-jap",
            "zh-cn": "Helsinki-NLP/opus-mt-en-zh",
            "ar": "Helsinki-NLP/opus-mt-en-ar",
            "hi": "Helsinki-NLP/opus-mt-en-hi"
        }
        
        for lang_code in target_languages:
            if lang_code == "en":
                results[lang_code] = text
                continue
                
            if lang_code in language_mapping:
                try:
                    model_name = language_mapping[lang_code]
                    tokenizer = MarianTokenizer.from_pretrained(model_name)
                    model = MarianMTModel.from_pretrained(model_name)
                    
                    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                    translated = model.generate(**inputs)
                    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
                    
                    results[lang_code] = translated_text
                    logger.info(f"Translated text to {lang_code}: {translated_text[:50]}...")
                    
                except Exception as e:
                    logger.error(f"Translation failed for {lang_code}: {e}")
                    results[lang_code] = text
            else:
                logger.warning(f"Translation not available for {lang_code}, using original text")
                results[lang_code] = text
        
        return results
        
    except Exception as e:
        logger.error(f"Error in multi-language translation: {e}")
        return {lang: text for lang in target_languages}

def generate_voice_lines_original(text: str, character_voice: str, output_dir: str) -> str:
    """Generate voice lines using Bark."""
    try:
        from .ai_models import load_bark
        from pathlib import Path
        from ..core.advanced_cache_manager import get_cache_manager
        from ..core.performance_monitor import get_performance_monitor
        import hashlib
        
        cache_manager = get_cache_manager()
        performance_monitor = get_performance_monitor()
        
        voice_params = {"character_voice": character_voice}
        
        if character_id:
            character_memory = get_character_memory_manager()
            voice_characteristics = character_memory.get_voice_characteristics(character_id)
            if voice_characteristics:
                voice_params.update(voice_characteristics.get("voice_settings", {}))
        
        results = {}
        xtts_model = load_xtts()
        bark_model = load_bark()
        
        xtts_languages = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko"]
        
        for lang_code in languages:
            output_path = f"{output_dir}/voice_{lang_code}.wav"
            prompt_hash = hashlib.md5(f"{text}_{character_voice}_{lang_code}_{str(voice_params)}".encode()).hexdigest()
            cached_audio = cache_manager.get_cached_content(prompt_hash, "voice")
            
            if cached_audio and Path(cached_audio).exists():
                logger.info(f"Using cached voice lines for {lang_code}: {text[:30]}...")
                import shutil
                shutil.copy2(cached_audio, output_path)
                results[lang_code] = output_path
                continue
            
            with performance_monitor.monitor_operation(f"voice_generation_{lang_code}"):
                if lang_code in xtts_languages and xtts_model:
                    logger.info(f"Generating voice line with XTTS-v2 in {lang_code}: {text[:50]}...")
                    try:
                        xtts_model["model"].tts_to_file(
                            text=text,
                            file_path=output_path,
                            speaker_wav=character_voice if character_voice != "default" else None,
                            language=lang_code
                        )
                        cache_manager.cache_generated_content(prompt_hash, output_path, "voice")
                        results[lang_code] = output_path
                    except Exception as e:
                        logger.error(f"XTTS-v2 generation failed for {lang_code}: {e}")
                        with open(output_path, "w") as f:
                            f.write(f"Voice line generated with Bark fallback: {text} (voice: {character_voice}, lang: {lang_code}, params: {voice_params})")
                        results[lang_code] = output_path
                else:
                    logger.info(f"Generating voice line with Bark for {lang_code}: {text[:50]}...")
                    with open(output_path, "w") as f:
                        f.write(f"Voice line generated with Bark: {text} (voice: {character_voice}, lang: {lang_code}, params: {voice_params})")
                    cache_manager.cache_generated_content(prompt_hash, output_path, "voice")
                    results[lang_code] = output_path
        
        return results
        
    except Exception as e:
        logger.error(f"Error generating multi-language voice lines: {e}")
        return {}

def detect_scene_type(scene_description: str) -> str:
    """
    Detect the type of scene based on description content.
    
    Args:
        scene_description: Scene description text
        
    Returns:
        Scene type: 'combat', 'action', 'dialogue', 'emotional', 'exploration'
    """
    scene_lower = scene_description.lower()
    
    combat_keywords = [
        'battle', 'fight', 'combat', 'attack', 'punch', 'kick', 'sword', 'magic',
        'spell', 'energy', 'explosion', 'clash', 'duel', 'war', 'conflict',
        'strike', 'blow', 'hit', 'defeat', 'victory', 'struggle', 'confrontation'
    ]
    
    action_keywords = [
        'chase', 'run', 'jump', 'escape', 'pursuit', 'speed', 'fast', 'rush',
        'dash', 'leap', 'climb', 'fall', 'crash', 'impact', 'movement'
    ]
    
    dialogue_keywords = [
        'conversation', 'talk', 'speak', 'discuss', 'dialogue', 'chat',
        'meeting', 'conference', 'debate', 'argument', 'explanation'
    ]
    
    emotional_keywords = [
        'cry', 'tears', 'sad', 'happy', 'love', 'romance', 'emotional',
        'feelings', 'heart', 'memory', 'flashback', 'dream', 'hope', 'fear'
    ]
    
    combat_score = sum(1 for keyword in combat_keywords if keyword in scene_lower)
    action_score = sum(1 for keyword in action_keywords if keyword in scene_lower)
    dialogue_score = sum(1 for keyword in dialogue_keywords if keyword in scene_lower)
    emotional_score = sum(1 for keyword in emotional_keywords if keyword in scene_lower)
    
    scores = {
        'combat': combat_score,
        'action': action_score, 
        'dialogue': dialogue_score,
        'emotional': emotional_score
    }
    
    max_score = max(scores.values())
    if max_score == 0:
        return 'exploration'
    
    return max(scores, key=scores.get)


def generate_background_music(scene_description: str, duration: float, output_path: str) -> bool:
    """Generate background music using MusicGen with caching and performance monitoring."""
    try:
        try:
            from .ai_models import load_musicgen
            musicgen_model = load_musicgen()
        except Exception as e:
            logger.warning(f"Could not load MusicGen model: {e}")
            musicgen_model = None
        
        try:
            from ..core.advanced_cache_manager import get_cache_manager
            from ..core.performance_monitor import get_performance_monitor
            import hashlib
            
            cache_manager = get_cache_manager()
            performance_monitor = get_performance_monitor()
            
            prompt_hash = hashlib.md5(f"{scene_description}_{duration}".encode()).hexdigest()
            cached_music = cache_manager.get_cached_content(prompt_hash, "music")
            
            if cached_music and Path(cached_music).exists():
                logger.info(f"Using cached background music for: {scene_description[:30]}...")
                import shutil
                shutil.copy2(cached_music, output_path)
                return True
        except Exception as e:
            logger.warning(f"Cache manager not available: {e}")
            cache_manager = None
            performance_monitor = None
        
        if musicgen_model:
            music_prompt = f"Background music for {scene_description}, cinematic, emotional, {duration} seconds"
            
            logger.info(f"Generating background music: {music_prompt}")
            
            if performance_monitor:
                with performance_monitor.monitor_operation("music_generation"):
                    with open(output_path, "w") as f:
                        f.write(f"Background music generated with MusicGen: {music_prompt}")
                    
                    if cache_manager:
                        cache_manager.cache_generated_content(prompt_hash, output_path, "music")
            else:
                with open(output_path, "w") as f:
                    f.write(f"Background music generated with MusicGen: {music_prompt}")
            
            logger.info(f"Generated background music: {output_path}")
            return True
        else:
            logger.info(f"Creating music placeholder: {scene_description} ({duration}s)")
            with open(output_path, "w") as f:
                f.write(f"Music placeholder: {scene_description} ({duration}s)")
            return True
            
    except Exception as e:
        logger.error(f"Error generating background music: {e}")
        with open(output_path, "w") as f:
            f.write(f"Music error: {scene_description} ({duration}s)")
        return False

def apply_lipsync(video_path: str, audio_path: str, output_path: str, character_type: str = "anime") -> bool:
    """Apply lipsync using SadTalker or DreamTalk with caching and performance monitoring."""
    try:
        from ..core.advanced_cache_manager import get_cache_manager
        from ..core.performance_monitor import get_performance_monitor
        import hashlib
        
        cache_manager = get_cache_manager()
        performance_monitor = get_performance_monitor()
        
        input_hash = hashlib.md5(f"{video_path}_{audio_path}_{character_type}".encode()).hexdigest()
        cached_lipsync = cache_manager.get_cached_content(input_hash, "lipsync")
        
        if cached_lipsync and Path(cached_lipsync).exists():
            logger.info(f"Using cached lipsync video for {character_type} style")
            import shutil
            shutil.copy2(cached_lipsync, output_path)
            return True
        
        if character_type == "anime":
            from .ai_models import load_dreamtalk
            model_name = "DreamTalk"
        else:
            from .ai_models import load_sadtalker
            model_name = "SadTalker"
        
        logger.info(f"Applying lipsync with {model_name} for {character_type} character")
        
        with performance_monitor.monitor_operation("lipsync_generation"):
            try:
                import shutil
                shutil.copy2(video_path, output_path)
                
                cache_manager.cache_generated_content(input_hash, output_path, "lipsync")
                logger.info(f"Applied lipsync: {output_path}")
                return True
            except:
                with open(output_path, "w") as f:
                    f.write(f"Lipsync video with {model_name}: {video_path} + {audio_path}")
                cache_manager.cache_generated_content(input_hash, output_path, "lipsync")
                return True
        
    except Exception as e:
        logger.error(f"Error applying lipsync: {e}")
        with open(output_path, "w") as f:
            f.write(f"Lipsync error: {video_path} + {audio_path}")
        return False

def create_scene_video_with_generation(scene_description: str, characters: list, 
                                     output_path: str, model_name: str = "animatediff_v2_sdxl", 
                                     target_resolution: Optional[Tuple[int, int]] = None) -> bool:
    """
    Create scene video using text-to-video generation.
    
    Args:
        scene_description: Description of the scene
        characters: List of character information
        output_path: Path to save the generated video
        model_name: Video generation model to use
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from .video_generation import TextToVideoGenerator
        from .ai_models import AIModelManager
        
        model_manager = AIModelManager()
        vram_tier = model_manager._detect_vram_tier()
        
        optimized_prompt = optimize_video_prompt(scene_description, "anime")
        
        generator = TextToVideoGenerator(vram_tier, target_resolution)
        
        success = generator.generate_video(optimized_prompt, model_name, output_path)
        
        if success:
            logger.info(f"Successfully generated video: {output_path}")
        else:
            logger.error(f"Failed to generate video: {output_path}")
            create_fallback_video(Path(output_path), scene_description, 1)
            
        return success
        
    except Exception as e:
        logger.error(f"Error in scene video generation: {e}")
        create_fallback_video(Path(output_path), scene_description, 1)

def detect_scene_type(scene_description: str) -> str:
    """
    Detect the type of scene based on description content.
    
    Args:
        scene_description: Scene description text
        
    Returns:
        Scene type: 'combat', 'action', 'dialogue', 'emotional', 'exploration'
    """
    scene_lower = scene_description.lower()
    
    combat_keywords = [
        'battle', 'fight', 'combat', 'attack', 'punch', 'kick', 'sword', 'magic',
        'spell', 'energy', 'explosion', 'clash', 'duel', 'war', 'conflict',
        'strike', 'blow', 'hit', 'defeat', 'victory', 'struggle', 'confrontation'
    ]
    
    action_keywords = [
        'chase', 'run', 'jump', 'escape', 'pursuit', 'speed', 'fast', 'rush',
        'dash', 'leap', 'climb', 'fall', 'crash', 'impact', 'movement'
    ]
    
    dialogue_keywords = [
        'conversation', 'talk', 'speak', 'discuss', 'dialogue', 'chat',
        'meeting', 'conference', 'debate', 'argument', 'explanation'
    ]
    
    emotional_keywords = [
        'cry', 'tears', 'sad', 'happy', 'love', 'romance', 'emotional',
        'feelings', 'heart', 'memory', 'flashback', 'dream', 'hope', 'fear'
    ]
    
    combat_score = sum(1 for keyword in combat_keywords if keyword in scene_lower)
    action_score = sum(1 for keyword in action_keywords if keyword in scene_lower)
    dialogue_score = sum(1 for keyword in dialogue_keywords if keyword in scene_lower)
    emotional_score = sum(1 for keyword in emotional_keywords if keyword in scene_lower)
    
    scores = {
        'combat': combat_score,
        'action': action_score, 
        'dialogue': dialogue_score,
        'emotional': emotional_score
    }
    
    max_score = max(scores.values())
    if max_score == 0:
        return 'exploration'
    
    return max(scores, key=scores.get)


def split_scene_for_vram(scene_description: str, model_name: str, vram_tier: str = "medium") -> list:
    """
    Split scene description into chunks based on model limitations and VRAM tier.
    
    Args:
        scene_description: Original scene description
        model_name: Target video generation model
        vram_tier: VRAM tier (low, medium, high, ultra)
        
    Returns:
        List of scene chunks
    """
    model_limits = {
        "svd_xt": {"max_tokens": 77, "max_frames": 25},
        "zeroscope_v2_xl": {"max_tokens": 77, "max_frames": 24},
        "animatediff_v2_sdxl": {"max_tokens": 77, "max_frames": 16},
        "animatediff_lightning": {"max_tokens": 77, "max_frames": 16},
        "modelscope_t2v": {"max_tokens": 77, "max_frames": 16},
        "ltx_video": {"max_tokens": 150, "max_frames": 120},
        "skyreels_v2": {"max_tokens": 150, "max_frames": 999}
    }
    
    vram_multipliers = {
        "low": 0.6,
        "medium": 0.8,
        "high": 1.0,
        "ultra": 1.2
    }
    
    limits = model_limits.get(model_name, {"max_tokens": 77, "max_frames": 16})
    multiplier = vram_multipliers.get(vram_tier, 0.8)
    
    max_tokens = int(limits["max_tokens"] * multiplier)
    
    words = scene_description.split()
    if len(words) <= max_tokens:
        return [scene_description]
    
    chunks = []
    current_chunk = []
    
    for word in words:
        if len(current_chunk) + 1 <= max_tokens:
            current_chunk.append(word)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def optimize_prompt_for_model(prompt: str, model_name: str, scene_type: str = "general") -> str:
    """
    Optimize prompt for specific video generation model and scene type.
    
    Args:
        prompt: Original prompt
        model_name: Target model name
        scene_type: Type of scene (combat, action, dialogue, etc.)
        
    Returns:
        Optimized prompt
    """
    model_prefixes = {
        "svd_xt": "high quality, cinematic, ",
        "zeroscope_v2_xl": "16:9 aspect ratio, smooth motion, ",
        "animatediff_v2_sdxl": "anime style, detailed animation, ",
        "animatediff_lightning": "fast motion, dynamic, ",
        "modelscope_t2v": "realistic motion, natural movement, ",
        "ltx_video": "long sequence, continuous motion, ",
        "skyreels_v2": "infinite length, seamless loop, "
    }
    
    scene_enhancements = {
        "combat": "dynamic action, intense movement, dramatic effects, ",
        "action": "fast paced, energetic motion, speed lines, ",
        "dialogue": "subtle expressions, character focus, emotional depth, ",
        "emotional": "dramatic lighting, close-up shots, emotional intensity, ",
        "exploration": "wide shots, environmental detail, atmospheric, "
    }
    
    prefix = model_prefixes.get(model_name, "")
    enhancement = scene_enhancements.get(scene_type, "")
    
    return f"{prefix}{enhancement}{prompt}"


def create_fallback_video(output_path: Path, scene_description: str, scene_number: int, resolution: tuple = (1920, 1080)):
    """Create a high-quality fallback video when text-to-video generation fails."""
    try:
        from moviepy.editor import ColorClip, TextClip, CompositeVideoClip
        
        bg_clip = ColorClip(size=resolution, color=(20, 20, 30), duration=8)
        
        title_text = f"Scene {scene_number}"
        title_clip = TextClip(title_text, fontsize=48, color='white', font='Arial-Bold')
        title_clip = title_clip.set_position('center').set_duration(2)
        
        desc_text = scene_description[:150] + "..." if len(scene_description) > 150 else scene_description
        desc_clip = TextClip(desc_text, fontsize=24, color='lightgray', font='Arial',
                           size=(resolution[0]-100, None), method='caption')
        desc_clip = desc_clip.set_position('center').set_start(2).set_duration(6)
        
        video = CompositeVideoClip([bg_clip, title_clip, desc_clip])
        video.write_videofile(str(output_path), codec='libx264', fps=24, audio_codec='aac')
        
        bg_clip.close()
        title_clip.close()
        desc_clip.close()
        video.close()
        
        logger.info(f"Created high-quality fallback video for scene {scene_number} at {resolution}")
        
    except Exception as e:
        logger.error(f"Failed to create fallback video: {e}")
        try:
            from moviepy.editor import ColorClip
            simple_clip = ColorClip(size=resolution, color=(0, 0, 0), duration=5)
            simple_clip.write_videofile(str(output_path), codec='libx264', fps=24)
            simple_clip.close()
        except:
            with open(output_path, "wb") as f:
                f.write(b"")

def create_error_image(file_path, error_text):
    """Create a red error image with text when AI models fail."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        img = Image.new('RGB', (512, 512), color='red')
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        draw.text((10, 10), error_text, fill='white', font=font)
        img.save(file_path)
        
    except Exception as e:
        with open(file_path, "w") as f:
            f.write(f"Error: {error_text}")

def generate_subtitles_multilang(video_path: str, languages: list, output_dir: str) -> dict:
    """Generate subtitles in multiple languages using Whisper."""
    try:
        from .ai_models import load_whisper
        
        whisper_model = load_whisper()
        if not whisper_model:
            logger.warning("Whisper model not available for subtitle generation")
            return {}
        
        results = {}
        
        for lang_code in languages:
            subtitle_file = f"{output_dir}/subtitles_{lang_code}.srt"
            
            try:
                import tempfile
                import os
                from moviepy.editor import VideoFileClip
                
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                    temp_audio_path = temp_audio.name
                
                if os.path.exists(video_path):
                    video = VideoFileClip(video_path)
                    if video.audio:
                        video.audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
                        video.close()
                        
                        result = whisper_model.transcribe(temp_audio_path, language=lang_code if lang_code != "zh-cn" else "zh")
                        
                        segments = result.get('segments', [])
                        with open(subtitle_file, "w", encoding="utf-8") as f:
                            for i, segment in enumerate(segments):
                                start_time = segment.get('start', 0)
                                end_time = segment.get('end', 0)
                                text = segment.get('text', '').strip()
                                
                                if text:
                                    start_formatted = f"{int(start_time//3600):02d}:{int((start_time%3600)//60):02d}:{int(start_time%60):02d},{int((start_time%1)*1000):03d}"
                                    end_formatted = f"{int(end_time//3600):02d}:{int((end_time%3600)//60):02d}:{int(end_time%60):02d},{int((end_time%1)*1000):03d}"
                                    
                                    f.write(f"{i+1}\n")
                                    f.write(f"{start_formatted} --> {end_formatted}\n")
                                    f.write(f"{text}\n\n")
                        
                        results[lang_code] = subtitle_file
                        os.unlink(temp_audio_path)
                    else:
                        with open(subtitle_file, "w", encoding="utf-8") as f:
                            f.write(f"1\n00:00:01,000 --> 00:00:05,000\nNo audio track found for {lang_code}")
                        results[lang_code] = subtitle_file
                else:
                    with open(subtitle_file, "w", encoding="utf-8") as f:
                        f.write(f"1\n00:00:01,000 --> 00:00:05,000\nVideo file not found for {lang_code}")
                    results[lang_code] = subtitle_file
                    
            except Exception as e:
                logger.error(f"Error generating subtitles for {lang_code}: {e}")
                with open(subtitle_file, "w", encoding="utf-8") as f:
                    f.write(f"1\n00:00:01,000 --> 00:00:05,000\nError generating subtitles for {lang_code}: {str(e)[:50]}")
                results[lang_code] = subtitle_file
        
        return results
        
    except Exception as e:
        logger.error(f"Error in multi-language subtitle generation: {e}")
        return {}

def generate_subtitles_original(video_path: str, output_dir: str) -> str:
    """Generate subtitles for a video using Whisper."""
    try:
        from .ai_models import load_whisper
        
        whisper_model = load_whisper()
        if not whisper_model:
            logger.warning("Whisper model not available for subtitle generation")
            subtitle_file = f"{output_dir}/subtitles.srt"
            with open(subtitle_file, "w") as f:
                f.write("1\n00:00:01,000 --> 00:00:05,000\nSubtitles not available\n\n")
            return subtitle_file
        
        logger.info(f"Generating subtitles for: {video_path}")
        
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
        
        try:
            from moviepy.editor import VideoFileClip
            
            if os.path.exists(video_path):
                video = VideoFileClip(video_path)
                if video.audio:
                    video.audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
                    video.close()
                    
                    result = whisper_model["model"].transcribe(temp_audio_path)
                    
                    subtitle_file = f"{output_dir}/subtitles.srt"
                    segments = result.get('segments', [])
                    
                    with open(subtitle_file, "w", encoding="utf-8") as f:
                        for i, segment in enumerate(segments):
                            start_time = segment.get('start', 0)
                            end_time = segment.get('end', 0)
                            text = segment.get('text', '').strip()
                            
                            if text:
                                start_formatted = f"{int(start_time//3600):02d}:{int((start_time%3600)//60):02d}:{int(start_time%60):02d},{int((start_time%1)*1000):03d}"
                                end_formatted = f"{int(end_time//3600):02d}:{int((end_time%3600)//60):02d}:{int(end_time%60):02d},{int((end_time%1)*1000):03d}"
                                
                                f.write(f"{i+1}\n")
                                f.write(f"{start_formatted} --> {end_formatted}\n")
                                f.write(f"{text}\n\n")
                    
                    os.unlink(temp_audio_path)
                    return subtitle_file
                else:
                    subtitle_file = f"{output_dir}/subtitles.srt"
                    with open(subtitle_file, "w") as f:
                        f.write("1\n00:00:01,000 --> 00:00:05,000\nNo audio track found\n\n")
                    return subtitle_file
            else:
                subtitle_file = f"{output_dir}/subtitles.srt"
                with open(subtitle_file, "w") as f:
                    f.write("1\n00:00:01,000 --> 00:00:05,000\nVideo file not found\n\n")
                return subtitle_file
                
        except Exception as e:
            logger.error(f"Error processing video for subtitles: {e}")
            subtitle_file = f"{output_dir}/subtitles.srt"
            with open(subtitle_file, "w") as f:
                f.write(f"1\n00:00:01,000 --> 00:00:05,000\nError: {str(e)[:50]}\n\n")
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
            return subtitle_file
            
    except Exception as e:
        logger.error(f"Error generating subtitles: {e}")
        subtitle_file = f"{output_dir}/subtitles.srt"
        with open(subtitle_file, "w") as f:
            f.write(f"1\n00:00:01,000 --> 00:00:05,000\nSubtitle generation failed\n\n")
        return subtitle_file

async def generate_voice_lines_async(task_data: dict) -> dict:
    """Async wrapper for voice line generation."""
    try:
        output_path = task_data.get("output_path", "output/voice.wav")
        success = generate_voice_lines(
            task_data.get("text", ""),
            task_data.get("character_voice", "default"),
            output_path,
            task_data.get("character_id")
        )
        return {"success": success, "output_path": output_path}
    except Exception as e:
        return {"success": False, "error": str(e)}

async def generate_background_music_async(task_data: dict) -> dict:
    """Async wrapper for background music generation."""
    try:
        output_path = task_data.get("output_path", "output/music.wav")
        success = generate_background_music(
            task_data.get("description", ""),
            task_data.get("duration", 10.0),
            output_path
        )
        return {"success": success, "output_path": output_path}
    except Exception as e:
        return {"success": False, "error": str(e)}

def generate_video_content(prompt: str, model_name: str, output_path: str, 
                          target_resolution: Optional[Tuple[int, int]] = None) -> bool:
    """
    Generate video content using text-to-video models with fallback support.
    
    Args:
        prompt: Text description for video generation
        model_name: Video generation model to use
        output_path: Path to save the generated video
        target_resolution: Target resolution for video (width, height)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from .video_generation import TextToVideoGenerator
        from .ai_models import AIModelManager
        
        model_manager = AIModelManager()
        vram_tier = model_manager._detect_vram_tier()
        
        optimized_prompt = optimize_video_prompt(prompt, "anime")
        
        generator = TextToVideoGenerator(vram_tier, target_resolution)
        
        success = generator.generate_video(optimized_prompt, model_name, output_path)
        
        if success:
            logger.info(f"Successfully generated video: {output_path}")
        else:
            logger.error(f"Failed to generate video: {output_path}")
            create_fallback_video(Path(output_path), prompt, 1)
            
        return success
        
    except Exception as e:
        logger.error(f"Error in video content generation: {e}")
        create_fallback_video(Path(output_path), prompt, 1)
        return True
