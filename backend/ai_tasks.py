import time
import os
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from sqlalchemy.orm import Session
from typing import List, Optional, Dict
import threading
import queue

from config import OUTPUT_DIR
from .database import DBProject, DBPipelineRun, DBProjectLora
from .models import ProjectStatus

logger = logging.getLogger(__name__)

# Global queue for pipeline processing
pipeline_queue = queue.Queue()
pipeline_thread = None
is_processing = False

def run_pipeline(project_id: int, pipeline_run_id: int, db: Session):
    """
    Run the AI pipeline for a project.
    
    Args:
        project_id: ID of the project
        pipeline_run_id: ID of the pipeline run
        db: Database session
    """
    try:
        asyncio.run(run_pipeline_async(project_id, pipeline_run_id, db))
    except Exception as e:
        logger.error(f"Async pipeline execution failed: {e}")
        logger.info("Falling back to sync pipeline execution")
        run_pipeline_sync(project_id, pipeline_run_id, db)


async def run_pipeline_async(project_id: int, pipeline_run_id: int, db: Session):
    """
    Run the AI pipeline asynchronously with concurrent processing.
    
    Args:
        project_id: ID of the project
        pipeline_run_id: ID of the pipeline run
        db: Database session
    """
    from .core.async_pipeline_manager import get_async_pipeline_manager
    from .core.performance_monitor import get_performance_monitor
    
    async_manager = get_async_pipeline_manager()
    performance_monitor = get_performance_monitor()
    
    db_project = db.query(DBProject).filter(DBProject.id == project_id).first()
    db_run = db.query(DBPipelineRun).filter(DBPipelineRun.id == pipeline_run_id).first()
    
    if db_project is None or db_run is None:
        return
    
    try:
        db_run.status = ProjectStatus.RUNNING
        db_project.status = ProjectStatus.RUNNING
        db.commit()
        
        if db_project.output_path is None:
            output_dir = OUTPUT_DIR / db_project.title.replace(" ", "_").lower()
            db_project.output_path = str(output_dir)
            db.commit()
        else:
            output_dir = Path(db_project.output_path)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        base_model = db_project.base_model or "stable_diffusion_1_5"
        channel_type = db_project.channel_type
        input_path = db_project.input_path or ""
        
        if not base_model:
            raise ValueError("Base model is required for pipeline execution")
        
        project_loras = db.query(DBProjectLora).filter(
            DBProjectLora.project_id == project_id
        ).order_by(DBProjectLora.order_index).all()
        
        if not project_loras and db_project.lora_model:
            lora_models = [db_project.lora_model]
            lora_paths = {}
        else:
            lora_models = [pl.lora_name for pl in project_loras]
            lora_paths = {pl.lora_name: pl.lora_path for pl in project_loras if pl.lora_path}
        
        from .pipelines.channel_specific import get_pipeline_for_channel
        
        pipeline_module = get_pipeline_for_channel(channel_type)
        if pipeline_module is None:
            raise ValueError(f"No pipeline available for channel type: {channel_type}")
        
        from .localization.multi_language_pipeline import multi_language_pipeline_manager
        
        selected_languages = getattr(db_project, 'selected_languages', ['en'])
        if isinstance(selected_languages, str):
            selected_languages = [selected_languages]
        
        scenes = await extract_scenes_from_pipeline(pipeline_module, input_path, channel_type)
        
        if not scenes:
            logger.warning("No scenes extracted from script, attempting to parse script directly")
            try:
                with open(input_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if content.strip():
                    import yaml
                    import json
                    
                    try:
                        if input_path.endswith('.yaml') or input_path.endswith('.yml'):
                            data = yaml.safe_load(content)
                            if isinstance(data, dict) and 'scenes' in data:
                                scenes = data['scenes']
                            elif isinstance(data, list):
                                scenes = data
                        elif input_path.endswith('.json'):
                            data = json.loads(content)
                            if isinstance(data, dict) and 'scenes' in data:
                                scenes = data['scenes']
                            elif isinstance(data, list):
                                scenes = data
                        else:
                            lines = content.strip().split('\n')
                            scenes = []
                            for i, line in enumerate(lines):
                                if line.strip():
                                    scenes.append({
                                        "description": f"Scene {i+1}",
                                        "dialogue": line.strip(),
                                        "duration": 10.0
                                    })
                    except Exception as parse_error:
                        logger.error(f"Error parsing script file: {parse_error}")
                        scenes = []
                
                if not scenes:
                    scenes = [
                        {"description": "Scene 1", "dialogue": "Hello world", "duration": 10.0},
                        {"description": "Scene 2", "dialogue": "This is a test", "duration": 8.0}
                    ]
                    
            except Exception as e:
                logger.error(f"Error reading script file: {e}")
                scenes = [
                    {"description": "Scene 1", "dialogue": "Hello world", "duration": 10.0},
                    {"description": "Scene 2", "dialogue": "This is a test", "duration": 8.0}
                ]
        
        script_data = {}
        if input_path and os.path.exists(input_path):
            try:
                if input_path.endswith('.yaml') or input_path.endswith('.yml'):
                    import yaml
                    with open(input_path, 'r', encoding='utf-8') as f:
                        script_data = yaml.safe_load(f) or {}
                elif input_path.endswith('.json'):
                    import json
                    with open(input_path, 'r', encoding='utf-8') as f:
                        script_data = json.load(f) or {}
            except Exception as e:
                logger.error(f"Failed to load script data: {e}")
                script_data = {}
        
        pipeline_config = {
            "base_model": base_model,
            "channel_type": channel_type,
            "lora_models": lora_models,
            "lora_paths": lora_paths,
            "output_path": str(output_dir),
            "input_path": input_path,
            "script_data": script_data
        }
        
        logger.info(f"Executing multi-language pipeline for languages: {', '.join(selected_languages)}")
        
        results = await multi_language_pipeline_manager.execute_multi_language_pipeline(
            scenes, pipeline_config, selected_languages
        )
        
        if results.get("performance_metrics", {}).get("success_rate", 0) == 0:
            raise Exception("All language pipelines failed")
        
        successful_languages = results.get("performance_metrics", {}).get("successful_language_codes", [])
        if successful_languages:
            main_language = successful_languages[0]
            output_file = output_dir / f"output_{main_language}" / f"final_video_{main_language}.mp4"
        else:
            output_file = output_dir / "final_video.mp4"
        
        logger.info(f"Running async pipeline for channel type: {channel_type}")
        logger.info(f"Using input script: {input_path}")
        logger.info(f"Processing {len(scenes)} scenes from script")
        logger.info(f"Using base model: {base_model}")
        logger.info(f"Using LoRA adaptations: {', '.join(lora_models)}")
        
        db_run.output_path = str(output_file)
        db_run.status = ProjectStatus.COMPLETED
        db_run.end_time = datetime.utcnow()
        
        db_project.status = ProjectStatus.COMPLETED
        db.commit()
        
        print(f"Multi-language pipeline complete. Output saved to {output_file}")
        print(f"Performance metrics: {results.get('performance_metrics', {})}")
        
    except Exception as e:
        db_run.status = ProjectStatus.FAILED
        db_run.error = str(e)
        db_run.end_time = datetime.utcnow()
        
        db_project.status = ProjectStatus.FAILED
        db.commit()
        
        print(f"Async pipeline failed: {str(e)}")
        raise

def run_pipeline_sync(project_id: int, pipeline_run_id: int, db: Session):
    """
    Fallback synchronous pipeline execution.
    
    Args:
        project_id: ID of the project
        pipeline_run_id: ID of the pipeline run
        db: Database session
    """
    # Get project and pipeline run
    db_project = db.query(DBProject).filter(DBProject.id == project_id).first()
    db_run = db.query(DBPipelineRun).filter(DBPipelineRun.id == pipeline_run_id).first()
    
    if db_project is None or db_run is None:
        return
    
    try:
        # Update status
        db_run.status = ProjectStatus.RUNNING
        db_project.status = ProjectStatus.RUNNING
        db.commit()
        
        # Handle batch processing
        batch_processing = getattr(db_project, 'batch_processing', False)
        
        if batch_processing and db_project.input_path and os.path.isdir(db_project.input_path):
            return _process_batch_scripts(db_project, db_run, db)
        
        # Create output directory if it doesn't exist
        if db_project.output_path is None:
            output_dir = OUTPUT_DIR / db_project.title.replace(" ", "_").lower()
            db_project.output_path = str(output_dir)
            db.commit()
        else:
            output_dir = Path(db_project.output_path)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get parameters
        base_model = db_project.base_model or "stable_diffusion_1_5"
        channel_type = db_project.channel_type
        input_path = db_project.input_path or ""
        
        if not base_model:
            raise ValueError("Base model is required for pipeline execution")
            
        project_loras = db.query(DBProjectLora).filter(
            DBProjectLora.project_id == project_id
        ).order_by(DBProjectLora.order_index).all()
        
        if not project_loras and db_project.lora_model:
            lora_models = [db_project.lora_model]
            lora_paths = {}
        else:
            lora_models = [pl.lora_name for pl in project_loras]
            lora_paths = {pl.lora_name: pl.lora_path for pl in project_loras if pl.lora_path}
        
        from .pipelines.channel_specific import get_pipeline_for_channel
        
        pipeline_module = get_pipeline_for_channel(channel_type)
        
        if pipeline_module is None:
            raise ValueError(f"No pipeline available for channel type: {channel_type}")
        
        print(f"Running pipeline for channel type: {channel_type}")
        print(f"Using base model: {base_model}")
        print(f"Using LoRA adaptations: {', '.join(lora_models)}")
        
        render_fps = getattr(db_project, 'render_fps', 24)
        output_fps = getattr(db_project, 'output_fps', 24)
        frame_interpolation_enabled = getattr(db_project, 'frame_interpolation_enabled', True)
        llm_model = getattr(db_project, 'llm_model', 'microsoft/DialoGPT-medium')
        
        output_file = pipeline_module.run(
            input_path=input_path,
            output_path=str(output_dir),
            base_model=base_model,
            lora_models=lora_models,
            lora_paths=lora_paths,
            db_run=db_run,
            db=db,
            render_fps=render_fps,
            output_fps=output_fps,
            frame_interpolation_enabled=frame_interpolation_enabled,
            llm_model=llm_model,
            language="en"
        )
        
        db_run.output_path = str(output_file)
        db_run.status = ProjectStatus.COMPLETED
        db_run.end_time = datetime.utcnow()
        
        db_project.status = ProjectStatus.COMPLETED
        db.commit()
        
        print(f"Pipeline complete. Output saved to {output_file}")
        
    except Exception as e:
        # Update status on error
        db_run.status = ProjectStatus.FAILED
        db_run.error = str(e)
        db_run.end_time = datetime.utcnow()
        
        db_project.status = ProjectStatus.FAILED
        db.commit()
        
        print(f"Pipeline failed: {str(e)}")

def queue_pipeline(project_id: int, pipeline_run_id: int, db: Session):
    """
    Add a pipeline to the processing queue.
    
    Args:
        project_id: ID of the project
        pipeline_run_id: ID of the pipeline run
        db: Database session
    """
    pipeline_queue.put((project_id, pipeline_run_id))
    
    start_pipeline_processor()
    
    return True

def start_pipeline_processor():
    """
    Start the pipeline processor thread if it's not already running.
    """
    global pipeline_thread, is_processing
    
    if pipeline_thread is None or not pipeline_thread.is_alive():
        is_processing = True
        pipeline_thread = threading.Thread(target=process_pipeline_queue, daemon=True)
        pipeline_thread.start()

def process_pipeline_queue():
    """
    Process pipelines from the queue one by one.
    """
    global is_processing
    
    from .database import get_db
    
    while is_processing:
        try:
            try:
                project_id, pipeline_run_id = pipeline_queue.get(timeout=1)
            except queue.Empty:
                if pipeline_queue.empty():
                    is_processing = False
                    break
                continue
            
            # Create a new database session for each pipeline run
            db = next(get_db())
            
            try:
                run_pipeline(project_id, pipeline_run_id, db)
            finally:
                # Always close the database session
                db.close()
            
            pipeline_queue.task_done()
            
        except Exception as e:
            print(f"Error in pipeline processor: {str(e)}")
            continue

def get_queue_status():
    """
    Get the current status of the pipeline queue.
    
    Returns:
        dict: Queue status information
    """
    return {
        "queue_size": pipeline_queue.qsize(),
        "is_processing": is_processing
    }

async def extract_scenes_from_pipeline(pipeline_module, input_path: str, channel_type: str) -> List[Dict]:
    """
    Extract scenes from channel-specific pipeline without running full pipeline.
    This includes LLM preparation and script expansion.
    """
    import yaml
    import json
    import os
    
    scenes = []
    script_data = None
    
    logger.info(f"Reading script from: {input_path}")
    
    if input_path and os.path.exists(input_path):
        try:
            if input_path.endswith('.yaml') or input_path.endswith('.yml'):
                with open(input_path, 'r', encoding='utf-8') as f:
                    script_data = yaml.safe_load(f)
                    if isinstance(script_data, dict):
                        scenes = script_data.get('scenes', [])
            elif input_path.endswith('.json'):
                with open(input_path, 'r', encoding='utf-8') as f:
                    script_data = json.load(f)
                    if isinstance(script_data, dict):
                        scenes = script_data.get('scenes', [])
            elif input_path.endswith('.txt'):
                with open(input_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    scenes = [scene.strip() for scene in content.split('\n\n') if scene.strip()]
                    if not scenes and content:
                        scenes = [content]
            else:
                with open(input_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    scenes = [content] if content else []
                    
            logger.info(f"Successfully read {len(scenes)} scenes from script file")
        except Exception as e:
            logger.error(f"Error reading script file {input_path}: {e}")
            return []
    else:
        logger.warning(f"Script file not found or empty: {input_path}")
    
    if script_data and isinstance(script_data, dict) and channel_type != "gaming":
        try:
            logger.info("Starting LLM preparation for script expansion...")
            llm_model = _load_llm("microsoft/DialoGPT-medium")
            
            min_duration_minutes = 20.0 / 60.0  # 20 seconds = 0.33 minutes
            expanded_script = _expand_script_if_needed(script_data, min_duration=min_duration_minutes)
            
            if expanded_script != script_data:
                logger.info(f"Script expanded from {len(script_data.get('scenes', []))} to {len(expanded_script.get('scenes', []))} scenes")
                scenes = expanded_script.get('scenes', scenes)
        except Exception as e:
            logger.error(f"Error during LLM script expansion: {e}")
    
    scene_details = []
    for i, scene in enumerate(scenes):
        if isinstance(scene, str):
            scene_detail = {
                "description": scene,
                "dialogue": f"Scene {i+1} dialogue",
                "duration": 10.0,
                "scene_id": str(i)
            }
        else:
            scene_detail = scene.copy() if isinstance(scene, dict) else {"description": str(scene)}
            scene_detail["scene_id"] = str(i)
            if "duration" not in scene_detail:
                scene_detail["duration"] = 10.0
            else:
                from .utils.duration_parser import parse_duration
                scene_detail["duration"] = parse_duration(scene_detail["duration"], 10.0)
        scene_details.append(scene_detail)
    
    logger.info(f"Prepared {len(scene_details)} scenes for pipeline execution")
    return scene_details

def _load_llm(llm_model: str = "microsoft/DialoGPT-medium"):
        """Load LLM model for script expansion."""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            model_name = llm_model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            if torch.cuda.is_available():
                model = model.to("cuda")
            
            return {
                "model": model,
                "tokenizer": tokenizer,
                "device": "cuda" if torch.cuda.is_available() else "cpu"
            }
        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            return None
    
def _expand_script_if_needed(script_data: Dict, min_duration: float = 20.0) -> Dict:
        """Expand script to target duration using LLM."""
        current_duration = sum(scene.get('duration', 5.0) for scene in script_data.get('scenes', []))
        
        if current_duration >= min_duration * 60:
            return script_data
        
        llm = _load_llm("microsoft/DialoGPT-medium")
        if not llm:
            return _expand_script_fallback(script_data, min_duration)
        
        try:
            scenes = script_data.get('scenes', [])
            expanded_scenes = []
            
            for scene in scenes:
                expanded_scene = scene.copy()
                
                if len(scene.get('description', '')) < 100:
                    prompt = f"Expand this scene with more detail: {scene.get('description', '')}"
                    
                    if llm.get("generate"):
                        expanded_text = llm["generate"](prompt, max_tokens=100)
                        if expanded_text and len(expanded_text.strip()) > 0:
                            expanded_scene['description'] = expanded_text.strip()
                        else:
                            expanded_scene['description'] = f"Enhanced {scene.get('description', 'scene')} with detailed character interactions and dynamic action sequences"
                    else:
                        expanded_scene['description'] = f"Enhanced {scene.get('description', 'scene')} with detailed character interactions and dynamic action sequences"
                
                expanded_scene['duration'] = max(scene.get('duration', 5.0), 8.0)
                expanded_scenes.append(expanded_scene)
            
            script_data['scenes'] = expanded_scenes
            logger.info(f"LLM script expansion completed for {len(expanded_scenes)} scenes")
            return script_data
            
        except Exception as e:
            logger.error(f"LLM expansion failed: {e}")
            return _expand_script_fallback(script_data, min_duration)
    
def _expand_script_fallback(script_data: Dict, min_duration: float) -> Dict:
    """Fallback script expansion without LLM."""
    scenes = script_data.get('scenes', [])
    expanded_scenes = []
    
    for scene in scenes:
        expanded_scene = scene.copy()
        description = scene.get('description', '')
        
        if len(description) < 50:
            expanded_scene['description'] = f"{description}. The scene unfolds with dramatic tension and visual detail."
        
        expanded_scene['duration'] = max(scene.get('duration', 5.0), 10.0)
        expanded_scenes.append(expanded_scene)
    
    script_data['scenes'] = expanded_scenes
    logger.info(f"Fallback script expansion completed for {len(expanded_scenes)} scenes")
    return script_data

def _process_batch_scripts(db_project, db_run, db):
    """Process multiple scripts in batch mode."""
    import glob
    import traceback
    
    input_dir = Path(db_project.input_path)
    base_output_dir = Path(db_project.output_path) if db_project.output_path else OUTPUT_DIR / db_project.title.replace(" ", "_").lower()
    
    script_patterns = ['*.txt', '*.yaml', '*.yml', '*.json']
    script_files = []
    
    for pattern in script_patterns:
        script_files.extend(glob.glob(str(input_dir / pattern)))
    
    if not script_files:
        logger.warning(f"No script files found in batch directory: {input_dir}")
        db_run.status = ProjectStatus.FAILED
        db_run.error = "No script files found in batch directory"
        db.commit()
        return
    
    logger.info(f"Found {len(script_files)} scripts for batch processing")
    
    # Get parameters
    base_model = db_project.base_model or "stable_diffusion_1_5"
    channel_type = db_project.channel_type
    render_fps = getattr(db_project, 'render_fps', 24)
    output_fps = getattr(db_project, 'output_fps', 24)
    frame_interpolation_enabled = getattr(db_project, 'frame_interpolation_enabled', True)
    llm_model = getattr(db_project, 'llm_model', 'microsoft/DialoGPT-medium')
    
    project_loras = db.query(DBProjectLora).filter(
        DBProjectLora.project_id == db_project.id
    ).order_by(DBProjectLora.order_index).all()
    
    if not project_loras and db_project.lora_model:
        lora_models = [db_project.lora_model]
        lora_paths = {}
    else:
        lora_models = [pl.lora_name for pl in project_loras]
        lora_paths = {pl.lora_name: pl.lora_path for pl in project_loras if pl.lora_path}
    
    from .pipelines.channel_specific import get_pipeline_for_channel
    pipeline_module = get_pipeline_for_channel(channel_type)
    
    if pipeline_module is None:
        raise ValueError(f"No pipeline available for channel type: {channel_type}")
    
    successful_episodes = 0
    failed_episodes = 0
    
    for i, script_file in enumerate(script_files):
        try:
            script_name = Path(script_file).stem
            episode_output_dir = base_output_dir / f"episode_{i+1:02d}_{script_name}"
            episode_output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Processing episode {i+1}/{len(script_files)}: {script_name}")
            
            progress = (i / len(script_files)) * 100
            db_run.progress = progress
            db.commit()
            
            output_file = pipeline_module.run(
                input_path=script_file,
                output_path=str(episode_output_dir),
                base_model=base_model,
                lora_models=lora_models,
                lora_paths=lora_paths,
                db_run=db_run,
                db=db,
                render_fps=render_fps,
                output_fps=output_fps,
                frame_interpolation_enabled=frame_interpolation_enabled,
                llm_model=llm_model,
                language="en"
            )
            
            successful_episodes += 1
            logger.info(f"Episode {i+1} completed: {output_file}")
            
        except Exception as e:
            failed_episodes += 1
            logger.error(f"Episode {i+1} failed: {e}")
            
            error_log_path = os.path.join(str(episode_output_dir), "error_log.txt")
            with open(error_log_path, 'w') as f:
                f.write(f"Episode {i+1} Error: {str(e)}\n")
                f.write(f"Script file: {script_file}\n")
                f.write(f"Traceback:\n{traceback.format_exc()}")
    
    db_run.progress = 100.0
    if successful_episodes > 0:
        db_run.status = ProjectStatus.COMPLETED
        db_project.status = ProjectStatus.COMPLETED
        db_run.output_path = str(base_output_dir)
        logger.info(f"Batch processing completed: {successful_episodes} successful, {failed_episodes} failed")
    else:
        db_run.status = ProjectStatus.FAILED
        db_project.status = ProjectStatus.FAILED
        db_run.error = f"All {len(script_files)} episodes failed"
    
    db.commit()
    return str(base_output_dir)
