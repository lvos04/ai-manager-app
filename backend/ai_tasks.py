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
            scenes = [
                {"description": "Scene 1", "dialogue": "Hello world", "duration": 10.0},
                {"description": "Scene 2", "dialogue": "This is a test", "duration": 8.0}
            ]
        
        pipeline_config = {
            "base_model": base_model,
            "channel_type": channel_type,
            "lora_models": lora_models,
            "lora_paths": lora_paths,
            "output_path": str(output_dir)
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
            llm_model = _load_llm()
            
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
                "duration": "10.0",
                "scene_id": str(i)
            }
        else:
            scene_detail = scene.copy() if isinstance(scene, dict) else {"description": str(scene)}
            scene_detail["scene_id"] = str(i)
            if "duration" not in scene_detail:
                scene_detail["duration"] = "10.0"
        scene_details.append(scene_detail)
    
    logger.info(f"Prepared {len(scene_details)} scenes for pipeline execution")
    return scene_details

def _load_llm():
        """Load LLM model for script expansion."""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            model_name = "microsoft/DialoGPT-medium"
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
        
        llm = _load_llm()
        if not llm:
            return _expand_script_fallback(script_data, min_duration)
        
        try:
            import torch
            scenes = script_data.get('scenes', [])
            expanded_scenes = []
            
            for scene in scenes:
                expanded_scene = scene.copy()
                
                if len(scene.get('description', '')) < 100:
                    prompt = f"Expand this scene with more detail: {scene.get('description', '')}"
                    
                    inputs = llm["tokenizer"](prompt, return_tensors="pt", truncation=True, max_length=512)
                    if torch.cuda.is_available():
                        inputs = {k: v.to(llm["device"]) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = llm["model"].generate(
                            **inputs,
                            max_length=inputs["input_ids"].shape[1] + 100,
                            num_return_sequences=1,
                            temperature=0.8,
                            do_sample=True,
                            pad_token_id=llm["tokenizer"].eos_token_id
                        )
                    
                    expanded_text = llm["tokenizer"].decode(outputs[0], skip_special_tokens=True)
                    expanded_scene['description'] = expanded_text[len(prompt):].strip()
                
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
