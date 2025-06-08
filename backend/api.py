from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from pathlib import Path

from config import OUTPUT_DIR, BASE_MODEL_VERSIONS, CHANNEL_BASE_MODELS
from .database import get_db, DBProject, DBPipelineRun, DBModel, DBSettings, DBProjectLora
from .models import (
    Project, ProjectCreate, ProjectList, ProjectStatus,
    ModelInfo, ModelList, ModelDownload, ModelDownloadStatus,
    PipelineRun, PipelineStatus
)
from .ai_tasks import run_pipeline, queue_pipeline, get_queue_status
from .model_manager import download_model, get_available_models
from .model_version_checker import get_version_checker

# Create FastAPI app
app = FastAPI(title="AI Project Manager API")

# Project endpoints
@app.get("/projects", response_model=ProjectList)
def get_projects(db: Session = Depends(get_db)):
    """
    Get all projects.
    """
    db_projects = db.query(DBProject).all()
    return {"projects": db_projects}

@app.get("/projects/{project_id}", response_model=Project)
def get_project(project_id: int, db: Session = Depends(get_db)):
    """
    Get a project by ID.
    """
    db_project = db.query(DBProject).filter(DBProject.id == project_id).first()
    if db_project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return db_project

@app.post("/projects", response_model=Project)
def create_project(project: ProjectCreate, db: Session = Depends(get_db)):
    """
    Create a new project.
    """
    # Create output path if not provided
    output_path = None
    if project.input_path:
        # Create output path based on project title
        output_dir = OUTPUT_DIR / project.title.replace(" ", "_").lower()
        output_path = str(output_dir)
    
    # Create project in database
    db_project = DBProject(
        title=project.title,
        description=project.description,
        base_model=project.base_model,
        lora_model=project.lora_model,  # Keep for backward compatibility
        channel_type=project.channel_type,
        input_path=project.input_path,
        output_path=output_path,
        video_format=project.video_format,
        upscale_enabled=project.upscale_enabled,
        target_resolution=project.target_resolution,
        status=ProjectStatus.PENDING
    )
    
    db.add(db_project)
    db.commit()
    db.refresh(db_project)
    
    for i, lora in enumerate(project.loras):
        db_lora = DBProjectLora(
            project_id=db_project.id,
            lora_name=lora.lora_name,
            lora_path=lora.lora_path,
            order_index=lora.order_index if lora.order_index is not None else i
        )
        db.add(db_lora)
    
    db.commit()
    db.refresh(db_project)
    
    return db_project

@app.post("/projects/{project_id}/run", response_model=PipelineStatus)
def start_pipeline(project_id: int, db: Session = Depends(get_db)):
    """
    Start the pipeline for a project and add it to the processing queue.
    """
    # Get project
    db_project = db.query(DBProject).filter(DBProject.id == project_id).first()
    if db_project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Update project status
    db_project.status = ProjectStatus.QUEUED
    db.commit()
    
    # Create pipeline run
    db_run = DBPipelineRun(
        project_id=project_id,
        status=ProjectStatus.QUEUED,
        start_time=datetime.utcnow()
    )
    db.add(db_run)
    db.commit()
    db.refresh(db_run)
    
    queue_pipeline(
        project_id=project_id,
        pipeline_run_id=db_run.id,
        db=db
    )
    
    return {
        "project_id": project_id,
        "status": ProjectStatus.QUEUED,
        "progress": 0.0,
        "start_time": db_run.start_time
    }

@app.get("/projects/{project_id}/status", response_model=PipelineStatus)
def get_pipeline_status(project_id: int, db: Session = Depends(get_db)):
    """
    Get the status of a project's pipeline.
    """
    # Get latest pipeline run
    db_run = db.query(DBPipelineRun).filter(
        DBPipelineRun.project_id == project_id
    ).order_by(DBPipelineRun.start_time.desc()).first()
    
    if db_run is None:
        raise HTTPException(status_code=404, detail="No pipeline runs found for this project")
    
    return {
        "project_id": project_id,
        "status": db_run.status,
        "progress": db_run.progress,
        "output_path": db_run.output_path,
        "error": db_run.error,
        "start_time": db_run.start_time,
        "end_time": db_run.end_time
    }

# Model endpoints
@app.get("/models", response_model=ModelList)
def list_models(db: Session = Depends(get_db)):
    """
    List all available models.
    """
    models = get_available_models()
    
    # Update database with ALL models from get_available_models()
    for model_info in models:
        db_model = db.query(DBModel).filter(DBModel.name == model_info["name"]).first()
        
        if db_model is None:
            # Create new model entry
            db_model = DBModel(
                name=model_info["name"],
                version=model_info.get("version", "1.0"),
                model_type=model_info["model_type"],
                channel_compatibility=",".join(model_info.get("channel_compatibility", [])),
                size_mb=model_info["size_mb"],
                downloaded=model_info["downloaded"],
                download_path=model_info.get("download_path", None),
                description=model_info["description"]
            )
            db.add(db_model)
        else:
            db_model.downloaded = model_info["downloaded"]
            db_model.channel_compatibility = ",".join(model_info.get("channel_compatibility", []))
            if model_info.get("download_path"):
                db_model.download_path = model_info.get("download_path")
        
        db.commit()
    
    # Get updated models from database
    db_models = db.query(DBModel).all()
    
    models = []
    for model in db_models:
        compatibility = model.channel_compatibility
        if isinstance(compatibility, str) and compatibility:
            compatibility_list = [c.strip() for c in compatibility.split(",")]
        else:
            compatibility_list = []
        
        models.append(ModelInfo(
            id=model.id,
            name=model.name,
            version=model.version,
            model_type=model.model_type,
            channel_compatibility=compatibility_list,
            size_mb=model.size_mb,
            downloaded=model.downloaded,
            download_path=model.download_path,
            description=model.description
        ))
    
    return {"models": models}

@app.post("/models/download", response_model=ModelDownloadStatus)
def start_model_download(model: ModelDownload, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """
    Start downloading a model (base model or LoRA).
    """
    # Check if model exists
    db_model = db.query(DBModel).filter(DBModel.name == model.name).first()
    
    if db_model is None:
        model_type = "base" if model.name in BASE_MODEL_VERSIONS else "lora"
        
        compatible_channels = []
        if model_type == "base":
            for channel, base_models in CHANNEL_BASE_MODELS.items():
                if model.name in base_models:
                    compatible_channels.append(channel)
        else:
            channel_name = model.name.replace('_style_lora', '').replace('_lora', '')
            if channel_name in CHANNEL_BASE_MODELS:
                compatible_channels.append(channel_name)
        
        # Create new model entry
        db_model = DBModel(
            name=model.name,
            version="latest",
            model_type=model_type,
            channel_compatibility=",".join(compatible_channels),
            size_mb=0.0,
            downloaded=False
        )
        db.add(db_model)
        db.commit()
        db.refresh(db_model)
    
    hf_token = None
    try:
        token_setting = db.query(DBSettings).filter(DBSettings.key == "huggingface_token").first()
        if token_setting:
            hf_token = token_setting.value
    except Exception as e:
        print(f"Error retrieving HuggingFace token: {str(e)}")
    
    # Start download in background
    background_tasks.add_task(
        download_model,
        model_name=model.name,
        db=db,
        model_id=db_model.id,
        hf_token=hf_token
    )
    
    return {
        "name": model.name,
        "status": "downloading",
        "progress": 0.0
    }

@app.get("/models/{model_name}/status", response_model=ModelDownloadStatus)
def get_model_status(model_name: str, db: Session = Depends(get_db)):
    """
    Get the status of a model download.
    """
    db_model = db.query(DBModel).filter(DBModel.name == model_name).first()
    
    if db_model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    
    status = "downloaded" if db_model.downloaded else "pending"
    progress = 100.0 if db_model.downloaded else 0.0
    
    return {
        "name": model_name,
        "status": status,
        "progress": progress
    }

@app.get("/queue/status")
def get_queue_status_endpoint():
    """
    Get the status of the pipeline processing queue.
    """
    status = get_queue_status()
    return status

# Health check endpoint
@app.get("/health")
def health_check():
    """
    Health check endpoint.
    """
    return {"status": "ok"}

@app.get("/templates/{channel_type}")
def get_channel_template(channel_type: str):
    """
    Get the input template for a specific channel type.
    """
    from .templates import get_project_template
    
    template = get_project_template(channel_type)
    if template is None:
        raise HTTPException(status_code=404, detail="Template not found")
    
    return template

@app.get("/templates")
def get_all_channel_templates():
    """
    Get all available channel templates.
    """
    from .templates import get_all_templates
    return get_all_templates()

@app.get("/settings/{key}")
def get_setting(key: str, db: Session = Depends(get_db)):
    """
    Get a setting value by key.
    """
    setting = db.query(DBSettings).filter(DBSettings.key == key).first()
    if setting is None:
        raise HTTPException(status_code=404, detail="Setting not found")
    return {"key": key, "value": setting.value}

@app.post("/settings")
def set_setting(key: str, value: str, db: Session = Depends(get_db)):
    """
    Set a setting value.
    """
    setting = db.query(DBSettings).filter(DBSettings.key == key).first()
    if setting is None:
        setting = DBSettings(key=key, value=value)
        db.add(setting)
    else:
        setting.value = value
    db.commit()
    return {"key": key, "value": value}

@app.delete("/projects/{project_id}")
def delete_project(project_id: int, db: Session = Depends(get_db)):
    """Delete a project and its associated pipeline runs."""
    db_project = db.query(DBProject).filter(DBProject.id == project_id).first()
    if db_project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    
    db.query(DBPipelineRun).filter(DBPipelineRun.project_id == project_id).delete()
    
    db.delete(db_project)
    db.commit()
    
    return {"message": "Project deleted successfully"}

@app.delete("/models/{model_name}")
def delete_model(model_name: str, db: Session = Depends(get_db)):
    """Delete a downloaded model and its files."""
    db_model = db.query(DBModel).filter(DBModel.name == model_name).first()
    
    if db_model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if db_model.download_path and Path(db_model.download_path).exists():
        import shutil
        shutil.rmtree(db_model.download_path, ignore_errors=True)
    
    # Update database
    db_model.downloaded = False
    db_model.download_path = None
    db.commit()
    
    return {"message": f"Model {model_name} deleted successfully"}

@app.get("/models/{model_name}/download-link")
def get_model_download_link(model_name: str, db: Session = Depends(get_db)):
    """
    Get the download link for a LoRA model without downloading it.
    """
    from .model_manager import CIVITAI_LORA_MODELS, get_available_models
    
    if model_name in CIVITAI_LORA_MODELS:
        model_info = CIVITAI_LORA_MODELS[model_name]
        
        civitai_token = None
        try:
            token_setting = db.query(DBSettings).filter(DBSettings.key == "civitai_token").first()
            if token_setting:
                civitai_token = token_setting.value
        except Exception:
            pass
        
        download_urls = [
            f"https://civitai.com/api/download/models/{model_info['version_id']}",
            f"https://api.civitai.com/api/download/models/{model_info['version_id']}"
        ]
        
        return {
            "model_name": model_name,
            "download_urls": download_urls,
            "model_id": model_info['model_id'],
            "version_id": model_info['version_id'],
            "requires_token": civitai_token is None
        }
    
    all_models = get_available_models()
    for model_info in all_models:
        if model_info["name"] == model_name:
            repo = model_info.get("repo", "")
            if repo:
                return {
                    "download_url": f"https://huggingface.co/{repo}",
                    "repo": repo,
                    "name": model_info["name"]
                }
    
    raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found in any mappings")

@app.post("/models/import")
def import_model(import_data: dict, db: Session = Depends(get_db)):
    """
    Import a manually downloaded model file.
    """
    model_name = import_data.get("name")
    file_path = import_data.get("path")
    model_type = import_data.get("model_type", "lora")
    
    if not model_name or not file_path:
        raise HTTPException(status_code=400, detail="Model name and file path required")
    
    if not Path(file_path).exists():
        raise HTTPException(status_code=400, detail="File does not exist")
    
    # Update or create model in database
    db_model = db.query(DBModel).filter(DBModel.name == model_name).first()
    
    if db_model is None:
        # Create new model entry
        db_model = DBModel(
            name=model_name,
            version="manual",
            model_type=model_type,
            channel_compatibility="",
            size_mb=Path(file_path).stat().st_size / (1024 * 1024),
            downloaded=True,
            download_path=str(Path(file_path).parent),
            description=f"Manually imported {model_type} model"
        )
        db.add(db_model)
    else:
        db_model.downloaded = True
        db_model.download_path = str(Path(file_path).parent)
        db_model.size_mb = Path(file_path).stat().st_size / (1024 * 1024)
    
    db.commit()
    
    return {"message": f"Model {model_name} imported successfully", "model_id": db_model.id}

@app.post("/models/lora/register")
def register_lora_model(lora_data: dict, db: Session = Depends(get_db)):
    """
    Register a manually downloaded LoRA model file.
    """
    model_name = lora_data.get("name")
    file_path = lora_data.get("path")
    
    if not model_name or not file_path:
        raise HTTPException(status_code=400, detail="Model name and file path required")
    
    if not Path(file_path).exists():
        raise HTTPException(status_code=400, detail="File does not exist")
    
    # Update or create model in database
    db_model = db.query(DBModel).filter(DBModel.name == model_name).first()
    
    if db_model is None:
        # Create new model entry
        db_model = DBModel(
            name=model_name,
            version="manual",
            model_type="lora",
            channel_compatibility="",
            size_mb=Path(file_path).stat().st_size / (1024 * 1024),
            downloaded=True,
            download_path=str(Path(file_path).parent),
            description=f"Manually imported LoRA model"
        )
        db.add(db_model)
    else:
        db_model.downloaded = True
        db_model.download_path = str(Path(file_path).parent)
        db_model.size_mb = Path(file_path).stat().st_size / (1024 * 1024)
        db_model.model_type = "lora"
    
    db.commit()
    
    return {"message": f"LoRA model {model_name} registered successfully", "model_id": db_model.id}

@app.get("/models/check_updates")
def check_model_updates(token: Optional[str] = None, silent: bool = False, db: Session = Depends(get_db)):
    """
    Check for model updates from HuggingFace.
    
    Args:
        token: HuggingFace API token (optional)
        silent: Whether to run in silent mode (no exceptions)
    
    Returns:
        Dictionary with base_models and loras lists containing update information
    """
    try:
        # If no token provided, try to get from database
        if not token:
            token_setting = db.query(DBSettings).filter(DBSettings.key == "huggingface_token").first()
            if token_setting:
                token = token_setting.value
        
        checker = get_version_checker(token)
        
        updates = checker.check_for_updates(force=not silent)
        return updates
    
    except Exception as e:
        if silent:
            return {"base_models": [], "loras": []}
        else:
            raise HTTPException(status_code=500, detail=f"Error checking for model updates: {str(e)}")
