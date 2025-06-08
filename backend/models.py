from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict

# Enums for project status and channel types
class ProjectStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class ChannelType(str, Enum):
    GAMING = "gaming"
    ANIME = "anime"
    SUPERHERO = "superhero"
    MANGA = "manga"
    MARVEL_DC = "marvel_dc"
    ORIGINAL_MANGA = "original_manga"

class ProjectLora(BaseModel):
    lora_name: str
    lora_path: Optional[str] = None
    order_index: int = 0
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

class ProjectLoraCreate(BaseModel):
    lora_name: str
    lora_path: Optional[str] = None
    order_index: int = 0

# Project models
class ProjectBase(BaseModel):
    title: str
    description: Optional[str] = None
    base_model: Optional[str] = None  # Add base model field
    lora_model: Optional[str] = None  # Kept for backward compatibility
    channel_type: ChannelType
    input_path: Optional[str] = None
    output_path: Optional[str] = None
    video_format: str = "mp4"  # Default to mp4, can be mp4, webm, mov, avi
    upscale_enabled: bool = True
    target_resolution: str = "1080p"  # 720p, 1080p, 1440p, 4k

class ProjectCreate(ProjectBase):
    loras: List[ProjectLoraCreate] = []

class Project(ProjectBase):
    id: int
    status: ProjectStatus = ProjectStatus.PENDING
    created_at: datetime
    updated_at: datetime
    loras: List[ProjectLora] = []
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

# Model management models
class ModelInfo(BaseModel):
    name: str
    version: str
    model_type: str = "lora"  # Can be "base" or "lora"
    channel_compatibility: Optional[List[str]] = None  # List of compatible channels
    size_mb: float
    downloaded: bool = False
    download_path: Optional[str] = None
    description: Optional[str] = None
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

class ModelDownload(BaseModel):
    name: str

class ModelDownloadStatus(BaseModel):
    name: str
    status: str
    progress: float = 0.0
    error: Optional[str] = None

# Pipeline execution models
class PipelineRun(BaseModel):
    project_id: int

class PipelineStatus(BaseModel):
    project_id: int
    status: ProjectStatus
    progress: float = 0.0
    output_path: Optional[str] = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

# API response models
class ProjectList(BaseModel):
    projects: List[Project]

class ModelList(BaseModel):
    models: List[ModelInfo]

class ErrorResponse(BaseModel):
    error: str
    details: Optional[Dict[str, Any]] = None
