import os
from datetime import datetime
from pathlib import Path
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from typing import List, Optional

from config import DATABASE_URL

# Create database directory if it doesn't exist
db_dir = Path(__file__).parent.parent / "database"
db_dir.mkdir(exist_ok=True)

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database models
class DBProject(Base):
    __tablename__ = "projects"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    base_model = Column(String(255), nullable=True)  # Add base model field
    lora_model = Column(String(255), nullable=True)  # Kept for backward compatibility
    channel_type = Column(String(50), nullable=False)
    status = Column(String(50), default="pending")
    input_path = Column(String(255), nullable=True)
    output_path = Column(String(255), nullable=True)
    video_format = Column(String(10), default="mp4")
    upscale_enabled = Column(Boolean, default=True)
    target_resolution = Column(String(10), default="1080p")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    pipeline_runs = relationship("DBPipelineRun", back_populates="project", cascade="all, delete-orphan")
    loras = relationship("DBProjectLora", back_populates="project", cascade="all, delete-orphan")

class DBPipelineRun(Base):
    __tablename__ = "pipeline_runs"
    
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    status = Column(String(50), default="pending")
    progress = Column(Float, default=0.0)
    output_path = Column(String(255), nullable=True)
    error = Column(Text, nullable=True)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    
    # Relationships
    project = relationship("DBProject", back_populates="pipeline_runs")

class DBModel(Base):
    __tablename__ = "models"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, unique=True)
    version = Column(String(50), nullable=False)
    model_type = Column(String(50), default="lora")  # "base" or "lora"
    channel_compatibility = Column(String(255), nullable=True)  # Comma-separated list of compatible channels
    size_mb = Column(Float, default=0.0)
    downloaded = Column(Boolean, default=False)
    download_path = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class DBSettings(Base):
    __tablename__ = "settings"
    
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(255), nullable=False, unique=True)
    value = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class DBProjectLora(Base):
    __tablename__ = "project_loras"
    
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    lora_name = Column(String(255), nullable=False)
    lora_path = Column(String(255), nullable=True)  # Path to manually selected LoRA file
    order_index = Column(Integer, nullable=False, default=0)  # Order of application
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    project = relationship("DBProject", back_populates="loras")

def init_db():
    """
    Initialize the database by creating all tables.
    """
    Base.metadata.create_all(bind=engine)

def get_db():
    """
    Get a database session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
def get_db_connection():
    """Get a direct database connection for testing purposes."""
    import sqlite3
    from config import DATABASE_URL
    db_path = DATABASE_URL.replace('sqlite:///', '')
    return sqlite3.connect(db_path)
