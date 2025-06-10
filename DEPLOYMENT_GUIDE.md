# AI Manager App - Linux Deployment Guide

## 🚀 Quick Start Deployment

### Prerequisites
- Python 3.12+ (recommended)
- NVIDIA GPU with 8GB+ VRAM (optional but recommended)
- 16GB+ system RAM
- 50GB+ free disk space (for models)

### 1. Quick Setup with .run Launcher

```bash
# Clone and setup
git clone https://github.com/lvos04/ai-manager-app.git
cd ai-manager-app

# Make launcher executable
chmod +x ai-manager.run

# Full deployment (handles everything automatically)
./ai-manager.run deploy
```

### 2. Launch Application

#### Using .run Launcher (Recommended)
```bash
# GUI Mode (Default)
./ai-manager.run start

# Headless Mode
./ai-manager.run start headless

# API Server Mode
./ai-manager.run start api
```

#### Manual Launch (Alternative)
```bash
# Activate environment first
source venv/bin/activate

# GUI Mode (Default)
python main.py

# Headless/API Mode
python main.py --headless

# Production API Server
uvicorn backend.api:app --host 0.0.0.0 --port 8000 --workers 4
```

## 🔧 Configuration

### Model Directory Setup
Edit `config.py` to set your model storage path:
```python
POSSIBLE_MODEL_PATHS = [
    Path("/your/custom/models/path"),  # Your preferred location
    BASE_DIR / "models"                # Default fallback
]
```

### Database Configuration
The app uses SQLite by default:
```python
DATABASE_URL = "sqlite:///database/app.db"
```

### Hardware Optimization

#### VRAM Tiers (Automatic Detection)
- **Low (8GB)**: Basic models, 512px output
- **Medium (16GB)**: Enhanced models, 768px output  
- **High (24GB)**: Premium models, 1024px output
- **Ultra (48GB)**: Maximum quality, 1536px output

#### Model Storage Requirements
- **Base Models**: ~20GB
- **LoRA Models**: ~5GB
- **Audio Models**: ~10GB
- **Video Models**: ~15GB
- **Total Recommended**: 50GB+

## 🔍 Health Monitoring

### Health Check Endpoint
```bash
curl http://localhost:8000/health
```

### Log Monitoring
```bash
# Application logs
tail -f logs/app.log

# Pipeline logs  
tail -f output/*/error.log
```

### Performance Monitoring
- GPU utilization: `nvidia-smi`
- Memory usage: `htop` or `ps aux`
- Disk space: `df -h`

## 🚨 Troubleshooting

### Common Issues

#### Python 3.12 Compatibility Issues
```bash
# TTS and audiocraft packages may not work with Python 3.12
# The application includes fallback mechanisms
# Check logs for compatibility warnings
./ai-manager.run test  # Verify installation
```

#### CUDA Not Detected
```bash
# Verify CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
# Use automatic CUDA setup
./ai-manager.run deploy  # Includes CUDA detection
```

#### Model Download Failures
```bash
# Check internet connectivity
# Verify disk space
# Check model directory permissions
./ai-manager.run test  # Verify setup
```

#### GUI Import Errors
```bash
# Install Qt dependencies
pip install PyQt6 PySide6
# Or run in headless mode
./ai-manager.run start headless
```

#### Memory Issues
```bash
# Reduce batch size in config.py
# Use lower VRAM tier models
# Enable model offloading
```

### .run Launcher Commands
```bash
./ai-manager.run deploy          # Full deployment
./ai-manager.run start [mode]    # Start application
./ai-manager.run test            # Test installation
./ai-manager.run clean           # Clean up installation
./ai-manager.run help            # Show help
```

### Performance Optimization

#### For Low-End Hardware
```python
# In config.py
MODEL_QUALITY_SETTINGS["low"] = {
    "width": 512,
    "height": 512, 
    "steps": 15,
    "batch_size": 1
}
```

#### For High-End Hardware
```python
# In config.py
MODEL_QUALITY_SETTINGS["ultra"] = {
    "width": 2048,
    "height": 2048,
    "steps": 60,
    "batch_size": 8
}
```

## 🔐 Deployment Checklist

- [ ] CUDA/GPU drivers installed
- [ ] System dependencies installed (ImageMagick, FFmpeg)
- [ ] Model directories configured
- [ ] Database initialized
- [ ] Health checks working
- [ ] Logging configured
- [ ] Performance testing completed
- [ ] .run launcher tested and working
- [ ] Python 3.12 compatibility verified
- [ ] Fallback mechanisms for TTS/audiocraft tested

## 📞 Support

For deployment issues:
1. Check logs in `output/*/error.log`
2. Verify system requirements
3. Test with minimal configuration
4. Review GitHub issues and documentation

---

**Note**: This deployment guide covers Linux deployment. The .run launcher script handles all dependency installation and setup automatically.
