# AI Manager App - Production Deployment Guide

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

### 1.1. Manual Environment Setup (Alternative)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. CUDA Setup (GPU Support)
```bash
# For CUDA 12.1 support (recommended)
pip install torch>=2.6.0 torchvision>=0.19.0 torchaudio>=2.6.0 --index-url https://download.pytorch.org/whl/cu121
```

### 3. System Dependencies

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install imagemagick ffmpeg
```

**Windows:**
- Download ImageMagick from https://imagemagick.org/script/download.php#windows
- Download FFmpeg from https://ffmpeg.org/download.html

**macOS:**
```bash
brew install imagemagick ffmpeg
```

### 4. Configuration

#### Model Directory Setup
Edit `config.py` to set your model storage path:
```python
POSSIBLE_MODEL_PATHS = [
    Path("/your/custom/models/path"),  # Your preferred location
    BASE_DIR / "models"                # Default fallback
]
```

#### Database Configuration
The app uses SQLite by default. For production, consider:
```python
DATABASE_URL = "sqlite:///path/to/your/production/database.db"
```

### 5. Launch Application

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

## 🔧 Production Configuration

### Environment Variables
```bash
export HEADLESS=true          # Force headless mode
export API_HOST=0.0.0.0      # Bind to all interfaces
export API_PORT=8000         # API port
export CUDA_VISIBLE_DEVICES=0,1  # GPU selection
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

### Security Considerations

#### API Security
```python
# In production, consider adding:
# - API key authentication
# - Rate limiting
# - HTTPS/TLS encryption
# - Input validation
```

#### File Permissions
```bash
# Secure model directory
chmod 755 models/
chmod 644 models/*

# Secure database
chmod 600 database/app.db
```

## 🐳 Docker Deployment

### Dockerfile
```dockerfile
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    imagemagick \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models output database

# Expose port
EXPOSE 8000

# Run in headless mode
CMD ["python", "main.py", "--headless"]
```

### Docker Compose
```yaml
version: '3.8'
services:
  ai-manager:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./output:/app/output
      - ./database:/app/database
    environment:
      - HEADLESS=true
      - API_HOST=0.0.0.0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

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

## 📊 Scaling Considerations

### Horizontal Scaling
- Deploy multiple instances behind load balancer
- Use shared model storage (NFS/S3)
- Implement job queuing system

### Vertical Scaling
- Multi-GPU support via CUDA_VISIBLE_DEVICES
- Increased RAM for larger models
- NVMe storage for faster model loading

## 🔐 Production Checklist

- [ ] CUDA/GPU drivers installed
- [ ] System dependencies installed (ImageMagick, FFmpeg)
- [ ] Model directories configured
- [ ] Database initialized
- [ ] Health checks working
- [ ] Logging configured
- [ ] Security measures implemented
- [ ] Backup strategy in place
- [ ] Monitoring setup
- [ ] Performance testing completed
- [ ] .run launcher tested and working
- [ ] Python 3.12 compatibility verified
- [ ] Fallback mechanisms for TTS/audiocraft tested
- [ ] Systemd service configured (if needed)
- [ ] Docker deployment tested (if needed)

## 📞 Support

For deployment issues:
1. Check logs in `output/*/error.log`
2. Verify system requirements
3. Test with minimal configuration
4. Review GitHub issues and documentation

---

**Note**: This deployment guide covers production deployment. For development setup, see the main README.md file.
