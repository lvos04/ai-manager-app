FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    imagemagick \
    ffmpeg \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with Python 3.12 compatibility fixes
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir wheel setuptools && \
    # Install PyTorch first with CUDA support
    pip install --no-cache-dir torch>=2.5.0 torchvision>=0.18.0 torchaudio>=2.5.0 --index-url https://download.pytorch.org/whl/cu121 && \
    # Install other dependencies, skipping problematic packages
    pip install --no-cache-dir -r requirements.txt || true && \
    # Install essential packages that might have failed
    pip install --no-cache-dir fastapi uvicorn sqlalchemy pydantic pillow numpy opencv-python

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models output database logs && \
    chmod 755 models output database logs

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "main.py", "--headless"]
