#!/bin/bash

set -e

echo "🚀 Installing AI Manager App..."

python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
if [[ $(echo "$python_version >= 3.12" | bc -l) -eq 0 ]]; then
    echo "❌ Python 3.12+ required. Current version: $python_version"
    exit 1
fi

echo "📦 Installing system dependencies..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y imagemagick ffmpeg python3-venv python3-pip git curl
elif command -v yum &> /dev/null; then
    sudo yum install -y ImageMagick ffmpeg python3 python3-pip git curl
elif command -v brew &> /dev/null; then
    brew install imagemagick ffmpeg python3 git curl
else
    echo "⚠️ Please install ImageMagick and FFmpeg manually"
fi

echo "🐍 Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip

echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

echo "🔥 Installing PyTorch with CUDA support..."
pip install torch>=2.6.0 torchvision>=0.19.0 torchaudio>=2.6.0 --index-url https://download.pytorch.org/whl/cu121

echo "📁 Creating directories..."
mkdir -p models output database logs

echo "🗄️ Initializing database..."
python -c "from backend.database import init_db; init_db()"

echo "🧪 Testing installation..."
python -c "
import torch
print(f'✅ PyTorch version: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ CUDA version: {torch.version.cuda}')
    print(f'✅ GPU count: {torch.cuda.device_count()}')
"

echo "✅ Installation complete!"
echo ""
echo "🚀 To start the application:"
echo "   source venv/bin/activate"
echo "   python main.py"
echo ""
echo "🌐 For headless mode:"
echo "   python main.py --headless"
echo ""
echo "📖 See DEPLOYMENT_GUIDE.md for more information"
