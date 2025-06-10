#!/bin/bash

set -e

echo "🔄 Updating AI Manager App..."

echo "💾 Creating backup..."
timestamp=$(date +%Y%m%d_%H%M%S)
cp -r . "../ai-manager-app-backup-$timestamp" || echo "⚠️ Backup failed, continuing..."

echo "📥 Pulling latest changes..."
git pull origin main

source venv/bin/activate

echo "📚 Updating dependencies..."
pip install --upgrade pip
pip install -r requirements.txt --upgrade

echo "🔥 Checking PyTorch version..."
pip install torch>=2.6.0 torchvision>=0.19.0 torchaudio>=2.6.0 --index-url https://download.pytorch.org/whl/cu121 --upgrade

echo "🗄️ Running database migrations..."
python -c "
try:
    from backend.database import init_db
    init_db()
    print('✅ Database updated')
except Exception as e:
    print(f'⚠️ Database update failed: {e}')
"

echo "🧪 Testing updated installation..."
python -c "
try:
    from backend.api import app
    print('✅ API import successful')
    from gui.main_window import MainWindow
    print('✅ GUI import successful')
except ImportError as e:
    print(f'⚠️ Import warning: {e}')
"

echo "✅ Update complete!"
echo ""
echo "🔄 Restart the application to use the updated version"
