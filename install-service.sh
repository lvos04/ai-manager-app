#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="ai-manager"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
INSTALL_DIR="/opt/ai-manager-app"

echo "🔧 Installing AI Manager App as system service..."

if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root (use sudo)" 
   exit 1
fi

mkdir -p "$INSTALL_DIR"

echo "📁 Copying application files..."
cp -r "$SCRIPT_DIR"/* "$INSTALL_DIR/"
chown -R ubuntu:ubuntu "$INSTALL_DIR"

echo "⚙️ Installing systemd service..."
cp "$SCRIPT_DIR/systemd/ai-manager.service" "$SERVICE_FILE"

sed -i "s|/opt/ai-manager-app|$INSTALL_DIR|g" "$SERVICE_FILE"

systemctl daemon-reload
systemctl enable "$SERVICE_NAME"

echo "✅ Service installed successfully!"
echo ""
echo "🚀 To start the service:"
echo "   sudo systemctl start $SERVICE_NAME"
echo ""
echo "📊 To check service status:"
echo "   sudo systemctl status $SERVICE_NAME"
echo ""
echo "📝 To view service logs:"
echo "   sudo journalctl -u $SERVICE_NAME -f"
