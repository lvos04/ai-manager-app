"""
GPU Settings Dialog with version-aware CUDA management and manual tier override.
"""
from gui.qt_compat import *
import logging
import threading

logger = logging.getLogger(__name__)

class GPUSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("GPU Settings & CUDA Management")
        self.setModal(True)
        self.resize(600, 500)
        
        self.setup_ui()
        self.load_current_settings()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        detection_group = QGroupBox("Current GPU Detection")
        detection_layout = QVBoxLayout(detection_group)
        
        self.current_gpu_label = QLabel("Detecting...")
        self.current_tier_label = QLabel("Tier: Unknown")
        self.cuda_status_label = QLabel("CUDA: Checking...")
        self.recommended_cuda_label = QLabel("Recommended CUDA: Unknown")
        
        detection_layout.addWidget(QLabel("Detected GPU:"))
        detection_layout.addWidget(self.current_gpu_label)
        detection_layout.addWidget(self.current_tier_label)
        detection_layout.addWidget(self.cuda_status_label)
        detection_layout.addWidget(self.recommended_cuda_label)
        
        refresh_btn = QPushButton("Refresh Detection")
        refresh_btn.clicked.connect(self.refresh_detection)
        detection_layout.addWidget(refresh_btn)
        
        layout.addWidget(detection_group)
        
        override_group = QGroupBox("Manual GPU Tier Override")
        override_layout = QVBoxLayout(override_group)
        
        override_layout.addWidget(QLabel("Override automatic detection:"))
        
        self.tier_combo = QComboBox()
        self.tier_combo.addItems([
            "auto (Automatic Detection)",
            "low (Basic GPU/CPU)",
            "medium (RTX 4060, 8GB VRAM)",
            "high (RTX 4070/3080, 12GB VRAM)", 
            "ultra (RTX 3090/4090, 24GB VRAM)"
        ])
        override_layout.addWidget(self.tier_combo)
        
        override_layout.addWidget(QLabel(
            "Note: Manual override will be used instead of automatic detection.\n"
            "RTX 4060 users should use 'medium' tier.\n"
            "RTX 3090 users should use 'ultra' tier."
        ))
        
        layout.addWidget(override_group)
        
        cuda_group = QGroupBox("Version-Aware CUDA Management")
        cuda_layout = QVBoxLayout(cuda_group)
        
        version_layout = QHBoxLayout()
        version_layout.addWidget(QLabel("CUDA Version:"))
        self.cuda_version_combo = QComboBox()
        self.cuda_version_combo.addItems([
            "auto (Optimal for GPU)",
            "12.1 (RTX 40-series optimal)",
            "11.8 (RTX 30/20-series optimal)"
        ])
        version_layout.addWidget(self.cuda_version_combo)
        cuda_layout.addLayout(version_layout)
        
        self.auto_install_btn = QPushButton("Auto-Install CUDA & PyTorch (Requires Admin)")
        self.auto_install_btn.clicked.connect(self.auto_install_cuda)
        cuda_layout.addWidget(self.auto_install_btn)
        
        self.manual_install_btn = QPushButton("Manual Installation Guide")
        self.manual_install_btn.clicked.connect(self.show_manual_guide)
        cuda_layout.addWidget(self.manual_install_btn)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        cuda_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("")
        cuda_layout.addWidget(self.status_label)
        
        layout.addWidget(cuda_group)
        
        button_layout = QHBoxLayout()
        
        save_btn = QPushButton("Save Settings")
        save_btn.clicked.connect(self.save_settings)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(save_btn)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)

    def load_current_settings(self):
        """Load current GPU detection and settings."""
        try:
            self.refresh_detection()
            
            try:
                from backend.database import get_db
                from backend.models import DBSettings
                
                with get_db() as db:
                    setting = db.query(DBSettings).filter(DBSettings.key == "manual_gpu_tier").first()
                    if setting:
                        tier_map = {
                            "auto": 0, "low": 1, "medium": 2, "high": 3, "ultra": 4
                        }
                        self.tier_combo.setCurrentIndex(tier_map.get(setting.value, 0))
            except:
                pass
                
        except Exception as e:
            logger.error(f"Failed to load settings: {e}")

    def refresh_detection(self):
        """Refresh GPU detection display with version recommendations."""
        try:
            from backend.cuda_installer import CUDAInstaller
            installer = CUDAInstaller()
            gpu_name, recommended_cuda = installer.detect_nvidia_gpu()
            
            if gpu_name:
                self.current_gpu_label.setText(f"{gpu_name}")
                self.recommended_cuda_label.setText(f"Recommended CUDA: {recommended_cuda}")
                
                if recommended_cuda == "12.1":
                    self.cuda_version_combo.setCurrentIndex(1)
                elif recommended_cuda == "11.8":
                    self.cuda_version_combo.setCurrentIndex(2)
            else:
                self.current_gpu_label.setText("No NVIDIA GPU detected")
                self.recommended_cuda_label.setText("Recommended CUDA: N/A")
            
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_name_cuda = torch.cuda.get_device_name(0)
                    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    self.cuda_status_label.setText(f"CUDA: Available ({vram_gb:.1f}GB VRAM)")
                else:
                    self.cuda_status_label.setText("CUDA: Not Available")
            except ImportError:
                self.cuda_status_label.setText("CUDA: PyTorch not installed")
            
            tier = self._detect_vram_tier()
            self.current_tier_label.setText(f"Tier: {tier}")
    
    def _detect_vram_tier(self):
        """Detect VRAM tier for model recommendations."""
        try:
            import torch
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if vram_gb >= 24:
                    return "extreme"
                elif vram_gb >= 16:
                    return "high"
                elif vram_gb >= 8:
                    return "medium"
                else:
                    return "low"
            else:
                return "cpu"
        except Exception:
            return "unknown"
            
        except Exception as e:
            self.current_gpu_label.setText(f"Detection failed: {e}")

    def auto_install_cuda(self):
        """Run automatic CUDA installation with version selection."""
        try:
            version_map = {
                0: "auto",  # Auto-detect optimal version
                1: "12.1",  # RTX 40-series optimal
                2: "11.8"   # RTX 30/20-series optimal
            }
            selected_version = version_map[self.cuda_version_combo.currentIndex()]
            
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate progress
            self.status_label.setText("Starting CUDA installation...")
            self.auto_install_btn.setEnabled(False)
            
            def install_thread():
                try:
                    from backend.cuda_installer import CUDAInstaller
                    installer = CUDAInstaller()
                    
                    if selected_version == "auto":
                        success = installer.auto_setup_cuda()
                    else:
                        gpu_name, _ = installer.detect_nvidia_gpu()
                        if gpu_name:
                            success = (installer.install_cuda_toolkit(selected_version) and 
                                     installer.install_pytorch_cuda(selected_version))
                        else:
                            success = False
                    
                    QTimer.singleShot(0, lambda: self.installation_complete(success, selected_version))
                    
                except Exception as e:
                    QTimer.singleShot(0, lambda: self.installation_error(str(e)))
            
            thread = threading.Thread(target=install_thread)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            self.installation_error(str(e))

    def installation_complete(self, success, version):
        """Handle installation completion."""
        self.progress_bar.setVisible(False)
        self.auto_install_btn.setEnabled(True)
        
        if success:
            self.status_label.setText(f"CUDA {version} installation completed successfully!")
            QMessageBox.information(self, "Success", 
                f"CUDA {version} and PyTorch installed successfully!\n"
                "Please restart the application to use GPU acceleration.")
            self.refresh_detection()
        else:
            self.status_label.setText(f"CUDA {version} installation failed")
            QMessageBox.warning(self, "Installation Failed",
                f"Automatic CUDA {version} installation failed.\n"
                "Please try manual installation or check system requirements.")

    def installation_error(self, error_msg):
        """Handle installation error."""
        self.progress_bar.setVisible(False)
        self.auto_install_btn.setEnabled(True)
        self.status_label.setText(f"Installation error: {error_msg}")
        QMessageBox.critical(self, "Error", f"Installation error: {error_msg}")

    def show_manual_guide(self):
        """Show version-aware manual installation guide."""
        guide_text = """
Version-Aware Manual CUDA Installation Guide:

1. Install NVIDIA Drivers:
   - Download from: https://www.nvidia.com/drivers/
   - Choose your GPU model (RTX 4060, RTX 3090, etc.)

2. Install Optimal CUDA Version:
   
   For RTX 40-series (4060, 4070, 4080, 4090):
   - Download CUDA 12.1: https://developer.nvidia.com/cuda-12-1-0-download-archive
   - PyTorch command: pip install torch --index-url https://download.pytorch.org/whl/cu121
   
   For RTX 30-series (3060, 3070, 3080, 3090):
   - Download CUDA 11.8: https://developer.nvidia.com/cuda-11-8-0-download-archive  
   - PyTorch command: pip install torch --index-url https://download.pytorch.org/whl/cu118
   
   For RTX 20-series (2070, 2080):
   - Download CUDA 11.8: https://developer.nvidia.com/cuda-11-8-0-download-archive
   - PyTorch command: pip install torch --index-url https://download.pytorch.org/whl/cu118

3. Verify Installation:
   - Restart this application
   - Check GPU detection in this dialog
   - Expected tiers: RTX 4060=medium, RTX 3090=ultra

4. Troubleshooting:
   - Run as Administrator for installation
   - Disable antivirus temporarily during installation
   - Ensure Visual C++ Redistributables are installed
"""
        
        QMessageBox.information(self, "Version-Aware Installation Guide", guide_text)

    def save_settings(self):
        """Save manual tier override setting."""
        try:
            tier_values = ["auto", "low", "medium", "high", "ultra"]
            selected_tier = tier_values[self.tier_combo.currentIndex()]
            
            from backend.database import get_db
            from backend.models import DBSettings
            
            with get_db() as db:
                setting = db.query(DBSettings).filter(DBSettings.key == "manual_gpu_tier").first()
                if setting:
                    setting.value = selected_tier
                else:
                    setting = DBSettings(key="manual_gpu_tier", value=selected_tier)
                    db.add(setting)
                db.commit()
            
            QMessageBox.information(self, "Settings Saved", 
                f"GPU tier override set to: {selected_tier}\n"
                "Restart the application to apply changes.")
            
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save settings: {e}")
