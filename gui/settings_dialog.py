"""
Settings dialog for configuring model paths and other application settings.
"""

import os
import json
from pathlib import Path
from gui.qt_compat import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QLineEdit, QFileDialog, QGroupBox, QFormLayout, QMessageBox,
    QTabWidget, QWidget, QCheckBox, QComboBox, QSpinBox
)

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AI Project Manager - Settings")
        self.setModal(True)
        self.resize(600, 500)
        
        self.settings_file = Path.home() / ".ai_project_manager_settings.json"
        self.settings = self.load_settings()
        
        self.init_ui()
        self.load_current_settings()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        tab_widget = QTabWidget()
        
        model_paths_tab = QWidget()
        self.setup_model_paths_tab(model_paths_tab)
        tab_widget.addTab(model_paths_tab, "Model Paths")
        
        general_tab = QWidget()
        self.setup_general_tab(general_tab)
        tab_widget.addTab(general_tab, "General")
        
        layout.addWidget(tab_widget)
        
        button_layout = QHBoxLayout()
        
        self.save_button = QPushButton("Save Settings")
        self.save_button.clicked.connect(self.save_settings)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
        self.reset_button = QPushButton("Reset to Defaults")
        self.reset_button.clicked.connect(self.reset_to_defaults)
        
        button_layout.addWidget(self.reset_button)
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.save_button)
        
        layout.addLayout(button_layout)
    
    def setup_model_paths_tab(self, tab):
        layout = QVBoxLayout(tab)
        
        paths_group = QGroupBox("Model Directory Paths")
        paths_layout = QFormLayout(paths_group)
        
        self.base_models_path = QLineEdit()
        base_browse_btn = QPushButton("Browse...")
        base_browse_btn.clicked.connect(lambda: self.browse_directory(self.base_models_path, "base"))
        
        base_layout = QHBoxLayout()
        base_layout.addWidget(self.base_models_path)
        base_layout.addWidget(base_browse_btn)
        paths_layout.addRow("Base Models:", base_layout)
        
        self.lora_models_path = QLineEdit()
        lora_browse_btn = QPushButton("Browse...")
        lora_browse_btn.clicked.connect(lambda: self.browse_directory(self.lora_models_path, "loras"))
        
        lora_layout = QHBoxLayout()
        lora_layout.addWidget(self.lora_models_path)
        lora_layout.addWidget(lora_browse_btn)
        paths_layout.addRow("LoRA Models:", lora_layout)
        
        self.video_models_path = QLineEdit()
        video_browse_btn = QPushButton("Browse...")
        video_browse_btn.clicked.connect(lambda: self.browse_directory(self.video_models_path, "video"))
        
        video_layout = QHBoxLayout()
        video_layout.addWidget(self.video_models_path)
        video_layout.addWidget(video_browse_btn)
        paths_layout.addRow("Video Models:", video_layout)
        
        self.audio_models_path = QLineEdit()
        audio_browse_btn = QPushButton("Browse...")
        audio_browse_btn.clicked.connect(lambda: self.browse_directory(self.audio_models_path, "audio"))
        
        audio_layout = QHBoxLayout()
        audio_layout.addWidget(self.audio_models_path)
        audio_layout.addWidget(audio_browse_btn)
        paths_layout.addRow("Audio Models:", audio_layout)
        
        self.text_models_path = QLineEdit()
        text_browse_btn = QPushButton("Browse...")
        text_browse_btn.clicked.connect(lambda: self.browse_directory(self.text_models_path, "text"))

        text_layout = QHBoxLayout()
        text_layout.addWidget(self.text_models_path)
        text_layout.addWidget(text_browse_btn)
        paths_layout.addRow("Text Models:", text_layout)
        
        self.editing_models_path = QLineEdit()
        editing_browse_btn = QPushButton("Browse...")
        editing_browse_btn.clicked.connect(lambda: self.browse_directory(self.editing_models_path, "editing"))

        editing_layout = QHBoxLayout()
        editing_layout.addWidget(self.editing_models_path)
        editing_layout.addWidget(editing_browse_btn)
        paths_layout.addRow("Editing Models:", editing_layout)

        layout.addWidget(paths_group)
        
        auto_detect_btn = QPushButton("Auto-Detect Model Paths")
        auto_detect_btn.clicked.connect(self.auto_detect_paths)
        layout.addWidget(auto_detect_btn)
        
        test_paths_btn = QPushButton("Test Model Detection")
        test_paths_btn.clicked.connect(self.test_model_detection)
        layout.addWidget(test_paths_btn)
        
        layout.addStretch()
    
    def setup_general_tab(self, tab):
        layout = QVBoxLayout(tab)
        
        output_group = QGroupBox("Output Settings")
        output_layout = QFormLayout(output_group)
        
        self.output_dir_path = QLineEdit()
        output_browse_btn = QPushButton("Browse...")
        output_browse_btn.clicked.connect(lambda: self.browse_directory(self.output_dir_path, "output"))
        
        output_layout_h = QHBoxLayout()
        output_layout_h.addWidget(self.output_dir_path)
        output_layout_h.addWidget(output_browse_btn)
        output_layout.addRow("Default Output Directory:", output_layout_h)
        
        layout.addWidget(output_group)
        
        perf_group = QGroupBox("Performance Settings")
        perf_layout = QFormLayout(perf_group)
        
        self.gpu_enabled = QCheckBox("Enable GPU Acceleration")
        perf_layout.addRow("GPU:", self.gpu_enabled)
        
        self.vram_tier = QComboBox()
        self.vram_tier.addItems(["low (8GB)", "medium (16GB)", "high (24GB)", "ultra (48GB)"])
        perf_layout.addRow("VRAM Tier:", self.vram_tier)
        
        self.concurrent_tasks = QSpinBox()
        self.concurrent_tasks.setRange(1, 16)
        self.concurrent_tasks.setValue(6)
        perf_layout.addRow("Concurrent Tasks:", self.concurrent_tasks)
        
        layout.addWidget(perf_group)
        layout.addStretch()
    
    def browse_directory(self, line_edit, model_type):
        """Browse for directory and update line edit."""
        current_path = line_edit.text() or str(Path.home())
        
        if model_type == "output":
            title = "Select Output Directory"
        else:
            title = f"Select {model_type.title()} Models Directory"
        
        directory = QFileDialog.getExistingDirectory(
            self, title, current_path
        )
        
        if directory:
            line_edit.setText(directory)
    
    def auto_detect_paths(self):
        """Auto-detect model paths from common locations."""
        common_paths = [
            Path("/media/leon/NieuwVolume/AI app/models"),
            Path("G:/ai_project_manager_app/models"),
            Path.home() / "models",
            Path.cwd() / "models",
            Path("/opt/ai_models"),
            Path("C:/AI_Models"),
        ]
        
        detected_paths = {}
        
        for base_path in common_paths:
            if base_path.exists():
                for model_type in ["base", "loras", "video", "audio", "text", "editing"]:
                    model_dir = base_path / model_type
                    if model_dir.exists() and any(model_dir.iterdir()):
                        detected_paths[model_type] = str(model_dir)
                        break
        
        if detected_paths:
            if "base" in detected_paths:
                self.base_models_path.setText(detected_paths["base"])
            if "loras" in detected_paths:
                self.lora_models_path.setText(detected_paths["loras"])
            if "video" in detected_paths:
                self.video_models_path.setText(detected_paths["video"])
            if "audio" in detected_paths:
                self.audio_models_path.setText(detected_paths["audio"])
            if "text" in detected_paths:
                self.text_models_path.setText(detected_paths["text"])
            if "editing" in detected_paths:
                self.editing_models_path.setText(detected_paths["editing"])
            
            QMessageBox.information(
                self, "Auto-Detection Complete",
                f"Found model directories in {len(detected_paths)} categories.\n"
                "Please verify the paths are correct."
            )
        else:
            QMessageBox.warning(
                self, "No Models Found",
                "Could not auto-detect any model directories.\n"
                "Please manually specify the paths."
            )
    
    def test_model_detection(self):
        """Test model detection with current paths."""
        try:
            test_paths = {
                "base": self.base_models_path.text(),
                "loras": self.lora_models_path.text(),
                "video": self.video_models_path.text(),
                "audio": self.audio_models_path.text(),
                "text": self.text_models_path.text(),
                "editing": self.editing_models_path.text(),
            }
            
            results = []
            total_models = 0
            
            for model_type, path in test_paths.items():
                if path and Path(path).exists():
                    model_files = []
                    for ext in ['.safetensors', '.ckpt', '.bin', '.pt', '.pth']:
                        model_files.extend(Path(path).glob(f"*{ext}"))
                        model_files.extend(Path(path).glob(f"**/*{ext}"))
                    
                    count = len(model_files)
                    total_models += count
                    results.append(f"{model_type.title()}: {count} models found")
                else:
                    results.append(f"{model_type.title()}: Path not found")
            
            message = "\n".join(results)
            message += f"\n\nTotal models detected: {total_models}"
            
            if total_models > 0:
                QMessageBox.information(self, "Model Detection Results", message)
            else:
                QMessageBox.warning(self, "No Models Detected", message)
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error testing model detection:\n{str(e)}")
    
    def load_settings(self):
        """Load settings from file."""
        if self.settings_file.exists():
            try:
                with open(self.settings_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        
        return {
            "model_paths": {
                "base": "",
                "loras": "",
                "video": "",
                "audio": "",
                "text": "",
                "editing": ""
            },
            "output_dir": "",
            "gpu_enabled": False,
            "vram_tier": "low",
            "concurrent_tasks": 6
        }
    
    def load_current_settings(self):
        """Load current settings into UI."""
        model_paths = self.settings.get("model_paths", {})
        
        self.base_models_path.setText(model_paths.get("base", ""))
        self.lora_models_path.setText(model_paths.get("loras", ""))
        self.video_models_path.setText(model_paths.get("video", ""))
        self.audio_models_path.setText(model_paths.get("audio", ""))
        self.text_models_path.setText(model_paths.get("text", ""))
        self.editing_models_path.setText(model_paths.get("editing", ""))
        
        self.output_dir_path.setText(self.settings.get("output_dir", ""))
        self.gpu_enabled.setChecked(self.settings.get("gpu_enabled", False))
        
        vram_tier = self.settings.get("vram_tier", "low")
        vram_index = {"low": 0, "medium": 1, "high": 2, "ultra": 3}.get(vram_tier, 0)
        self.vram_tier.setCurrentIndex(vram_index)
        
        self.concurrent_tasks.setValue(self.settings.get("concurrent_tasks", 6))
    
    def save_settings(self):
        """Save settings to file."""
        try:
            self.settings["model_paths"] = {
                "base": self.base_models_path.text(),
                "loras": self.lora_models_path.text(),
                "video": self.video_models_path.text(),
                "audio": self.audio_models_path.text(),
                "text": self.text_models_path.text(),
                "editing": self.editing_models_path.text()
            }
            
            self.settings["output_dir"] = self.output_dir_path.text()
            self.settings["gpu_enabled"] = self.gpu_enabled.isChecked()
            
            vram_tiers = ["low", "medium", "high", "ultra"]
            self.settings["vram_tier"] = vram_tiers[self.vram_tier.currentIndex()]
            
            self.settings["concurrent_tasks"] = self.concurrent_tasks.value()
            
            with open(self.settings_file, 'w') as f:
                json.dump(self.settings, f, indent=2)
            
            QMessageBox.information(
                self, "Settings Saved",
                "Settings have been saved successfully.\n"
                "Please restart the application for all changes to take effect."
            )
            
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saving settings:\n{str(e)}")
    
    def reset_to_defaults(self):
        """Reset all settings to defaults."""
        reply = QMessageBox.question(
            self, "Reset Settings",
            "Are you sure you want to reset all settings to defaults?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.settings = {
                "model_paths": {
                    "base": "",
                    "loras": "",
                    "video": "",
                    "audio": "",
                    "text": "",
                    "editing": ""
                },
                "output_dir": "",
                "gpu_enabled": False,
                "vram_tier": "low",
                "concurrent_tasks": 6
            }
            self.load_current_settings()
    
    def get_model_paths(self):
        """Get the configured model paths."""
        return self.settings.get("model_paths", {})
