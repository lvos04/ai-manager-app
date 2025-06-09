from .qt_compat import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
    QTextEdit, QPushButton, QComboBox, QFileDialog, QCheckBox,
    QGroupBox, QListWidget, QListWidgetItem, QAbstractItemView,
    QWidget, QToolButton, QScrollArea, QMessageBox, Qt, Signal,
    QIcon, QDrag, QPixmap, QDesktopServices, QUrl, QSpinBox
)
import requests
import os
from config import API_HOST, API_PORT
from .widgets.multi_language_widget import MultiLanguageSelectionWidget
from .styles import get_app_stylesheet, get_header_style, get_warning_style, get_success_style, get_subheader_style, get_info_style

try:
    from lora_config import CHANNEL_LORAS, LORA_DESCRIPTIONS, DEFAULT_LORA_COMBINATIONS
except ImportError as e:
    print(f"Warning: Could not import lora_config: {e}")
    CHANNEL_LORAS = {}
    LORA_DESCRIPTIONS = {}
    DEFAULT_LORA_COMBINATIONS = {}

class MultiLoRASelectionWidget(QWidget):
    """Widget for selecting multiple LoRAs with drag-and-drop reordering."""
    
    lora_selection_changed = Signal()
    
    def __init__(self, channel_type="gaming", parent=None):
        super().__init__(parent)
        self.channel_type = channel_type
        self.max_loras = 5
        self.selected_loras = []
        self.api_url = f"http://{API_HOST}:{API_PORT}"
        
        self.init_ui()
        self.load_loras()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        available_group = QGroupBox("Available LoRAs")
        available_layout = QVBoxLayout()
        
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter LoRAs...")
        self.filter_edit.textChanged.connect(self.filter_loras)
        available_layout.addWidget(self.filter_edit)
        
        self.available_list = QListWidget()
        self.available_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.available_list.setDragEnabled(True)
        self.available_list.itemDoubleClicked.connect(self.add_selected_lora)
        available_layout.addWidget(self.available_list)
        
        add_button = QPushButton("Add Selected LoRA")
        add_button.clicked.connect(self.add_selected_lora)
        available_layout.addWidget(add_button)
        
        download_links_button = QPushButton("Get Download Links...")
        download_links_button.clicked.connect(self.show_download_links)
        available_layout.addWidget(download_links_button)
        
        browse_button = QPushButton("Browse Local LoRA File...")
        browse_button.clicked.connect(self.browse_lora_file)
        available_layout.addWidget(browse_button)
        
        available_group.setLayout(available_layout)
        
        selected_group = QGroupBox("Selected LoRAs (Max 5)")
        selected_layout = QVBoxLayout()
        
        self.selected_list = QListWidget()
        self.selected_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.selected_list.setDragEnabled(True)
        self.selected_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.selected_list.setAcceptDrops(True)
        self.selected_list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.selected_list.model().rowsMoved.connect(self.update_selected_loras)
        selected_layout.addWidget(self.selected_list)
        
        remove_button = QPushButton("Remove Selected LoRA")
        remove_button.clicked.connect(self.remove_selected_lora)
        selected_layout.addWidget(remove_button)
        
        combo_button = QPushButton("Use Recommended Combination")
        combo_button.clicked.connect(self.show_recommended_combinations)
        selected_layout.addWidget(combo_button)
        
        selected_group.setLayout(selected_layout)
        
        main_layout = QHBoxLayout()
        main_layout.addWidget(available_group)
        main_layout.addWidget(selected_group)
        layout.addLayout(main_layout)
        
        # Description section
        desc_label = QLabel("Selected LoRA Description:")
        layout.addWidget(desc_label)
        
        self.desc_text = QTextEdit()
        self.desc_text.setReadOnly(True)
        self.desc_text.setMaximumHeight(80)
        layout.addWidget(self.desc_text)
        
        # Connect selection change to update description
        self.available_list.currentItemChanged.connect(self.update_description)
        self.selected_list.currentItemChanged.connect(self.update_description)
    
    def load_loras(self):
        """Load LoRAs for the current channel type."""
        self.available_list.clear()
        
        try:
            response = requests.get(f"{self.api_url}/models")
            if response.status_code == 200:
                models_data = response.json()
                
                for model in models_data.get("models", []):
                    if (model.get("model_type") == "lora" and 
                        self.channel_type in model.get("channel_compatibility", [])):
                        item = QListWidgetItem(model["name"])
                        item.setData(Qt.ItemDataRole.UserRole, {"name": model["name"], "path": None})
                        self.available_list.addItem(item)
            
            if self.available_list.count() == 0:
                if self.channel_type in CHANNEL_LORAS:
                    for lora_name in CHANNEL_LORAS[self.channel_type]:
                        item = QListWidgetItem(lora_name)
                        item.setData(Qt.ItemDataRole.UserRole, {"name": lora_name, "path": None})
                        self.available_list.addItem(item)
                
        except Exception as e:
            print(f"Failed to load LoRA models: {str(e)}")
            
            if self.channel_type in CHANNEL_LORAS:
                for lora_name in CHANNEL_LORAS[self.channel_type]:
                    item = QListWidgetItem(lora_name)
                    item.setData(Qt.ItemDataRole.UserRole, {"name": lora_name, "path": None})
                    self.available_list.addItem(item)
    
    def filter_loras(self, text):
        """Filter the available LoRAs list based on search text."""
        for i in range(self.available_list.count()):
            item = self.available_list.item(i)
            if text.lower() in item.text().lower():
                item.setHidden(False)
            else:
                item.setHidden(True)
    
    def add_selected_lora(self):
        """Add the selected LoRA to the selected list."""
        if self.selected_list.count() >= self.max_loras:
            return
            
        current_item = self.available_list.currentItem()
        if current_item:
            lora_data = current_item.data(Qt.ItemDataRole.UserRole)
            
            for i in range(self.selected_list.count()):
                item = self.selected_list.item(i)
                if item.data(Qt.ItemDataRole.UserRole)["name"] == lora_data["name"]:
                    return
            
            new_item = QListWidgetItem(current_item.text())
            new_item.setData(Qt.ItemDataRole.UserRole, lora_data)
            self.selected_list.addItem(new_item)
            
            self.update_selected_loras()
            self.lora_selection_changed.emit()
    
    def remove_selected_lora(self):
        """Remove the selected LoRA from the selected list."""
        current_item = self.selected_list.currentItem()
        if current_item:
            row = self.selected_list.row(current_item)
            self.selected_list.takeItem(row)
            
            self.update_selected_loras()
            self.lora_selection_changed.emit()
    
    def update_selected_loras(self):
        """Update the internal list of selected LoRAs."""
        self.selected_loras = []
        for i in range(self.selected_list.count()):
            item = self.selected_list.item(i)
            lora_data = item.data(Qt.ItemDataRole.UserRole)
            self.selected_loras.append({
                "lora_name": lora_data["name"],
                "lora_path": lora_data["path"],
                "order_index": i
            })
    
    def update_description(self, current, previous):
        """Update the description text based on the selected LoRA."""
        if not current:
            self.desc_text.clear()
            return
            
        lora_name = current.data(Qt.ItemDataRole.UserRole)["name"]
        if lora_name in LORA_DESCRIPTIONS:
            self.desc_text.setText(LORA_DESCRIPTIONS[lora_name])
        else:
            self.desc_text.setText(f"No description available for {lora_name}")
    
    def show_download_links(self):
        """Show download links for available LoRA models."""
        if not hasattr(self, 'channel_type') or not self.channel_type:
            QMessageBox.warning(self, "Error", "Please select a channel type first")
            return
        
        from lora_config import CHANNEL_LORAS
        if self.channel_type not in CHANNEL_LORAS:
            QMessageBox.warning(self, "Error", f"No LoRA models available for {self.channel_type}")
            return
        
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Download Links for {self.channel_type.title()} LoRAs")
        dialog.setMinimumWidth(600)
        dialog.setMinimumHeight(400)
        
        layout = QVBoxLayout(dialog)
        
        info_label = QLabel(f"Download links for {self.channel_type} channel LoRA models:")
        info_label.setStyleSheet("font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(info_label)
        
        instructions_label = QLabel("1. Click a download link to open it in your browser\n2. Download the file manually\n3. Use 'Browse Local File...' to import it")
        layout.addWidget(instructions_label)
        
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        api_url = f"http://{API_HOST}:{API_PORT}"
        
        for lora_name in CHANNEL_LORAS[self.channel_type]:
            lora_group = QGroupBox(lora_name)
            lora_layout = QVBoxLayout()
            
            try:
                response = requests.get(f"{api_url}/models/{lora_name}/download-link", timeout=5)
                if response.status_code == 200:
                    link_data = response.json()
                    
                    for i, url in enumerate(link_data['download_urls']):
                        link_button = QPushButton(f"Download Link {i+1}")
                        link_button.clicked.connect(lambda checked, u=url: self.open_download_url(u))
                        lora_layout.addWidget(link_button)
                else:
                    error_label = QLabel("Download links not available")
                    error_label.setStyleSheet("color: red;")
                    lora_layout.addWidget(error_label)
            except Exception:
                error_label = QLabel("Error getting download links")
                error_label.setStyleSheet("color: red;")
                lora_layout.addWidget(error_label)
            
            lora_group.setLayout(lora_layout)
            scroll_layout.addWidget(lora_group)
        
        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)
        
        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button)
        
        dialog.exec()
    
    def open_download_url(self, url):
        """Open download URL in browser."""
        QDesktopServices.openUrl(QUrl(url))

    def browse_lora_file(self):
        """Browse for a local LoRA file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select LoRA File", 
            "", 
            "LoRA Files (*.safetensors *.ckpt *.pt);;All Files (*)"
        )
        
        if file_path:
            file_name = os.path.basename(file_path)
            base_name, _ = os.path.splitext(file_name)
            
            new_item = QListWidgetItem(f"{base_name} (Local)")
            new_item.setData(Qt.ItemDataRole.UserRole, {"name": base_name, "path": file_path})
            self.selected_list.addItem(new_item)
            
            self.update_selected_loras()
            self.lora_selection_changed.emit()
    
    def show_recommended_combinations(self):
        """Show a dialog with recommended LoRA combinations."""
        if self.channel_type not in DEFAULT_LORA_COMBINATIONS:
            return
            
        combo_dialog = QDialog(self)
        combo_dialog.setWindowTitle("Recommended LoRA Combinations")
        combo_dialog.setMinimumWidth(400)
        combo_dialog.setStyleSheet(get_app_stylesheet())
        
        layout = QVBoxLayout(combo_dialog)
        
        label = QLabel("Select a recommended combination:")
        layout.addWidget(label)
        
        combo_list = QListWidget()
        for i, combo in enumerate(DEFAULT_LORA_COMBINATIONS[self.channel_type]):
            combo_text = " + ".join(combo)
            item = QListWidgetItem(f"Combination {i+1}: {combo_text}")
            item.setData(Qt.ItemDataRole.UserRole, combo)
            combo_list.addItem(item)
        
        layout.addWidget(combo_list)
        
        button_layout = QHBoxLayout()
        cancel_button = QPushButton("Cancel")
        select_button = QPushButton("Select")
        
        cancel_button.clicked.connect(combo_dialog.reject)
        select_button.clicked.connect(lambda: self.apply_combination(combo_list.currentItem(), combo_dialog))
        
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(select_button)
        layout.addLayout(button_layout)
        
        combo_dialog.exec()
    
    def apply_combination(self, item, dialog):
        """Apply the selected LoRA combination."""
        if not item:
            dialog.reject()
            return
            
        combo = item.data(Qt.ItemDataRole.UserRole)
        
        # Clear current selection
        self.selected_list.clear()
        
        for i, lora_name in enumerate(combo):
            new_item = QListWidgetItem(lora_name)
            new_item.setData(Qt.ItemDataRole.UserRole, {"name": lora_name, "path": None})
            self.selected_list.addItem(new_item)
        
        self.update_selected_loras()
        self.lora_selection_changed.emit()
        dialog.accept()
    
    def set_channel_type(self, channel_type):
        """Set the channel type and reload LoRAs."""
        self.channel_type = channel_type
        self.selected_list.clear()
        self.selected_loras = []
        self.load_loras()
        self.lora_selection_changed.emit()
    
    def get_selected_loras(self):
        """Get the list of selected LoRAs."""
        return self.selected_loras

class MultiLanguageSelectionWidget(QWidget):
    """Widget for selecting up to 4 languages for multi-language video generation."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_languages = []
        self.max_languages = 4
        
        self.available_languages = [
            ("English", "en"),
            ("Mandarin Chinese", "zh-cn"), 
            ("Spanish", "es"),
            ("Hindi", "hi"),
            ("Arabic", "ar"),
            ("Bengali", "bn"),
            ("Portuguese", "pt"),
            ("Russian", "ru"),
            ("Japanese", "ja"),
            ("French", "fr")
        ]
        
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        instructions = QLabel(f"Select up to {self.max_languages} languages for voice generation:")
        instructions.setStyleSheet("font-weight: bold; color: #1976D2;")
        layout.addWidget(instructions)
        
        available_label = QLabel("Available Languages:")
        layout.addWidget(available_label)
        
        self.available_list = QListWidget()
        self.available_list.setMaximumHeight(150)
        for lang_name, lang_code in self.available_languages:
            item = QListWidgetItem(f"{lang_name} ({lang_code})")
            item.setData(Qt.UserRole, lang_code)
            self.available_list.addItem(item)
        layout.addWidget(self.available_list)
        
        button_layout = QHBoxLayout()
        self.add_button = QPushButton("Add Language →")
        self.add_button.clicked.connect(self.add_language)
        self.remove_button = QPushButton("← Remove Language")
        self.remove_button.clicked.connect(self.remove_language)
        button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.remove_button)
        layout.addLayout(button_layout)
        
        selected_label = QLabel("Selected Languages:")
        layout.addWidget(selected_label)
        
        self.selected_list = QListWidget()
        self.selected_list.setMaximumHeight(120)
        layout.addWidget(self.selected_list)
        
        self.add_default_language()
    
    def add_default_language(self):
        """Add English as default language."""
        for i in range(self.available_list.count()):
            item = self.available_list.item(i)
            if item.data(Qt.UserRole) == "en":
                self.available_list.setCurrentItem(item)
                self.add_language()
                break
    
    def add_language(self):
        """Add selected language to the list."""
        current_item = self.available_list.currentItem()
        if not current_item:
            return
        
        lang_code = current_item.data(Qt.UserRole)
        if lang_code in self.selected_languages:
            return
        
        if len(self.selected_languages) >= self.max_languages:
            QMessageBox.warning(self, "Language Limit", f"You can select up to {self.max_languages} languages only.")
            return
        
        self.selected_languages.append(lang_code)
        self.update_selected_list()
    
    def remove_language(self):
        """Remove selected language from the list."""
        current_item = self.selected_list.currentItem()
        if not current_item:
            return
        
        lang_code = current_item.data(Qt.UserRole)
        if lang_code in self.selected_languages:
            self.selected_languages.remove(lang_code)
            self.update_selected_list()
    
    def update_selected_list(self):
        """Update the selected languages display."""
        self.selected_list.clear()
        for lang_code in self.selected_languages:
            lang_name = next((name for name, code in self.available_languages if code == lang_code), lang_code)
            item = QListWidgetItem(f"{lang_name} ({lang_code})")
            item.setData(Qt.UserRole, lang_code)
            self.selected_list.addItem(item)
    
    def get_selected_languages(self):
        """Get list of selected language codes."""
        return self.selected_languages.copy()


class NewProjectDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.api_url = f"http://{API_HOST}:{API_PORT}"
        self.setWindowTitle("Create New Project")
        self.setMinimumWidth(900)
        self.setMinimumHeight(800)
        self.resize(1000, 900)
        self.setStyleSheet(get_app_stylesheet())
        
        self.init_ui()
        self.on_channel_changed()
    
    def init_ui(self):
        scroll_area = QScrollArea(self)
        scroll_widget = QWidget()
        layout = QVBoxLayout(scroll_widget)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title_layout = QHBoxLayout()
        title_label = QLabel("Title:")
        title_label.setMinimumWidth(120)
        self.title_input = QLineEdit()
        title_layout.addWidget(title_label)
        title_layout.addWidget(self.title_input)
        layout.addLayout(title_layout)
        
        # Description
        desc_layout = QVBoxLayout()
        desc_label = QLabel("Description:")
        self.desc_input = QTextEdit()
        self.desc_input.setMaximumHeight(80)
        self.desc_input.setMinimumHeight(60)
        desc_layout.addWidget(desc_label)
        desc_layout.addWidget(self.desc_input)
        layout.addLayout(desc_layout)
        
        # Input path
        input_layout = QHBoxLayout()
        input_label = QLabel("Input Path:")
        input_label.setMinimumWidth(120)
        self.input_path = QLineEdit()
        browse_button = QPushButton("Browse...")
        browse_button.setMaximumWidth(100)
        browse_button.clicked.connect(self.browse_input)
        input_layout.addWidget(input_label)
        input_layout.addWidget(self.input_path)
        input_layout.addWidget(browse_button)
        layout.addLayout(input_layout)
        
        # Output path selection
        output_layout = QHBoxLayout()
        output_label = QLabel("Output Directory:")
        output_label.setMinimumWidth(120)
        self.output_path = QLineEdit()
        self.output_path.setPlaceholderText("Leave empty for default location")
        output_browse_button = QPushButton("Browse...")
        output_browse_button.setMaximumWidth(100)
        output_browse_button.clicked.connect(self.browse_output)
        output_layout.addWidget(output_label)
        output_layout.addWidget(self.output_path)
        output_layout.addWidget(output_browse_button)
        layout.addLayout(output_layout)
        
        self.batch_processing_checkbox = QCheckBox("Batch Processing (multiple episodes)")
        self.batch_processing_checkbox.setToolTip("Process multiple episodes from an input directory")
        layout.addWidget(self.batch_processing_checkbox)
        
        # Channel type
        channel_layout = QHBoxLayout()
        channel_label = QLabel("Channel Type:")
        channel_label.setMinimumWidth(120)
        self.channel_combo = QComboBox()
        self.channel_combo.addItems([
            "gaming", "anime", "superhero", "manga", "marvel_dc", "original_manga"
        ])
        self.channel_combo.currentTextChanged.connect(self.on_channel_changed)
        channel_layout.addWidget(channel_label)
        channel_layout.addWidget(self.channel_combo)
        layout.addLayout(channel_layout)
        
        # Base model selection
        base_model_layout = QHBoxLayout()
        base_model_label = QLabel("Base Model:")
        base_model_label.setMinimumWidth(120)
        self.base_model_combo = QComboBox()
        base_model_layout.addWidget(base_model_label)
        base_model_layout.addWidget(self.base_model_combo)
        layout.addLayout(base_model_layout)
        
        lora_group = QGroupBox("LoRA Style Selection (up to 5)")
        lora_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        lora_layout = QVBoxLayout()
        
        self.lora_selection_widget = MultiLoRASelectionWidget(
            channel_type=self.channel_combo.currentText()
        )
        lora_layout.addWidget(self.lora_selection_widget)
        
        lora_group.setLayout(lora_layout)
        layout.addWidget(lora_group)
        
        from .character_selection_widget import CharacterImageSelectionWidget
        char_group = QGroupBox("Character Management")
        char_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        char_layout = QVBoxLayout()
        
        self.character_selection_widget = CharacterImageSelectionWidget()
        char_layout.addWidget(self.character_selection_widget)
        
        char_group.setLayout(char_layout)
        layout.addWidget(char_group)
        
        # Multi-language selection
        language_group = QGroupBox("Multi-Language Voice Generation")
        language_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        language_layout = QVBoxLayout()
        
        self.language_selection_widget = MultiLanguageSelectionWidget()
        language_layout.addWidget(self.language_selection_widget)
        
        language_group.setLayout(language_layout)
        layout.addWidget(language_group)
        
        # Video format selection
        format_layout = QHBoxLayout()
        format_label = QLabel("Video Format:")
        format_label.setMinimumWidth(120)
        self.format_combo = QComboBox()
        self.format_combo.addItems(["mp4", "webm", "mov", "avi"])
        format_layout.addWidget(format_label)
        format_layout.addWidget(self.format_combo)
        layout.addLayout(format_layout)
        
        fps_group = QGroupBox("FPS Rendering & Frame Interpolation")
        fps_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        fps_layout = QVBoxLayout()
        
        # Render FPS selection
        render_fps_layout = QHBoxLayout()
        render_fps_label = QLabel("Render FPS:")
        render_fps_label.setMinimumWidth(120)
        self.render_fps_combo = QComboBox()
        self.render_fps_combo.addItems(["12", "15", "20", "24", "30"])
        self.render_fps_combo.setCurrentText("24")
        render_fps_layout.addWidget(render_fps_label)
        render_fps_layout.addWidget(self.render_fps_combo)
        
        # Output FPS selection
        output_fps_layout = QHBoxLayout()
        output_fps_label = QLabel("Output FPS:")
        output_fps_label.setMinimumWidth(120)
        self.output_fps_combo = QComboBox()
        self.output_fps_combo.addItems(["24", "30", "48", "60"])
        self.output_fps_combo.setCurrentText("24")
        output_fps_layout.addWidget(output_fps_label)
        output_fps_layout.addWidget(self.output_fps_combo)
        
        self.frame_interpolation_enabled = QCheckBox("Enable AI Frame Interpolation")
        self.frame_interpolation_enabled.setChecked(True)
        self.frame_interpolation_enabled.setToolTip("Use AI to generate intermediate frames for smoother motion")
        
        # Connect validation
        self.render_fps_combo.currentTextChanged.connect(self.validate_fps_settings)
        self.output_fps_combo.currentTextChanged.connect(self.validate_fps_settings)
        
        fps_layout.addLayout(render_fps_layout)
        fps_layout.addLayout(output_fps_layout)
        fps_layout.addWidget(self.frame_interpolation_enabled)
        
        fps_group.setLayout(fps_layout)
        layout.addWidget(fps_group)
        
        upscaler_group = QGroupBox("Video Upscaling")
        upscaler_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        upscaler_layout = QVBoxLayout()
        
        self.upscale_enabled = QCheckBox("Enable video upscaling")
        self.upscale_enabled.setChecked(True)
        
        resolution_layout = QHBoxLayout()
        resolution_label = QLabel("Target Resolution:")
        resolution_label.setMinimumWidth(120)
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["720p", "1080p", "1440p", "4k"])
        self.resolution_combo.setCurrentText("1080p")
        
        resolution_layout.addWidget(resolution_label)
        resolution_layout.addWidget(self.resolution_combo)
        
        upscaler_layout.addWidget(self.upscale_enabled)
        upscaler_layout.addLayout(resolution_layout)
        
        upscaler_group.setLayout(upscaler_layout)
        layout.addWidget(upscaler_group)
        
        advanced_group = QGroupBox("Advanced Generation Settings")
        advanced_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        advanced_layout = QVBoxLayout()
        
        combat_layout = QHBoxLayout()
        self.combat_enabled = QCheckBox("Enable Enhanced Combat Scenes")
        self.combat_enabled.setChecked(True)
        self.combat_enabled.setToolTip("Generate dynamic choreography and camera effects for combat scenes")
        combat_layout.addWidget(self.combat_enabled)
        
        combat_difficulty_label = QLabel("Combat Difficulty:")
        combat_difficulty_label.setMinimumWidth(120)
        self.combat_difficulty_combo = QComboBox()
        self.combat_difficulty_combo.addItems(["easy", "medium", "hard", "epic"])
        self.combat_difficulty_combo.setCurrentText("medium")
        combat_layout.addWidget(combat_difficulty_label)
        combat_layout.addWidget(self.combat_difficulty_combo)
        
        advanced_layout.addLayout(combat_layout)
        
        script_layout = QHBoxLayout()
        self.script_expansion_enabled = QCheckBox("Auto-Expand Short Scripts")
        self.script_expansion_enabled.setChecked(True)
        self.script_expansion_enabled.setToolTip("Automatically expand scripts to minimum 20 minutes using LLM")
        script_layout.addWidget(self.script_expansion_enabled)
        
        duration_label = QLabel("Min Duration (minutes):")
        duration_label.setMinimumWidth(120)
        self.min_duration_spin = QSpinBox()
        self.min_duration_spin.setRange(5, 60)
        self.min_duration_spin.setValue(20)
        script_layout.addWidget(duration_label)
        script_layout.addWidget(self.min_duration_spin)
        
        advanced_layout.addLayout(script_layout)
        
        llm_layout = QHBoxLayout()
        self.llm_override_enabled = QCheckBox("Manual LLM Override")
        self.llm_override_enabled.setToolTip("Override automatic LLM model selection")
        llm_layout.addWidget(self.llm_override_enabled)
        
        llm_model_label = QLabel("LLM Model:")
        llm_model_label.setMinimumWidth(120)
        self.llm_model_combo = QComboBox()
        self.llm_model_combo.addItems(["auto", "deepseek-llama-8b", "deepseek-r1", "phi-2"])
        llm_layout.addWidget(llm_model_label)
        llm_layout.addWidget(self.llm_model_combo)
        
        advanced_layout.addLayout(llm_layout)
        
        advanced_group.setLayout(advanced_layout)
        layout.addWidget(advanced_group)
        
        template_layout = QHBoxLayout()
        template_label = QLabel("Input Template:")
        template_label.setMinimumWidth(120)
        self.template_button = QPushButton("Show Template")
        template_layout.addWidget(template_label)
        template_layout.addWidget(self.template_button)
        self.template_button.clicked.connect(self.show_template)
        layout.addLayout(template_layout)
        
        self.warning_label = QLabel("⚠️ Warning: LoRAs require a base model to function properly!")
        self.warning_label.setStyleSheet(get_warning_style())
        self.warning_label.setVisible(False)
        layout.addWidget(self.warning_label)
        
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(scroll_area)
        
        button_layout = QHBoxLayout()
        cancel_button = QPushButton("Cancel")
        create_button = QPushButton("Create")
        cancel_button.clicked.connect(self.reject)
        create_button.clicked.connect(self.accept)
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(create_button)
        main_layout.addLayout(button_layout)
    
    def browse_input(self):
        if hasattr(self, 'batch_processing_checkbox') and self.batch_processing_checkbox.isChecked():
            path = QFileDialog.getExistingDirectory(self, "Select Input Directory for Batch Processing")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select Input File")
        
        if path:
            self.input_path.setText(path)
    
    def browse_output(self):
        """Browse for output directory."""
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.output_path.setText(path)
            if hasattr(self, 'character_selection_widget'):
                self.character_selection_widget.set_project_directory(path)
    
    def on_channel_changed(self):
        """Update available base models and LoRAs when channel changes."""
        self.load_base_models()
        
        # Update the LoRA selection widget with the new channel type
        channel_type = self.channel_combo.currentText()
        self.lora_selection_widget.set_channel_type(channel_type)
        
        self.validate_selection()
    
    def load_base_models(self):
        """Load base models compatible with selected channel."""
        try:
            print(f"🔍 DEBUG: Loading base models for channel: {self.channel_combo.currentText()}")
            print(f"🔍 DEBUG: Making API request to: {self.api_url}/models")
            
            response = requests.get(f"{self.api_url}/models", timeout=10)
            print(f"🔍 DEBUG: API response status: {response.status_code}")
            
            if response.status_code == 200:
                models_data = response.json()
                print(f"🔍 DEBUG: API returned {len(models_data.get('models', []))} total models")
                
                self.base_model_combo.clear()
                
                channel_type = self.channel_combo.currentText()
                
                optimal_model = None
                compatible_models = []
                all_base_models = []
                
                for model in models_data.get("models", []):
                    if model.get("model_type") == "base":
                        all_base_models.append({
                            "name": model.get("name"),
                            "downloaded": model.get("downloaded", False),
                            "compatibility": model.get("channel_compatibility", [])
                        })
                        
                        if model.get("downloaded", False):
                            channel_compatibility = model.get("channel_compatibility", [])
                            if isinstance(channel_compatibility, str):
                                channel_compatibility = [c.strip() for c in channel_compatibility.split(",") if c.strip()]
                            
                            if not channel_compatibility or channel_type in channel_compatibility:
                                model_name = model["name"]
                                model_description = model.get("description", "")
                                compatible_models.append(model_name)
                                
                                if model_description and "Best for:" in model_description:
                                    best_for_part = model_description.split("Best for:")[1].split(".")[0].strip()
                                    display_text = f"{model_name} - {best_for_part}"
                                else:
                                    display_text = model_name
                                
                                if model_name == optimal_model:
                                    self.base_model_combo.addItem(f"{display_text} (Recommended for your GPU)")
                                else:
                                    self.base_model_combo.addItem(display_text)
                
                print(f"🔍 DEBUG: Found {len(all_base_models)} base models total:")
                for model in all_base_models:
                    print(f"  - {model['name']}: downloaded={model['downloaded']}, compatibility={model['compatibility']}")
                
                print(f"🔍 DEBUG: Found {len(compatible_models)} compatible models for {channel_type}: {compatible_models}")
                
                if len(compatible_models) == 0:
                    self.base_model_combo.addItem("No compatible base models downloaded")
                    print(f"❌ DEBUG: No compatible base models found for channel: {channel_type}")
                    print(f"❌ DEBUG: Available models: {[m.get('name') for m in models_data.get('models', []) if m.get('model_type') == 'base']}")
                else:
                    print(f"✅ DEBUG: Successfully loaded {len(compatible_models)} compatible base models for {channel_type}")
            else:
                print(f"❌ DEBUG: API request failed with status {response.status_code}")
                print(f"❌ DEBUG: Response text: {response.text}")
                raise Exception(f"API request failed: {response.status_code}")
                    
        except Exception as e:
            print(f"❌ DEBUG: Exception in load_base_models: {str(e)}")
            import traceback
            traceback.print_exc()
            
            self.base_model_combo.clear()
            channel_type = self.channel_combo.currentText()
            if channel_type in ["gaming", "superhero", "marvel_dc"]:
                self.base_model_combo.addItems(["stable_diffusion_1_5", "stable_diffusion_xl"])
            elif channel_type in ["anime", "manga", "original_manga"]:
                self.base_model_combo.addItems(["anythingv5", "counterfeitv3"])
    
    def load_lora_models(self):
        """
        Legacy method kept for backward compatibility.
        LoRA models are now loaded by the MultiLoRASelectionWidget.
        """
        pass
    
    def validate_fps_settings(self):
        """Validate that output FPS is a multiple of render FPS."""
        try:
            render_fps = int(self.render_fps_combo.currentText())
            output_fps = int(self.output_fps_combo.currentText())
            
            if output_fps % render_fps != 0:
                self.warning_label.setText("⚠️ Warning: Output FPS must be a multiple of Render FPS!")
                self.warning_label.setVisible(True)
                return False
            else:
                if not self.validate_selection():
                    return False
                self.warning_label.setVisible(False)
                return True
        except ValueError:
            return False
    
    def validate_selection(self):
        """Validate that a base model is selected and at least one LoRA is available."""
        base_model = self.base_model_combo.currentText()
        selected_loras = self.lora_selection_widget.get_selected_loras()
        
        if "No compatible" in base_model:
            self.warning_label.setText("⚠️ Warning: Please download a compatible base model first!")
            self.warning_label.setVisible(True)
            return False
        
        if len(selected_loras) == 0 and self.lora_selection_widget.available_list.count() == 0:
            self.warning_label.setText("⚠️ Warning: No compatible LoRA models available for this channel!")
            self.warning_label.setVisible(True)
            return False
        
        self.warning_label.setVisible(False)
        return True
    
    def accept(self):
        """Override accept to validate selection."""
        if self.validate_selection() and self.validate_fps_settings():
            super().accept()
    
    def get_project_data(self):
        base_model = self.base_model_combo.currentText()
        if "(" in base_model:
            base_model = base_model.split("(")[0].strip()
        
        selected_loras = self.lora_selection_widget.get_selected_loras()
        selected_languages = self.language_selection_widget.get_selected_languages()
        
        lora_model = None
        if selected_loras:
            lora_model = selected_loras[0]["lora_name"]
            
        return {
            "title": self.title_input.text(),
            "description": self.desc_input.toPlainText(),
            "input_path": self.input_path.text(),
            "output_path": self.output_path.text() if self.output_path.text().strip() else None,
            "base_model": base_model,
            "lora_model": lora_model,
            "loras": selected_loras,
            "languages": selected_languages,
            "channel_type": self.channel_combo.currentText(),
            "video_format": self.format_combo.currentText(),
            "upscale_enabled": self.upscale_enabled.isChecked(),
            "target_resolution": self.resolution_combo.currentText(),
            "batch_processing": self.batch_processing_checkbox.isChecked(),
            "combat_enabled": self.combat_enabled.isChecked(),
            "combat_difficulty": self.combat_difficulty_combo.currentText(),
            "script_expansion_enabled": self.script_expansion_enabled.isChecked(),
            "min_duration_minutes": self.min_duration_spin.value(),
            "llm_override_enabled": self.llm_override_enabled.isChecked(),
            "llm_model": self.llm_model_combo.currentText(),
            "render_fps": int(self.render_fps_combo.currentText()),
            "output_fps": int(self.output_fps_combo.currentText()),
            "frame_interpolation_enabled": self.frame_interpolation_enabled.isChecked(),
            "character_data": self.character_selection_widget.get_character_data() if hasattr(self, 'character_selection_widget') else {}
        }
        
    def show_template(self):
        """Show the input template for the selected channel type."""
        from backend.templates import get_project_template
        
        channel_type = self.channel_combo.currentText()
        template = get_project_template(channel_type)
        
        if template:
            template_dialog = TemplateDialog(template, self)
            template_dialog.exec()

class TemplateDialog(QDialog):
    def __init__(self, template, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle(f"Template: {template['name']}")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)
        self.setStyleSheet(get_app_stylesheet())
        
        layout = QVBoxLayout(self)
        
        # Description
        desc_label = QLabel(template['description'])
        desc_label.setStyleSheet(get_header_style())
        layout.addWidget(desc_label)
        
        instructions_label = QLabel("Instructions:")
        instructions_label.setStyleSheet(get_subheader_style())
        layout.addWidget(instructions_label)
        
        for instruction in template['instructions']:
            inst_label = QLabel(f"• {instruction}")
            inst_label.setWordWrap(True)
            layout.addWidget(inst_label)
        
        example_label = QLabel("Example:")
        example_label.setStyleSheet(get_subheader_style())
        layout.addWidget(example_label)
        
        example_text = QTextEdit()
        example_text.setReadOnly(True)
        
        if 'example_yaml' in template:
            example_text.setPlainText(template['example_yaml'])
        elif 'example_json' in template:
            import json
            example_text.setPlainText(json.dumps(template['example_json'], indent=2))
        elif 'example_txt' in template:
            example_text.setPlainText(template['example_txt'])
        
        layout.addWidget(example_text)
        
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)
