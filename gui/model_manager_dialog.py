import requests
from .qt_compat import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QComboBox, QMessageBox, QProgressBar, QWidget, QLineEdit,
    QGroupBox, Qt, QThread, Signal, QTimer, QTabWidget, QDesktopServices, QUrl,
    QFileDialog
)

from config import API_HOST, API_PORT
from .styles import (
    get_app_stylesheet, get_header_style, get_subheader_style,
    get_warning_style, get_success_style, get_error_style, get_info_style
)
import logging
logger = logging.getLogger(__name__)

class DownloadThread(QThread):
    progress_signal = Signal(int)
    finished_signal = Signal(bool, str)
    
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.api_url = f"http://{API_HOST}:{API_PORT}"
    
    def run(self):
        try:
            token = None
            try:
                token_response = requests.get(f"{self.api_url}/settings/huggingface_token")
                if token_response.status_code == 200:
                    token_data = token_response.json()
                    token = token_data.get("value")
            except Exception as e:
                print(f"Error getting HuggingFace token: {str(e)}")
            
            self.progress_signal.emit(5)
            
            # Call API to download model with token
            response = requests.post(
                f"{self.api_url}/models/download",
                json={"name": self.model_name, "token": token}
            )
            
            self.progress_signal.emit(100)
            
            if response.status_code == 200:
                self.finished_signal.emit(True, "Download complete")
            else:
                self.finished_signal.emit(False, f"Error: {response.text}")
        except Exception as e:
            self.finished_signal.emit(False, f"Error: {str(e)}")

class ModelManagerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.api_url = f"http://{API_HOST}:{API_PORT}"
        
        self.setWindowTitle("AI Model Manager")
        self.setMinimumSize(700, 600)
        
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.auto_check_updates)
        self.update_timer.start(3600000)  # Check once per hour (3600000 ms)
        
        self.init_ui()
        self.load_models()
        self.load_hf_token()
        self.load_civitai_token()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        self.setStyleSheet(get_app_stylesheet())
        
        header_label = QLabel("AI Model Manager")
        header_label.setStyleSheet(get_header_style())
        layout.addWidget(header_label)
        
        token_group = QWidget()
        token_layout = QVBoxLayout(token_group)
        token_label = QLabel("HuggingFace API Token:")
        token_label.setStyleSheet(get_subheader_style())
        
        token_input_layout = QHBoxLayout()
        self.token_input = QLineEdit()
        self.token_input.setPlaceholderText("Enter your HuggingFace API token (optional)")
        self.token_input.setEchoMode(QLineEdit.EchoMode.Password)
        
        self.save_token_button = QPushButton("Save Token")
        self.save_token_button.clicked.connect(self.save_hf_token)
        
        token_input_layout.addWidget(self.token_input)
        token_input_layout.addWidget(self.save_token_button)
        
        token_layout.addWidget(token_label)
        token_layout.addLayout(token_input_layout)
        layout.addWidget(token_group)
        
        civitai_group = QGroupBox("Civitai API Key")
        civitai_layout = QVBoxLayout()
        
        civitai_label = QLabel("Civitai API Key:")
        civitai_label.setStyleSheet("font-weight: bold; color: #333;")
        
        civitai_input_layout = QHBoxLayout()
        self.civitai_token_input = QLineEdit()
        self.civitai_token_input.setPlaceholderText("Enter your Civitai API key for authenticated downloads")
        self.civitai_token_input.setEchoMode(QLineEdit.EchoMode.Password)
        
        self.civitai_save_button = QPushButton("Save Civitai Key")
        self.civitai_save_button.clicked.connect(self.save_civitai_token)
        
        civitai_input_layout.addWidget(self.civitai_token_input)
        civitai_input_layout.addWidget(self.civitai_save_button)
        
        civitai_layout.addWidget(civitai_label)
        civitai_layout.addLayout(civitai_input_layout)
        civitai_group.setLayout(civitai_layout)
        layout.addWidget(civitai_group)
        
        self.model_tabs = QTabWidget()
        
        # Base models tab
        base_tab = QWidget()
        base_layout = QVBoxLayout(base_tab)
        base_models_label = QLabel("Base AI Models")
        base_models_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #2980b9;")
        self.base_models_list = QListWidget()
        
        base_download_layout = QHBoxLayout()
        base_download_label = QLabel("Download Base Model:")
        self.base_model_combo = QComboBox()
        self.base_download_button = QPushButton("Download Base Model")
        self.base_download_button.clicked.connect(lambda: self.download_model("base"))
        
        base_download_layout.addWidget(base_download_label)
        base_download_layout.addWidget(self.base_model_combo)
        base_download_layout.addWidget(self.base_download_button)
        
        self.base_delete_button = QPushButton("Delete Selected Model")
        self.base_delete_button.clicked.connect(lambda: self.delete_selected_model("base"))
        
        base_management_layout = QHBoxLayout()
        base_management_layout.addLayout(base_download_layout)
        base_management_layout.addWidget(self.base_delete_button)
        
        base_layout.addWidget(base_models_label)
        base_layout.addWidget(self.base_models_list)
        base_layout.addLayout(base_management_layout)
        
        # LoRA models tab
        lora_tab = QWidget()
        lora_layout = QVBoxLayout(lora_tab)
        lora_models_label = QLabel("LoRA Style Adaptations")
        lora_models_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #FF9800;")
        self.lora_models_list = QListWidget()
        
        lora_download_layout = QHBoxLayout()
        lora_download_label = QLabel("Download LoRA:")
        self.lora_model_combo = QComboBox()
        self.lora_download_button = QPushButton("Download LoRA")
        self.lora_download_button.clicked.connect(lambda: self.download_model("lora"))
        
        lora_download_layout.addWidget(lora_download_label)
        lora_download_layout.addWidget(self.lora_model_combo)
        lora_download_layout.addWidget(self.lora_download_button)
        
        self.lora_delete_button = QPushButton("Delete Selected Model")
        self.lora_delete_button.clicked.connect(lambda: self.delete_selected_model("lora"))
        
        lora_management_layout = QHBoxLayout()
        lora_management_layout.addLayout(lora_download_layout)
        lora_management_layout.addWidget(self.lora_delete_button)
        
        lora_warning_label = QLabel("⚠️ LoRAs require a base model to function properly!")
        lora_warning_label.setStyleSheet("color: #FF5722; font-weight: bold; padding: 10px; background-color: #FFF3E0; border-radius: 4px; margin: 10px 0;")
        
        lora_layout.addWidget(lora_models_label)
        lora_layout.addWidget(self.lora_models_list)
        lora_layout.addWidget(lora_warning_label)
        lora_layout.addLayout(lora_management_layout)
        
        audio_tab = QWidget()
        audio_layout = QVBoxLayout(audio_tab)
        audio_models_label = QLabel("Audio Processing Models")
        audio_models_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #9b59b6;")
        self.audio_models_list = QListWidget()
        
        audio_download_layout = QHBoxLayout()
        audio_download_label = QLabel("Download Audio Model:")
        self.audio_model_combo = QComboBox()
        self.audio_download_button = QPushButton("Download Audio Model")
        self.audio_download_button.clicked.connect(lambda: self.download_model("audio"))
        
        audio_download_layout.addWidget(audio_download_label)
        audio_download_layout.addWidget(self.audio_model_combo)
        audio_download_layout.addWidget(self.audio_download_button)
        
        self.audio_delete_button = QPushButton("Delete Selected Model")
        self.audio_delete_button.clicked.connect(lambda: self.delete_selected_model("audio"))
        
        audio_management_layout = QHBoxLayout()
        audio_management_layout.addLayout(audio_download_layout)
        audio_management_layout.addWidget(self.audio_delete_button)
        
        audio_layout.addWidget(audio_models_label)
        audio_layout.addWidget(self.audio_models_list)
        audio_layout.addLayout(audio_management_layout)
        
        video_tab = QWidget()
        video_layout = QVBoxLayout(video_tab)
        video_models_label = QLabel("Video Processing Models")
        video_models_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #e74c3c;")
        self.video_models_list = QListWidget()
        
        video_download_layout = QHBoxLayout()
        video_download_label = QLabel("Download Video Model:")
        self.video_model_combo = QComboBox()
        self.video_download_button = QPushButton("Download Video Model")
        self.video_download_button.clicked.connect(lambda: self.download_model("video"))
        
        video_download_layout.addWidget(video_download_label)
        video_download_layout.addWidget(self.video_model_combo)
        video_download_layout.addWidget(self.video_download_button)
        
        self.video_delete_button = QPushButton("Delete Selected Model")
        self.video_delete_button.clicked.connect(lambda: self.delete_selected_model("video"))
        
        video_management_layout = QHBoxLayout()
        video_management_layout.addLayout(video_download_layout)
        video_management_layout.addWidget(self.video_delete_button)
        
        video_layout.addWidget(video_models_label)
        video_layout.addWidget(self.video_models_list)
        video_layout.addLayout(video_management_layout)
        
        text_tab = QWidget()
        text_layout = QVBoxLayout(text_tab)
        text_models_label = QLabel("Text Processing Models")
        text_models_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #27ae60;")
        self.text_models_list = QListWidget()
        
        text_download_layout = QHBoxLayout()
        text_download_label = QLabel("Download Text Model:")
        self.text_model_combo = QComboBox()
        self.text_download_button = QPushButton("Download Text Model")
        self.text_download_button.clicked.connect(lambda: self.download_model("text"))
        
        text_download_layout.addWidget(text_download_label)
        text_download_layout.addWidget(self.text_model_combo)
        text_download_layout.addWidget(self.text_download_button)
        
        self.text_delete_button = QPushButton("Delete Selected Model")
        self.text_delete_button.clicked.connect(lambda: self.delete_selected_model("text"))
        
        text_management_layout = QHBoxLayout()
        text_management_layout.addLayout(text_download_layout)
        text_management_layout.addWidget(self.text_delete_button)
        
        text_layout.addWidget(text_models_label)
        text_layout.addWidget(self.text_models_list)
        text_layout.addLayout(text_management_layout)
        
        self.model_tabs.addTab(base_tab, "Base Models")
        self.model_tabs.addTab(lora_tab, "LoRA Styles")
        self.model_tabs.addTab(audio_tab, "Audio Models")
        self.model_tabs.addTab(video_tab, "Video Models")
        editing_tab = QWidget()
        editing_layout = QVBoxLayout(editing_tab)
        editing_models_label = QLabel("Video Editing Models")
        editing_models_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #795548;")
        self.editing_models_list = QListWidget()
        
        editing_download_layout = QHBoxLayout()
        editing_download_label = QLabel("Download Editing Model:")
        self.editing_model_combo = QComboBox()
        self.editing_download_button = QPushButton("Download Editing Model")
        self.editing_download_button.clicked.connect(lambda: self.download_model("editing"))
        
        editing_download_layout.addWidget(editing_download_label)
        editing_download_layout.addWidget(self.editing_model_combo)
        editing_download_layout.addWidget(self.editing_download_button)
        
        self.editing_delete_button = QPushButton("Delete Selected Model")
        self.editing_delete_button.clicked.connect(lambda: self.delete_selected_model("editing"))
        
        editing_management_layout = QHBoxLayout()
        editing_management_layout.addLayout(editing_download_layout)
        editing_management_layout.addWidget(self.editing_delete_button)
        
        editing_layout.addWidget(editing_models_label)
        editing_layout.addWidget(self.editing_models_list)
        editing_layout.addLayout(editing_management_layout)
        
        self.model_tabs.addTab(base_tab, "Base Models")
        self.model_tabs.addTab(lora_tab, "LoRA Models")
        self.model_tabs.addTab(audio_tab, "Audio Models")
        self.model_tabs.addTab(video_tab, "Video Models")
        self.model_tabs.addTab(text_tab, "Text/LLM Models")
        self.model_tabs.addTab(editing_tab, "Editing Models")
        
        llm_tab = QWidget()
        llm_layout = QVBoxLayout(llm_tab)
        llm_models_label = QLabel("Large Language Models")
        llm_models_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #8e44ad;")
        self.llm_models_list = QListWidget()
        
        llm_download_layout = QHBoxLayout()
        llm_download_label = QLabel("Download LLM Model:")
        self.llm_model_combo = QComboBox()
        self.llm_download_button = QPushButton("Download LLM Model")
        self.llm_download_button.clicked.connect(lambda: self.download_model("llm"))
        
        llm_download_layout.addWidget(llm_download_label)
        llm_download_layout.addWidget(self.llm_model_combo)
        llm_download_layout.addWidget(self.llm_download_button)
        
        self.llm_delete_button = QPushButton("Delete Selected Model")
        self.llm_delete_button.clicked.connect(lambda: self.delete_selected_model("llm"))
        
        llm_management_layout = QHBoxLayout()
        llm_management_layout.addLayout(llm_download_layout)
        llm_management_layout.addWidget(self.llm_delete_button)
        
        llm_layout.addWidget(llm_models_label)
        llm_layout.addWidget(self.llm_models_list)
        llm_layout.addLayout(llm_management_layout)
        
        self.model_tabs.addTab(llm_tab, "LLM Models")
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        version_check_group = QGroupBox("Model Version Checking")
        version_check_layout = QVBoxLayout(version_check_group)
        
        check_button_layout = QHBoxLayout()
        self.check_updates_button = QPushButton("Check for Model Updates")
        self.check_updates_button.clicked.connect(self.check_for_updates)
        check_button_layout.addWidget(self.check_updates_button)
        
        self.version_status_label = QLabel("No updates checked yet")
        self.version_status_label.setStyleSheet("color: #666; font-style: italic;")
        
        version_check_layout.addLayout(check_button_layout)
        version_check_layout.addWidget(self.version_status_label)
        
        filter_group = QGroupBox("Model Filters")
        filter_layout = QHBoxLayout(filter_group)
        
        filter_type_label = QLabel("Filter Type:")
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All", "Downloaded", "Available"])
        self.filter_combo.currentTextChanged.connect(self.load_models)
        
        vram_filter_label = QLabel("VRAM Requirement:")
        self.vram_filter_combo = QComboBox()
        self.vram_filter_combo.addItems(["All", "Low (4-8GB)", "Medium (8-16GB)", "High (16-24GB)", "Ultra (24GB+)"])
        self.vram_filter_combo.currentTextChanged.connect(self.load_models)
        
        filter_layout.addWidget(filter_type_label)
        filter_layout.addWidget(self.filter_combo)
        filter_layout.addWidget(vram_filter_label)
        filter_layout.addWidget(self.vram_filter_combo)
        
        # Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        
        layout.addWidget(filter_group)
        layout.addWidget(self.model_tabs)
        layout.addWidget(version_check_group)
        layout.addWidget(self.progress_bar)
        layout.addWidget(close_button)
    
    def load_models(self):
        try:
            # Call API to get available models
            response = requests.get(f"{self.api_url}/models")
            if response.status_code == 200:
                models_data = response.json()
                
                self.base_models_list.clear()
                self.lora_models_list.clear()
                self.audio_models_list.clear()
                self.video_models_list.clear()
                self.text_models_list.clear()
                self.editing_models_list.clear()
                
                self.base_model_combo.clear()
                self.lora_model_combo.clear()
                self.audio_model_combo.clear()
                self.video_model_combo.clear()
                self.text_model_combo.clear()
                self.editing_model_combo.clear()
                self.llm_model_combo.clear()
                
                model_categories = {
                    "base": {"list": self.base_models_list, "combo": self.base_model_combo, "available": []},
                    "lora": {"list": self.lora_models_list, "combo": self.lora_model_combo, "available": []},
                    "audio": {"list": self.audio_models_list, "combo": self.audio_model_combo, "available": []},
                    "video": {"list": self.video_models_list, "combo": self.video_model_combo, "available": []},
                    "text": {"list": self.text_models_list, "combo": self.text_model_combo, "available": []},
                    "editing": {"list": self.editing_models_list, "combo": self.editing_model_combo, "available": []},
                    "llm": {"list": self.llm_models_list, "combo": self.llm_model_combo, "available": []}
                }
                
                filter_type = getattr(self, 'filter_combo', None)
                filter_type = filter_type.currentText() if filter_type else "All"
                vram_filter = getattr(self, 'vram_filter_combo', None)
                vram_filter = vram_filter.currentText() if vram_filter else "All"
                
                for model in models_data.get("models", []):
                    model_type = model.get("model_type")
                    
                    if vram_filter != "All":
                        vram_mapping = {
                            "Low (4-8GB)": "low",
                            "Medium (8-16GB)": "medium", 
                            "High (16-24GB)": "high",
                            "Ultra (24GB+)": "ultra"
                        }
                        target_vram = vram_mapping.get(vram_filter)
                        if target_vram and model.get("vram_requirement", "medium") != target_vram:
                            continue
                    
                    if filter_type == "Downloaded":
                        if not model.get("downloaded", False):
                            continue
                    elif filter_type == "Available":
                        if model.get("downloaded", False):
                            continue
                    
                    if model_type in model_categories:
                        vram_req = model.get("vram_requirement", "medium").upper()
                        size_info = f" [{model.get('size_mb', 0)//1024}GB, {vram_req} VRAM]" if model.get('size_mb') else f" [{vram_req} VRAM]"
                        
                        if model.get("downloaded", False):
                            model_categories[model_type]["list"].addItem(
                                f"{model['name']} (v{model['version']}){size_info} - Downloaded"
                            )
                        else:
                            model_categories[model_type]["available"].append(model["name"])
                
                # Populate combo boxes with available models
                for category, data in model_categories.items():
                    data["combo"].addItems(data["available"])
            
            else:
                self.base_models_list.clear()
                self.lora_models_list.clear()
                self.audio_models_list.clear()
                self.video_models_list.clear()
                self.text_models_list.clear()
                self.editing_models_list.clear()
                
                fallback_data = {
                    "base": {
                        "list": self.base_models_list,
                        "combo": self.base_model_combo,
                        "items": [
                            "stable_diffusion_1_5 (v1.5) - Available for download",
                            "stable_diffusion_xl (v1.0) - Available for download", 
                            "anythingv5 (latest) - Available for download",
                            "counterfeitv3 (v3.0) - Available for download",
                            "realisticvision (v6.0) - Available for download"
                        ],
                        "models": [
                            "stable_diffusion_1_5", "stable_diffusion_xl", 
                            "anythingv5", "counterfeitv3", "realisticvision"
                        ]
                    },
                    "lora": {
                        "list": self.lora_models_list,
                        "combo": self.lora_model_combo,
                        "items": [
                            "RealisticVisionV6_LoRA (v6.0) - Available for download",
                            "EpicPhotographicLoRA (v2.0) - Available for download",
                            "Cinematic-Lighting-LoRA (v2.0) - Available for download",
                            "Flat2D-AnimeLoRA (v2.0) - Available for download",
                            "AnimeStyleLoRA (v2.0) - Available for download",
                            "CuteAnimeGirlLoRA (v2.0) - Available for download",
                            "MangaStyle_LoRA (v2.0) - Available for download",
                            "BlackAndWhiteMangaLoRA (v2.0) - Available for download",
                            "MangaLineArtLoRA (v2.0) - Available for download",
                            "SuperheroStyleLoRA (v2.0) - Available for download",
                            "HeroicPoseLoRA (v2.0) - Available for download",
                            "ActionHeroLoRA (v2.0) - Available for download",
                            "MarvelStyleLoRA (v2.0) - Available for download",
                            "DCComicsStyleLoRA (v2.0) - Available for download",
                            "ComicBookStyleLoRA (v2.0) - Available for download",
                            "OriginalMangaStyleLoRA (v2.0) - Available for download",
                            "JapaneseMangaLoRA (v2.0) - Available for download",
                            "MangaCharacterLoRA (v2.0) - Available for download"
                        ],
                        "models": [
                            "RealisticVisionV6_LoRA", "EpicPhotographicLoRA", "Cinematic-Lighting-LoRA",
                            "Flat2D-AnimeLoRA", "AnimeStyleLoRA", "CuteAnimeGirlLoRA",
                            "MangaStyle_LoRA", "BlackAndWhiteMangaLoRA", "MangaLineArtLoRA",
                            "SuperheroStyleLoRA", "HeroicPoseLoRA", "ActionHeroLoRA",
                            "MarvelStyleLoRA", "DCComicsStyleLoRA", "ComicBookStyleLoRA",
                            "OriginalMangaStyleLoRA", "JapaneseMangaLoRA", "MangaCharacterLoRA"
                        ]
                    },
                    "audio": {
                        "list": self.audio_models_list,
                        "combo": self.audio_model_combo,
                        "items": [
                            "whisper (large-v3) - Available for download",
                            "bark (latest) - Available for download",
                            "musicgen (large) - Available for download",
                            "rvc (v2.0) - Available for download"
                        ],
                        "models": ["whisper", "bark", "musicgen", "rvc"]
                    },
                    "video": {
                        "list": self.video_models_list,
                        "combo": self.video_model_combo,
                        "items": [
                            "svd_xt (Stable Video Diffusion XT) - 1024×576, 25 frames, 16-24GB VRAM",
                            "zeroscope_v2_xl (Zeroscope v2 XL) - 1024×576, 24 frames, 12-16GB VRAM",
                            "animatediff_v2_sdxl (AnimateDiff v2 SDXL) - 1024×1024, 16 frames, 13-16GB VRAM",
                            "animatediff_lightning (AnimateDiff Lightning) - 512×512, 16 frames, 8-12GB VRAM",
                            "modelscope_t2v (ModelScope T2V) - 256×256, 16 frames, 8-12GB VRAM",
                            "ltx_video (LTX-Video) - 768×512, 120 frames, 24-48GB VRAM",
                            "skyreels_v2 (SkyReels V2) - 540p, unlimited frames, 24-48GB VRAM",
                            "sadtalker (SadTalker) - 512×512, 30 frames, 8-16GB VRAM",
                            "dreamtalk (DreamTalk) - 512×512, 30 frames, 8-16GB VRAM"
                        ],
                        "models": [
                            "svd_xt", "zeroscope_v2_xl", "animatediff_v2_sdxl", "animatediff_lightning",
                            "modelscope_t2v", "ltx_video", "skyreels_v2", "sadtalker", "dreamtalk"
                        ]
                    },
                    "text": {
                        "list": self.text_models_list,
                        "combo": self.text_model_combo,
                        "items": [
                            "deepseek_llama_8b_peft (Deepseek Llama 8B PEFT v5) - 16GB+ VRAM",
                            "deepseek_r1_distill (Deepseek R1 Distill) - 8-12GB VRAM",
                            "dialogpt_medium (DialoGPT Medium) - 4-6GB VRAM"
                        ],
                        "models": [
                            "deepseek_llama_8b_peft", "deepseek_r1_distill", "dialogpt_medium"
                        ]
                    },
                    "editing": {
                        "list": self.editing_models_list,
                        "combo": self.editing_model_combo,
                        "items": [
                            "scene_detection_v2 (Advanced Scene Detection) - Gaming optimized",
                            "highlight_extraction_gaming (Gaming Highlight Extractor) - Action detection",
                            "auto_editor_pro (Auto Editor Pro) - Smart cuts and transitions",
                            "shorts_generator_ai (AI Shorts Generator) - Viral content optimization",
                            "commentary_generator (AI Commentary) - Voice synthesis",
                            "upscaler_real_esrgan (Real-ESRGAN Upscaler) - 4x upscaling"
                        ],
                        "models": [
                            "scene_detection_v2", "highlight_extraction_gaming", "auto_editor_pro",
                            "shorts_generator_ai", "commentary_generator", "upscaler_real_esrgan"
                        ]
                    }
                }
                
                # Populate lists and combos with fallback data
                for category, data in fallback_data.items():
                    for item in data["items"]:
                        data["list"].addItem(item)
                    
                    data["combo"].clear()
                    data["combo"].addItems(data["models"])
                
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load models: {str(e)}")
            self.base_model_combo.addItems(["stable_diffusion_1_5", "anythingv5"])
            self.lora_model_combo.addItems(["anime_style_lora", "gaming_style_lora"])
            self.audio_model_combo.addItems(["whisper", "bark"])
            self.video_model_combo.addItems(["sadtalker"])
            self.text_model_combo.addItems(["llm"])
    
    def download_model(self, model_type="lora"):
        """
        Show download links for a model instead of automatically downloading.
        """
        model_combos = {
            "base": self.base_model_combo,
            "lora": self.lora_model_combo,
            "audio": self.audio_model_combo,
            "video": self.video_model_combo,
            "text": self.text_model_combo,
            "editing": self.editing_model_combo
        }
        
        if model_type not in model_combos:
            QMessageBox.warning(self, "Error", f"Unknown model type: {model_type}")
            return
            
        model_name = model_combos[model_type].currentText()
        if not model_name:
            QMessageBox.warning(self, "Error", f"No {model_type} model selected")
            return
        
        if model_type == "lora":
            self.show_lora_download_links(model_name)
        else:
            self.download_model_automatic(model_name, model_type)
    
    def download_model_automatic(self, model_name, model_type):
        """
        Download a model automatically (for non-LoRA models).
        """
        download_buttons = {
            "base": self.base_download_button,
            "lora": self.lora_download_button,
            "audio": self.audio_download_button,
            "video": self.video_download_button,
            "text": self.text_download_button,
            "editing": self.editing_download_button
        }
        
        download_buttons[model_type].setEnabled(False)
        
        # Show progress bar
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        
        # Create and start download thread
        self.download_thread = DownloadThread(model_name)
        self.download_thread.progress_signal.connect(self.update_progress)
        self.download_thread.finished_signal.connect(
            lambda success, message: self.download_finished(success, message, model_type)
        )
        self.download_thread.start()
    
    def show_lora_download_links(self, model_name):
        """Show a dialog with LoRA download links."""
        try:
            response = requests.get(f"{self.api_url}/models/{model_name}/download-link")
            if response.status_code == 200:
                link_data = response.json()
                
                dialog = QDialog(self)
                dialog.setWindowTitle(f"Download Links for {model_name}")
                dialog.setMinimumWidth(500)
                
                layout = QVBoxLayout(dialog)
                
                info_label = QLabel(f"Manual download required for: {model_name}")
                info_label.setStyleSheet("font-weight: bold; margin-bottom: 10px;")
                layout.addWidget(info_label)
                
                instructions_label = QLabel("1. Click a download link below\n2. Save the file to your models folder\n3. Use 'Import Downloaded LoRA' to add it")
                layout.addWidget(instructions_label)
                
                for i, url in enumerate(link_data['download_urls']):
                    link_button = QPushButton(f"Download Link {i+1}")
                    link_button.clicked.connect(lambda checked=False, u=url: self.open_download_url(u))
                    layout.addWidget(link_button)
                
                import_button = QPushButton("Import Downloaded LoRA File...")
                import_button.clicked.connect(lambda checked=False: self.import_lora_file(model_name))
                layout.addWidget(import_button)
                
                close_button = QPushButton("Close")
                close_button.clicked.connect(dialog.accept)
                layout.addWidget(close_button)
                
                dialog.exec()
            else:
                QMessageBox.warning(self, "Error", f"Failed to get download links: {response.text}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to get download links: {str(e)}")
    
    def open_download_url(self, url):
        """Open download URL in browser."""
        QDesktopServices.openUrl(QUrl(url))
    
    def import_lora_file(self, model_name):
        """Import a manually downloaded LoRA file with proper validation and error handling."""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, 
                f"Import LoRA file for {model_name}",
                "",
                "LoRA files (*.safetensors *.ckpt *.pt);;All files (*.*)"
            )
            
            if not file_path:
                return  # User cancelled
            
            from pathlib import Path
            import shutil
            import os
            
            if not os.path.exists(file_path):
                QMessageBox.critical(self, "Error", "Selected file does not exist!")
                return
                
            if not os.access(file_path, os.R_OK):
                QMessageBox.critical(self, "Error", "Cannot read the selected file!")
                return
            
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in ['.safetensors', '.ckpt', '.pt']:
                reply = QMessageBox.question(
                    self, "Unknown File Type", 
                    f"File extension '{file_ext}' is not a standard LoRA format. Continue anyway?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.No:
                    return
            
            models_dir = Path("models/loras")
            models_dir.mkdir(parents=True, exist_ok=True)
            
            original_name = Path(file_path).name
            safe_name = "".join(c for c in original_name if c.isalnum() or c in '._-')
            dest_path = models_dir / safe_name
            
            if dest_path.exists():
                reply = QMessageBox.question(
                    self, "File Exists", 
                    f"A file named '{safe_name}' already exists. Overwrite?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.No:
                    return
            
            try:
                shutil.copy2(file_path, dest_path)
            except Exception as e:
                QMessageBox.critical(self, "Copy Error", f"Failed to copy file: {str(e)}")
                return
            
            try:
                response = requests.post(
                    f"{self.api_url}/models/lora/register", 
                    json={"name": model_name, "path": str(dest_path)}
                )
                
                if response.status_code == 200:
                    QMessageBox.information(self, "Success", f"LoRA '{model_name}' imported successfully!")
                    self.load_models()  # Refresh the model list
                else:
                    QMessageBox.warning(self, "Registration Error", f"Failed to register LoRA: {response.text}")
                    try:
                        dest_path.unlink()
                    except:
                        pass
                        
            except requests.exceptions.RequestException as e:
                QMessageBox.critical(self, "API Error", f"Failed to connect to API: {str(e)}")
                try:
                    dest_path.unlink()
                except:
                    pass
                    
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to import LoRA: {str(e)}")
    
    def update_progress(self, value):
        self.progress_bar.setValue(value)
    
    def download_finished(self, success, message, model_type="lora"):
        """
        Handle download completion for the specified model type.
        
        Args:
            success: Whether the download was successful
            message: Message to display
            model_type: Type of model that was downloaded ("base", "lora", "audio", "video", or "text")
        """
        download_buttons = {
            "base": self.base_download_button,
            "lora": self.lora_download_button,
            "audio": self.audio_download_button,
            "video": self.video_download_button,
            "text": self.text_download_button,
            "editing": self.editing_download_button
        }
        
        if model_type in download_buttons:
            download_buttons[model_type].setEnabled(True)
        
        if success:
            QMessageBox.information(self, "Success", f"{model_type.title()} model download complete")
            self.load_models()
        else:
            QMessageBox.warning(self, "Error", message)
        
        self.progress_bar.setVisible(False)
    
    def save_hf_token(self):
        """Save HuggingFace API token to settings."""
        token = self.token_input.text().strip()
        if not token:
            QMessageBox.warning(self, "Error", "Please enter a valid API token")
            return
        
        try:
            response = requests.post(
                f"{self.api_url}/settings",
                params={"key": "huggingface_token", "value": token}
            )
            if response.status_code == 200:
                QMessageBox.information(self, "Success", "HuggingFace API token saved successfully")
            else:
                QMessageBox.warning(self, "Error", "Failed to save API token")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save API token: {str(e)}")
    
    def load_hf_token(self):
        """Load saved HuggingFace API token."""
        try:
            response = requests.get(f"{self.api_url}/settings/huggingface_token")
            if response.status_code == 200:
                data = response.json()
                self.token_input.setText(data.get("value", ""))
        except Exception:
            pass  # Token not set yet
            
    def save_civitai_token(self):
        """Save Civitai API token to settings."""
        token = self.civitai_token_input.text().strip()
        
        try:
            response = requests.post(
                f"{self.api_url}/settings",
                params={"key": "civitai_token", "value": token}
            )
            
            if response.status_code == 200:
                QMessageBox.information(self, "Success", "Civitai API key saved successfully!")
            else:
                QMessageBox.warning(self, "Error", f"Failed to save Civitai API key: {response.text}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saving Civitai API key: {str(e)}")
    
    def load_civitai_token(self):
        """Load saved Civitai API token."""
        try:
            response = requests.get(f"{self.api_url}/settings/civitai_token")
            if response.status_code == 200:
                data = response.json()
                self.civitai_token_input.setText(data.get("value", ""))
        except Exception:
            pass  # Token not set yet
    
    def check_for_updates(self):
        """Check for model updates from HuggingFace."""
        self.check_updates_button.setEnabled(False)
        self.version_status_label.setText("Checking for updates...")
        self.version_status_label.setStyleSheet("color: #1976D2; font-style: italic;")
        
        try:
            token = self.token_input.text().strip()
            
            # Call API to check for updates
            response = requests.get(
                f"{self.api_url}/models/check_updates",
                params={"token": token if token else None}
            )
            
            if response.status_code == 200:
                updates_data = response.json()
                base_updates = updates_data.get("base_models", [])
                lora_updates = updates_data.get("loras", [])
                
                total_updates = len(base_updates) + len(lora_updates)
                
                if total_updates > 0:
                    self.version_status_label.setText(f"Found {total_updates} model updates available!")
                    self.version_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
                    
                    update_details = "Available Updates:\n\n"
                    
                    if base_updates:
                        update_details += "Base Models:\n"
                        for update in base_updates:
                            update_details += f"• {update['name']}: {update['current_version']} → {update['latest_version']}\n"
                        update_details += "\n"
                    
                    if lora_updates:
                        update_details += "LoRA Models:\n"
                        for update in lora_updates:
                            update_details += f"• {update['name']}: {update['current_version']} → {update['latest_version']}\n"
                    
                    QMessageBox.information(self, "Model Updates Available", update_details)
                else:
                    self.version_status_label.setText("All models are up to date")
                    self.version_status_label.setStyleSheet("color: #4CAF50;")
            else:
                self.version_status_label.setText("Failed to check for updates")
                self.version_status_label.setStyleSheet("color: #F44336;")
                QMessageBox.warning(self, "Error", "Failed to check for model updates")
        
        except Exception as e:
            self.version_status_label.setText("Error checking for updates")
            self.version_status_label.setStyleSheet("color: #F44336;")
            QMessageBox.warning(self, "Error", f"Failed to check for model updates: {str(e)}")
        
        finally:
            self.check_updates_button.setEnabled(True)
    
    def auto_check_updates(self):
        """Automatically check for updates on a timer."""
        try:
            token = self.token_input.text().strip()
            
            # Call API to check for updates silently
            response = requests.get(
                f"{self.api_url}/models/check_updates",
                params={"token": token if token else None, "silent": True}
            )
            
            if response.status_code == 200:
                updates_data = response.json()
                base_updates = updates_data.get("base_models", [])
                lora_updates = updates_data.get("loras", [])
                
                total_updates = len(base_updates) + len(lora_updates)
                
                if total_updates > 0:
                    self.version_status_label.setText(f"{total_updates} model updates available")
                    self.version_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
                else:
                    self.version_status_label.setText("All models are up to date")
                    self.version_status_label.setStyleSheet("color: #4CAF50;")
        except Exception:
            pass
            
    def download_moviepy(self):
        """Download and install moviepy dependency."""
        try:
            import subprocess
            import sys
            
            reply = QMessageBox.question(
                self, "Install MoviePy",
                "MoviePy is required for video processing. Would you like to install it now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.progress_bar.setValue(0)
                self.progress_bar.setVisible(True)
                
                try:
                    result = subprocess.run(
                        [sys.executable, "-m", "pip", "install", "moviepy"],
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    
                    self.progress_bar.setValue(100)
                    QMessageBox.information(self, "Success", "MoviePy installed successfully!")
                    
                except subprocess.CalledProcessError as e:
                    QMessageBox.warning(self, "Error", f"Failed to install MoviePy: {e.stderr}")
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to install MoviePy: {str(e)}")
                finally:
                    self.progress_bar.setVisible(False)
                    
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to install MoviePy: {str(e)}")

    def delete_selected_model(self, model_type):
        """Delete the selected model of the specified type."""
        model_lists = {
            "base": self.base_models_list,
            "lora": self.lora_models_list,
            "audio": self.audio_models_list,
            "video": self.video_models_list,
            "text": self.text_models_list,
            "editing": self.editing_models_list
        }
        
        if model_type not in model_lists:
            QMessageBox.warning(self, "Error", f"Unknown model type: {model_type}")
            return
        
        model_list = model_lists[model_type]
        selected_items = model_list.selectedItems()
        
        if not selected_items:
            QMessageBox.warning(self, "No Selection", f"Please select a {model_type} model to delete")
            return
        
        item_text = selected_items[0].text()
        if "Downloaded" not in item_text:
            QMessageBox.warning(self, "Cannot Delete", "Only downloaded models can be deleted")
            return
        
        model_name = item_text.split(" (")[0]
        
        reply = QMessageBox.question(
            self, "Delete Model",
            f"Are you sure you want to delete the model '{model_name}'?\nThis will remove all downloaded files.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                response = requests.delete(f"{self.api_url}/models/{model_name}")
                if response.status_code == 200:
                    QMessageBox.information(self, "Success", f"Model '{model_name}' deleted successfully")
                    self.load_models()  # Refresh the model list
                else:
                    QMessageBox.warning(self, "Error", f"Failed to delete model: {response.text}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to delete model: {str(e)}")
    
    def show_video_models(self):
        """Display available video models."""
        self.model_list.clear()
        
        try:
            from backend.model_manager import get_available_models
            models = get_available_models()
            
            video_models = [m for m in models if m.get("model_type") == "video"]
            
            for model in video_models:
                item_text = f"{model['name']} - {model.get('description', 'Video model')}"
                if model.get('downloaded', False):
                    item_text += " [Downloaded]"
                else:
                    size_gb = model.get('size_mb', 0) / 1024
                    item_text += f" ({size_gb:.1f}GB)"
                
                item = QListWidgetItem(item_text)
                item.setData(Qt.ItemDataRole.UserRole, model)
                self.model_list.addItem(item)
                
        except Exception as e:
            logger.error(f"Error loading video models: {e}")
            item = QListWidgetItem(f"Error loading video models: {e}")
            self.model_list.addItem(item)
    
    def show_text_models(self):
        """Display available text/LLM models."""
        self.model_list.clear()
        
        try:
            from backend.model_manager import get_available_models
            
            models = get_available_models()
            model_manager = self._get_model_manager_fallback()
            current_vram_tier = model_manager._detect_vram_tier()
            
            text_models = [m for m in models if m.get("model_type") == "text"]
            
            for model in text_models:
                vram_req = model.get('vram_requirement', 'medium')
                compatible = self.is_vram_compatible(current_vram_tier, vram_req)
                
                item_text = f"{model['name']} - {model.get('description', 'Text model')}"
                if model.get('downloaded', False):
                    item_text += " [Downloaded]"
                else:
                    size_gb = model.get('size_mb', 0) / 1024
                    item_text += f" ({size_gb:.1f}GB)"
                
                if not compatible:
                    item_text += f" [Requires {vram_req} VRAM]"
                
                item = QListWidgetItem(item_text)
                item.setData(Qt.ItemDataRole.UserRole, model)
                
                if not compatible:
                    from .qt_compat import QColor
                    item.setForeground(QColor(128, 128, 128))
                
                self.model_list.addItem(item)
                
        except Exception as e:
            logger.error(f"Error loading text models: {e}")
            item = QListWidgetItem(f"Error loading text models: {e}")
            self.model_list.addItem(item)
    
    def show_editing_models(self):
        """Display available editing models."""
        self.model_list.clear()
        
        try:
            from backend.model_manager import get_available_models
            models = get_available_models()
            
            editing_models = [m for m in models if m.get("model_type") == "editing"]
            
            for model in editing_models:
                item_text = f"{model['name']} - {model.get('description', 'Editing model')}"
                if model.get('downloaded', False):
                    item_text += " [Downloaded]"
                else:
                    size_gb = model.get('size_mb', 0) / 1024
                    item_text += f" ({size_gb:.1f}GB)"
                
                item = QListWidgetItem(item_text)
                item.setData(Qt.ItemDataRole.UserRole, model)
                self.model_list.addItem(item)
                
        except Exception as e:
            logger.error(f"Error loading editing models: {e}")
            item = QListWidgetItem(f"Error loading editing models: {e}")
            self.model_list.addItem(item)
    
    def filter_models_by_vram(self):
        """Filter models by VRAM tier."""
        self.load_models()
    
    def is_vram_compatible(self, current_tier: str, required_tier: str) -> bool:
        """Check if current VRAM tier is compatible with required tier."""
        tier_levels = {"low": 1, "medium": 2, "high": 3, "ultra": 4}
        current_level = tier_levels.get(current_tier, 1)
        required_level = tier_levels.get(required_tier, 2)
        return current_level >= required_level
