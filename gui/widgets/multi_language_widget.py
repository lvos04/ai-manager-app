from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QListWidget, QListWidgetItem, QPushButton, QMessageBox)
from PyQt6.QtCore import Qt

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
            item.setData(Qt.ItemDataRole.UserRole, lang_code)
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
            if item.data(Qt.ItemDataRole.UserRole) == "en":
                self.available_list.setCurrentItem(item)
                self.add_language()
                break
    
    def add_language(self):
        """Add selected language to the list."""
        current_item = self.available_list.currentItem()
        if not current_item:
            return
        
        lang_code = current_item.data(Qt.ItemDataRole.UserRole)
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
        
        lang_code = current_item.data(Qt.ItemDataRole.UserRole)
        if lang_code in self.selected_languages:
            self.selected_languages.remove(lang_code)
            self.update_selected_list()
    
    def update_selected_list(self):
        """Update the selected languages display."""
        self.selected_list.clear()
        for lang_code in self.selected_languages:
            lang_name = next((name for name, code in self.available_languages if code == lang_code), lang_code)
            item = QListWidgetItem(f"{lang_name} ({lang_code})")
            item.setData(Qt.ItemDataRole.UserRole, lang_code)
            self.selected_list.addItem(item)
    
    def get_selected_languages(self):
        """Get list of selected language codes."""
        return self.selected_languages.copy()
