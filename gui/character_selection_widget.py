"""
Character Image Selection Widget for AI Project Manager
Allows users to select and manage character reference images for consistency across episodes.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional

try:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
        QListWidget, QListWidgetItem, QFileDialog, QMessageBox,
        QScrollArea, QFrame, QGridLayout, QTextEdit, QGroupBox
    )
    from PyQt6.QtCore import Qt, pyqtSignal
    from PyQt6.QtGui import QPixmap, QIcon
except ImportError:
    try:
        from PySide6.QtWidgets import (
            QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
            QListWidget, QListWidgetItem, QFileDialog, QMessageBox,
            QScrollArea, QFrame, QGridLayout, QTextEdit, QGroupBox
        )
        from PySide6.QtCore import Qt, Signal as pyqtSignal
        from PySide6.QtGui import QPixmap, QIcon
    except ImportError:
        print("Warning: Neither PyQt6 nor PySide6 available. Character selection widget will not work.")
        QWidget = QVBoxLayout = QHBoxLayout = QLabel = QPushButton = None
        QListWidget = QListWidgetItem = QFileDialog = QMessageBox = None
        QScrollArea = QFrame = QGridLayout = QTextEdit = QGroupBox = None
        Qt = pyqtSignal = QPixmap = QIcon = None

class CharacterImageSelectionWidget(QWidget):
    """Widget for selecting and managing character reference images."""
    
    characters_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.characters_data = {}
        self.project_characters_dir = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        title_label = QLabel("Character Reference Images")
        title_label.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 10px;")
        layout.addWidget(title_label)
        
        char_group = QGroupBox("Character Management")
        char_layout = QVBoxLayout(char_group)
        
        add_char_layout = QHBoxLayout()
        self.add_char_button = QPushButton("Add New Character")
        self.add_char_button.clicked.connect(self.add_character)
        add_char_layout.addWidget(self.add_char_button)
        add_char_layout.addStretch()
        char_layout.addLayout(add_char_layout)
        
        self.character_list = QListWidget()
        self.character_list.setMaximumHeight(150)
        self.character_list.itemSelectionChanged.connect(self.on_character_selected)
        char_layout.addWidget(self.character_list)
        
        layout.addWidget(char_group)
        
        details_group = QGroupBox("Character Details")
        details_layout = QVBoxLayout(details_group)
        
        info_layout = QHBoxLayout()
        
        name_layout = QVBoxLayout()
        name_layout.addWidget(QLabel("Character Name:"))
        self.char_name_label = QLabel("Select a character")
        self.char_name_label.setStyleSheet("font-weight: bold;")
        name_layout.addWidget(self.char_name_label)
        info_layout.addLayout(name_layout)
        
        desc_layout = QVBoxLayout()
        desc_layout.addWidget(QLabel("Description:"))
        self.char_description = QTextEdit()
        self.char_description.setMaximumHeight(60)
        self.char_description.setPlaceholderText("Enter character description...")
        self.char_description.textChanged.connect(self.on_description_changed)
        desc_layout.addWidget(self.char_description)
        info_layout.addLayout(desc_layout)
        
        details_layout.addLayout(info_layout)
        
        images_layout = QVBoxLayout()
        
        images_header = QHBoxLayout()
        images_header.addWidget(QLabel("Reference Images:"))
        
        self.add_image_button = QPushButton("Add Image")
        self.add_image_button.clicked.connect(self.add_reference_image)
        self.add_image_button.setEnabled(False)
        images_header.addWidget(self.add_image_button)
        
        self.remove_image_button = QPushButton("Remove Selected")
        self.remove_image_button.clicked.connect(self.remove_reference_image)
        self.remove_image_button.setEnabled(False)
        images_header.addWidget(self.remove_image_button)
        
        images_header.addStretch()
        images_layout.addLayout(images_header)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(200)
        
        self.image_grid_widget = QWidget()
        self.image_grid_layout = QGridLayout(self.image_grid_widget)
        self.image_grid_layout.setSpacing(5)
        
        scroll_area.setWidget(self.image_grid_widget)
        images_layout.addWidget(scroll_area)
        
        details_layout.addLayout(images_layout)
        
        actions_layout = QHBoxLayout()
        
        self.remove_char_button = QPushButton("Remove Character")
        self.remove_char_button.clicked.connect(self.remove_character)
        self.remove_char_button.setEnabled(False)
        actions_layout.addWidget(self.remove_char_button)
        
        actions_layout.addStretch()
        
        self.load_existing_button = QPushButton("Load from Previous Project")
        self.load_existing_button.clicked.connect(self.load_existing_characters)
        actions_layout.addWidget(self.load_existing_button)
        
        details_layout.addLayout(actions_layout)
        
        layout.addWidget(details_group)
        
        self.selected_image_labels = []
        
    def set_project_directory(self, project_dir: str):
        """Set the project directory for character storage."""
        self.project_characters_dir = Path(project_dir) / "characters"
        self.project_characters_dir.mkdir(exist_ok=True)
        self.load_project_characters()
    
    def load_project_characters(self):
        """Load existing characters from project directory."""
        if not self.project_characters_dir or not self.project_characters_dir.exists():
            return
        
        self.characters_data.clear()
        self.character_list.clear()
        
        for char_dir in self.project_characters_dir.iterdir():
            if char_dir.is_dir():
                char_data_file = char_dir / "character_data.json"
                if char_data_file.exists():
                    try:
                        with open(char_data_file, 'r') as f:
                            char_data = json.load(f)
                        
                        char_name = char_data.get('name', char_dir.name)
                        self.characters_data[char_name] = char_data
                        
                        item = QListWidgetItem(char_name)
                        self.character_list.addItem(item)
                        
                    except Exception as e:
                        print(f"Error loading character data from {char_data_file}: {e}")
    
    def add_character(self):
        """Add a new character."""
        from PyQt6.QtWidgets import QInputDialog
        
        char_name, ok = QInputDialog.getText(self, "Add Character", "Character Name:")
        
        if ok and char_name.strip():
            char_name = char_name.strip()
            
            if char_name in self.characters_data:
                QMessageBox.warning(self, "Warning", f"Character '{char_name}' already exists!")
                return
            
            char_data = {
                "name": char_name,
                "description": "",
                "reference_images": [],
                "created_timestamp": __import__('time').time()
            }
            
            self.characters_data[char_name] = char_data
            
            item = QListWidgetItem(char_name)
            self.character_list.addItem(item)
            
            self.character_list.setCurrentItem(item)
            
            self.save_character_data(char_name)
            
            self.characters_changed.emit()
    
    def remove_character(self):
        """Remove the selected character."""
        current_item = self.character_list.currentItem()
        if not current_item:
            return
        
        char_name = current_item.text()
        
        reply = QMessageBox.question(
            self, "Confirm Removal", 
            f"Are you sure you want to remove character '{char_name}' and all reference images?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            if char_name in self.characters_data:
                del self.characters_data[char_name]
            
            row = self.character_list.row(current_item)
            self.character_list.takeItem(row)
            
            if self.project_characters_dir:
                char_dir = self.project_characters_dir / char_name
                if char_dir.exists():
                    import shutil
                    shutil.rmtree(char_dir)
            
            self.clear_character_details()
            
            self.characters_changed.emit()
    
    def on_character_selected(self):
        """Handle character selection."""
        current_item = self.character_list.currentItem()
        
        if current_item:
            char_name = current_item.text()
            self.display_character_details(char_name)
            self.add_image_button.setEnabled(True)
            self.remove_char_button.setEnabled(True)
        else:
            self.clear_character_details()
            self.add_image_button.setEnabled(False)
            self.remove_char_button.setEnabled(False)
    
    def display_character_details(self, char_name: str):
        """Display details for the selected character."""
        if char_name not in self.characters_data:
            return
        
        char_data = self.characters_data[char_name]
        
        self.char_name_label.setText(char_name)
        self.char_description.setText(char_data.get('description', ''))
        
        self.update_image_grid(char_name)
    
    def clear_character_details(self):
        """Clear character details display."""
        self.char_name_label.setText("Select a character")
        self.char_description.clear()
        self.clear_image_grid()
        self.remove_image_button.setEnabled(False)
    
    def on_description_changed(self):
        """Handle description text changes."""
        current_item = self.character_list.currentItem()
        if current_item:
            char_name = current_item.text()
            if char_name in self.characters_data:
                self.characters_data[char_name]['description'] = self.char_description.toPlainText()
                self.save_character_data(char_name)
    
    def add_reference_image(self):
        """Add a reference image for the selected character."""
        current_item = self.character_list.currentItem()
        if not current_item:
            return
        
        char_name = current_item.text()
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Image", "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.tiff)"
        )
        
        if file_path:
            char_dir = self.project_characters_dir / char_name
            char_dir.mkdir(exist_ok=True)
            
            import shutil
            file_name = Path(file_path).name
            dest_path = char_dir / f"ref_{len(self.characters_data[char_name]['reference_images'])}_{file_name}"
            
            try:
                shutil.copy2(file_path, dest_path)
                
                self.characters_data[char_name]['reference_images'].append(str(dest_path))
                self.save_character_data(char_name)
                
                self.update_image_grid(char_name)
                
                self.characters_changed.emit()
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to copy image: {e}")
    
    def remove_reference_image(self):
        """Remove selected reference image."""
        current_item = self.character_list.currentItem()
        if not current_item:
            return
        
        char_name = current_item.text()
        
        if self.characters_data[char_name]['reference_images']:
            image_path = self.characters_data[char_name]['reference_images'].pop()
            
            try:
                if os.path.exists(image_path):
                    os.remove(image_path)
            except Exception as e:
                print(f"Error removing image file: {e}")
            
            self.save_character_data(char_name)
            self.update_image_grid(char_name)
            
            self.characters_changed.emit()
    
    def update_image_grid(self, char_name: str):
        """Update the image grid display."""
        self.clear_image_grid()
        
        if char_name not in self.characters_data:
            return
        
        images = self.characters_data[char_name]['reference_images']
        
        if images:
            self.remove_image_button.setEnabled(True)
            
            cols = 3
            for i, image_path in enumerate(images):
                if os.path.exists(image_path):
                    row = i // cols
                    col = i % cols
                    
                    image_label = QLabel()
                    image_label.setFixedSize(80, 80)
                    image_label.setStyleSheet("border: 1px solid gray;")
                    image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    
                    pixmap = QPixmap(image_path)
                    if not pixmap.isNull():
                        scaled_pixmap = pixmap.scaled(
                            78, 78, Qt.AspectRatioMode.KeepAspectRatio, 
                            Qt.TransformationMode.SmoothTransformation
                        )
                        image_label.setPixmap(scaled_pixmap)
                    else:
                        image_label.setText("Invalid\nImage")
                    
                    self.image_grid_layout.addWidget(image_label, row, col)
                    self.selected_image_labels.append(image_label)
        else:
            self.remove_image_button.setEnabled(False)
    
    def clear_image_grid(self):
        """Clear the image grid."""
        for label in self.selected_image_labels:
            label.deleteLater()
        self.selected_image_labels.clear()
        
        while self.image_grid_layout.count():
            child = self.image_grid_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
    
    def save_character_data(self, char_name: str):
        """Save character data to file."""
        if not self.project_characters_dir or char_name not in self.characters_data:
            return
        
        char_dir = self.project_characters_dir / char_name
        char_dir.mkdir(exist_ok=True)
        
        char_data_file = char_dir / "character_data.json"
        
        try:
            with open(char_data_file, 'w') as f:
                json.dump(self.characters_data[char_name], f, indent=2)
        except Exception as e:
            print(f"Error saving character data: {e}")
    
    def load_existing_characters(self):
        """Load characters from an existing project."""
        project_dir = QFileDialog.getExistingDirectory(
            self, "Select Previous Project Directory"
        )
        
        if project_dir:
            prev_chars_dir = Path(project_dir) / "characters"
            if prev_chars_dir.exists():
                import shutil
                
                for char_dir in prev_chars_dir.iterdir():
                    if char_dir.is_dir():
                        dest_char_dir = self.project_characters_dir / char_dir.name
                        
                        if dest_char_dir.exists():
                            reply = QMessageBox.question(
                                self, "Character Exists", 
                                f"Character '{char_dir.name}' already exists. Overwrite?",
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                            )
                            if reply == QMessageBox.StandardButton.No:
                                continue
                            
                            shutil.rmtree(dest_char_dir)
                        
                        shutil.copytree(char_dir, dest_char_dir)
                
                self.load_project_characters()
                self.characters_changed.emit()
                
                QMessageBox.information(self, "Success", "Characters loaded successfully!")
            else:
                QMessageBox.warning(self, "Warning", "No characters directory found in selected project.")
    
    def get_character_data(self) -> Dict:
        """Get all character data."""
        return self.characters_data.copy()
    
    def get_character_names(self) -> List[str]:
        """Get list of character names."""
        return list(self.characters_data.keys())
    
    def has_characters(self) -> bool:
        """Check if any characters are defined."""
        return len(self.characters_data) > 0
