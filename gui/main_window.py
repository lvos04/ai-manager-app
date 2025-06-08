import sys
import os
import requests
from pathlib import Path
from .qt_compat import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QListWidget, QListWidgetItem,
    QMessageBox, QMenu, QStatusBar, QToolBar, QFileDialog,
    QProgressBar, QDialog, QGridLayout, Qt, QTimer, QUrl,
    QAction, QIcon
)

from .project_dialogs import NewProjectDialog
from .model_manager_dialog import ModelManagerDialog
from .styles import get_app_stylesheet, get_header_style, get_warning_style, get_success_style, get_subheader_style, get_info_style
from config import API_HOST, API_PORT

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.api_url = f"http://{API_HOST}:{API_PORT}"
        
        self.setWindowTitle("AI Project Manager")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet(get_app_stylesheet())
        
        self.init_ui()
        self.setup_toolbar()
        self.setup_statusbar()
        
        # Set up timer for refreshing project list
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_projects)
        self.refresh_timer.start(5000)  # Refresh every 5 seconds
        
        # Set up timer for checking pipeline progress
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.check_pipeline_progress)
        self.progress_timer.start(2000)  # Check progress every 2 seconds
        
        self.active_projects = {}
        
        # Initial project refresh
        self.refresh_projects()
    
    def init_ui(self):
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Create project list section
        project_section = QWidget()
        project_layout = QVBoxLayout(project_section)
        
        # Add project list header
        header_layout = QHBoxLayout()
        header_label = QLabel("Projects")
        header_label.setStyleSheet(get_header_style())
        header_layout.addWidget(header_label)
        
        # Add new project button
        new_project_button = QPushButton("New Project")
        new_project_button.clicked.connect(self.create_new_project)
        header_layout.addWidget(new_project_button)
        
        project_layout.addLayout(header_layout)
        
        # Add project list
        self.project_list = QListWidget()
        self.project_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.project_list.customContextMenuRequested.connect(self.show_project_context_menu)
        self.project_list.itemSelectionChanged.connect(self.on_project_selection_changed)
        project_layout.addWidget(self.project_list)
        
        # Add project management buttons
        project_buttons_layout = QHBoxLayout()
        
        self.run_project_button = QPushButton("Run Selected Project")
        self.run_project_button.clicked.connect(self.run_selected_project)
        self.run_project_button.setEnabled(False)
        
        self.view_project_button = QPushButton("View Details")
        self.view_project_button.clicked.connect(self.view_selected_project)
        self.view_project_button.setEnabled(False)
        
        self.delete_project_button = QPushButton("Delete Selected Project")
        self.delete_project_button.clicked.connect(self.delete_selected_project)
        self.delete_project_button.setEnabled(False)
        
        project_buttons_layout.addWidget(self.run_project_button)
        project_buttons_layout.addWidget(self.view_project_button)
        project_buttons_layout.addWidget(self.delete_project_button)
        project_buttons_layout.addStretch()  # Push buttons to the left
        
        project_layout.addLayout(project_buttons_layout)
        
        # Add project section to main layout
        main_layout.addWidget(project_section)
    
    def setup_toolbar(self):
        # Create toolbar
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        
        # Add refresh action
        refresh_action = QAction("Refresh", self)
        refresh_action.triggered.connect(self.refresh_projects)
        toolbar.addAction(refresh_action)
        
        # Add model manager action
        model_manager_action = QAction("LoRA Model Manager", self)
        model_manager_action.triggered.connect(self.open_model_manager)
        toolbar.addAction(model_manager_action)
        
        settings_action = QAction("Settings", self)
        settings_action.triggered.connect(self.open_app_settings)
        toolbar.addAction(settings_action)
        
        toolbar.addSeparator()
        
        # Add new project action
        new_project_action = QAction("New Project", self)
        new_project_action.triggered.connect(self.create_new_project)
        toolbar.addAction(new_project_action)
    
    def setup_statusbar(self):
        # Create status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")
    
    def refresh_projects(self):
        try:
            response = requests.get(f"{self.api_url}/projects")
            if response.status_code == 200:
                data = response.json()
                self.update_project_list(data["projects"])
                self.statusBar.showMessage("Projects refreshed")
            else:
                self.statusBar.showMessage(f"Error refreshing projects: {response.status_code}")
        except Exception as e:
            self.statusBar.showMessage(f"Error connecting to API: {str(e)}")
    
    def update_project_list(self, projects):
        self.project_list.clear()
        
        for project in projects:
            status_text = project['status']
            if project['status'] in ['running', 'queued']:
                self.active_projects[project['id']] = project
                
                if project['status'] == 'queued':
                    status_text = "QUEUED - Waiting to process"
            else:
                if project['id'] in self.active_projects:
                    self.active_projects.pop(project['id'])
            
            item = QListWidgetItem(f"{project['title']} - {status_text}")
            item.setData(Qt.ItemDataRole.UserRole, project)
            self.project_list.addItem(item)
    
    def create_new_project(self):
        dialog = NewProjectDialog(self)
        if dialog.exec():
            project_data = dialog.get_project_data()
            try:
                response = requests.post(f"{self.api_url}/projects", json=project_data)
                if response.status_code == 200:
                    self.statusBar.showMessage("Project created successfully")
                    self.refresh_projects()
                else:
                    QMessageBox.warning(self, "Error", f"Failed to create project: {response.text}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to create project: {str(e)}")
    
    def show_project_context_menu(self, position):
        item = self.project_list.itemAt(position)
        if item is None:
            return
        
        project = item.data(Qt.ItemDataRole.UserRole)
        
        menu = QMenu()
        run_action = menu.addAction("Run Pipeline")
        view_action = menu.addAction("View Details")
        delete_action = menu.addAction("Delete Project")
        
        action = menu.exec(self.project_list.mapToGlobal(position))
        
        if action == run_action:
            self.run_project_pipeline(project["id"])
        elif action == view_action:
            self.view_project_details(project)
        elif action == delete_action:
            self.delete_project(project["id"])
    
    def run_project_pipeline(self, project_id):
        try:
            response = requests.post(f"{self.api_url}/projects/{project_id}/run")
            if response.status_code == 200:
                self.statusBar.showMessage("Pipeline added to queue")
                self.refresh_projects()
                
                self.check_queue_status()
            else:
                QMessageBox.warning(self, "Error", f"Failed to start pipeline: {response.text}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to start pipeline: {str(e)}")
            
    def delete_project(self, project_id):
        """Delete a project after confirmation."""
        reply = QMessageBox.question(
            self, "Delete Project", 
            "Are you sure you want to delete this project? This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                response = requests.delete(f"{self.api_url}/projects/{project_id}")
                if response.status_code == 200:
                    self.statusBar.showMessage("Project deleted successfully")
                    self.refresh_projects()
                else:
                    QMessageBox.warning(self, "Error", f"Failed to delete project: {response.text}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to delete project: {str(e)}")
            
    def check_queue_status(self):
        """Check the status of the pipeline processing queue."""
        try:
            response = requests.get(f"{self.api_url}/queue/status")
            if response.status_code == 200:
                queue_data = response.json()
                queue_size = queue_data.get("queue_size", 0)
                is_processing = queue_data.get("is_processing", False)
                
                if queue_size > 0:
                    self.statusBar.showMessage(f"Queue: {queue_size} project(s) waiting. Processing: {'Yes' if is_processing else 'No'}")
                else:
                    if not is_processing:
                        self.statusBar.showMessage("Queue empty. No projects processing.")
        except Exception as e:
            print(f"Error checking queue status: {str(e)}")
            
    def check_pipeline_progress(self):
        """Check the progress of active pipelines."""
        if not self.active_projects:
            return
            
        for project_id in list(self.active_projects.keys()):
            try:
                response = requests.get(f"{self.api_url}/projects/{project_id}/status")
                if response.status_code == 200:
                    status_data = response.json()
                    
                    if status_data["status"] in ["completed", "failed"]:
                        if project_id in self.active_projects:
                            self.active_projects.pop(project_id)
                    
                    if status_data["status"] == "running":
                        progress = status_data.get("progress", 0.0)
                        self.show_progress_dialog(project_id, status_data)
                        
                        self.statusBar.showMessage(f"Processing project {project_id}: {progress:.1f}% complete")
            except Exception as e:
                print(f"Error checking pipeline progress: {str(e)}")
                
        self.check_queue_status()
    
    def view_project_details(self, project):
        lora_info = project['lora_model'] or 'N/A'
        
        try:
            response = requests.get(f"{self.api_url}/projects/{project['id']}")
            if response.status_code == 200:
                project_data = response.json()
                if 'loras' in project_data and project_data['loras']:
                    lora_names = [lora['lora_name'] for lora in project_data['loras']]
                    lora_info = ", ".join(lora_names)
        except Exception as e:
            print(f"Error fetching project LoRAs: {str(e)}")
        
        details = f"""
        Project: {project['title']}
        Description: {project['description'] or 'N/A'}
        Status: {project['status']}
        LoRA Models: {lora_info}
        Channel Type: {project['channel_type']}
        Created: {project['created_at']}
        Last Updated: {project['updated_at']}
        Input Path: {project['input_path'] or 'N/A'}
        Output Path: {project['output_path'] or 'N/A'}
        """
        
        QMessageBox.information(self, "Project Details", details)
    
    def open_model_manager(self):
        dialog = ModelManagerDialog(self)
        dialog.exec()
    
    def open_app_settings(self):
        """Open the application settings dialog."""
        from gui.settings_dialog import SettingsDialog
        dialog = SettingsDialog(self)
        dialog.exec()
        
    def on_project_selection_changed(self):
        """Handle project selection changes to enable/disable buttons."""
        selected_items = self.project_list.selectedItems()
        has_selection = len(selected_items) > 0
        
        self.run_project_button.setEnabled(has_selection)
        self.view_project_button.setEnabled(has_selection)
        self.delete_project_button.setEnabled(has_selection)
    
    def run_selected_project(self):
        """Run the pipeline for the selected project."""
        selected_items = self.project_list.selectedItems()
        if not selected_items:
            return
        
        project = selected_items[0].data(Qt.ItemDataRole.UserRole)
        self.run_project_pipeline(project["id"])
    
    def view_selected_project(self):
        """View details for the selected project."""
        selected_items = self.project_list.selectedItems()
        if not selected_items:
            return
        
        project = selected_items[0].data(Qt.ItemDataRole.UserRole)
        self.view_project_details(project)
    
    def delete_selected_project(self):
        """Delete the selected project."""
        selected_items = self.project_list.selectedItems()
        if not selected_items:
            return
        
        project = selected_items[0].data(Qt.ItemDataRole.UserRole)
        self.delete_project(project["id"])
    
    def show_progress_dialog(self, project_id, status_data):
        """Show progress in main window instead of separate dialog."""
        project_title = "Project"
        project_data = None
        
        for i in range(self.project_list.count()):
            item = self.project_list.item(i)
            project = item.data(Qt.ItemDataRole.UserRole)
            if project['id'] == project_id:
                project_title = project['title']
                project_data = project
                break
        
        if not project_data:
            return
            
        progress = status_data.get("progress", 0.0)
        status_text = f"RUNNING - {progress:.1f}% complete"
        
        if not hasattr(self, 'progress_bar'):
            self.progress_bar = QProgressBar()
            self.progress_label = QLabel()
            central_widget = self.centralWidget()
            main_layout = central_widget.layout()
            main_layout.addWidget(self.progress_label)
            main_layout.addWidget(self.progress_bar)
        
        self.progress_label.setText(f"Running: {project_title}")
        self.progress_bar.setValue(int(progress))
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        
        if progress >= 100:
            self.progress_bar.setVisible(False)
            self.progress_label.setText("Ready")
        
        for i in range(self.project_list.count()):
            item = self.project_list.item(i)
            project = item.data(Qt.ItemDataRole.UserRole)
            if project['id'] == project_id:
                item.setText(f"{project_title} - {status_text}")
                break
