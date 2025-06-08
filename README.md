# AI Project Manager

A desktop application for managing AI content generation projects with LoRA model selection.

## Features

- Project management with LoRA model selection for different channel types
- Support for multiple content channel types: gaming, anime, superhero, manga, marvel/DC, original manga
- Model management with download and versioning capabilities
- Local processing of AI models without external API dependencies
- Comprehensive FastAPI backend with asynchronous execution
- Modern PyQt6 GUI interface
- Multiple video export formats (MP4, WebM, MOV, AVI)
- Automatic model version checking from HuggingFace
- VRAM-based model quantization for optimal performance
- Enhanced error handling with recovery options
- Sequential processing queue for multiple projects

## Requirements

- Python 3.12+
- PyQt6 6.5.0+
- FastAPI 0.109.0+
- SQLAlchemy 2.0.27+
- Pydantic 2.5.3+
- Other dependencies listed in requirements.txt

## Installation

1. Clone the repository
2. Install dependencies: 
   ```bash
   pip install -r requirements.txt
   ```
3. For Linux/Unix systems, ensure you have the required system dependencies:
   ```bash
   sudo apt-get install -y libxcb-cursor0 libxcb-xinerama0 libxcb-xkb1 libxcb-shape0 libxcb-icccm4 libxcb-keysyms1 libxcb-image0 libxcb-render-util0 libxcb-randr0 libxcb-sync1 libxcb-xfixes0 libxkbcommon-x11-0
   ```
4. Run the application: 
   ```bash
   python main.py
   ```

## Usage

1. **Create a New Project**:
   - Click "New Project" button
   - Enter project details
   - Select a LoRA model for the project
   - Choose a channel type (gaming, anime, superhero, manga, marvel_dc, original_manga)

2. **Manage Models**:
   - Click "Model Manager" button
   - View available LoRA models
   - Download models as needed

3. **Run Projects**:
   - Select a project from the list
   - Click "Run" or use the context menu
   - Monitor progress in the status bar

## Building Executable

To build a standalone executable:

```bash
pyinstaller build.spec
```

The executable will be created in the `dist/AI_Project_Manager` directory.

## Directory Structure

- `main.py`: Application entry point
- `config.py`: Configuration settings
- `backend/`: FastAPI backend and business logic
- `gui/`: PyQt6 graphical user interface
- `database/`: SQLite database
- `assets/`: Application assets
- `models/`: AI model storage
- `output/`: Pipeline output directory

## License

This project is licensed under the MIT License - see the LICENSE file for details.
