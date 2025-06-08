"""
Qt Framework Compatibility Layer
Dynamically imports PyQt6 or PySide6 based on availability
"""

QT_FRAMEWORK = None

try:
    from PyQt6.QtWidgets import *
    from PyQt6.QtCore import *
    from PyQt6.QtGui import *
    from PyQt6.QtCore import pyqtSignal as Signal, QUrl, Qt
    from PyQt6.QtGui import QDesktopServices
    
    if not hasattr(Qt, 'UserRole'):
        Qt.UserRole = 0x0100
    if not hasattr(Qt, 'ItemDataRole'):
        Qt.ItemDataRole = Qt
    if not hasattr(Qt, 'ScrollBarAsNeeded'):
        Qt.ScrollBarAsNeeded = Qt.ScrollBarPolicy.ScrollBarAsNeeded if hasattr(Qt, 'ScrollBarPolicy') else 0
    if not hasattr(Qt, 'ScrollBarAlwaysOff'):
        Qt.ScrollBarAlwaysOff = Qt.ScrollBarPolicy.ScrollBarAlwaysOff if hasattr(Qt, 'ScrollBarPolicy') else 1
    if not hasattr(Qt, 'ScrollBarAlwaysOn'):
        Qt.ScrollBarAlwaysOn = Qt.ScrollBarPolicy.ScrollBarAlwaysOn if hasattr(Qt, 'ScrollBarPolicy') else 2
    QT_FRAMEWORK = "PyQt6"
    print("Using PyQt6 for GUI")
except ImportError as pyqt6_error:
    print(f"PyQt6 import failed: {pyqt6_error}")
    try:
        from PySide6.QtWidgets import *
        from PySide6.QtCore import *
        from PySide6.QtGui import *
        from PySide6.QtCore import Signal, QUrl, Qt
        from PySide6.QtGui import QDesktopServices
        
        if not hasattr(Qt, 'UserRole'):
            Qt.UserRole = 0x0100
        if not hasattr(Qt, 'ItemDataRole'):
            Qt.ItemDataRole = Qt
        if not hasattr(Qt, 'ScrollBarAsNeeded'):
            Qt.ScrollBarAsNeeded = Qt.ScrollBarPolicy.ScrollBarAsNeeded if hasattr(Qt, 'ScrollBarPolicy') else 0
        if not hasattr(Qt, 'ScrollBarAlwaysOff'):
            Qt.ScrollBarAlwaysOff = Qt.ScrollBarPolicy.ScrollBarAlwaysOff if hasattr(Qt, 'ScrollBarPolicy') else 1
        if not hasattr(Qt, 'ScrollBarAlwaysOn'):
            Qt.ScrollBarAlwaysOn = Qt.ScrollBarPolicy.ScrollBarAlwaysOn if hasattr(Qt, 'ScrollBarPolicy') else 2
        QT_FRAMEWORK = "PySide6"
        print("Using PySide6 for GUI")
    except ImportError as pyside6_error:
        print(f"PySide6 import failed: {pyside6_error}")
        print("No Qt framework available - GUI will not work")
        
        if "libEGL.so.1" in str(pyqt6_error) or "libEGL.so.1" in str(pyside6_error):
            print("Missing system libraries detected. Install with: sudo apt-get install libegl1-mesa-dev libgl1-mesa-glx")
        
        raise ImportError("No Qt framework available. Please install PyQt6 or PySide6.")

def detect_qt_framework():
    """Detect which Qt framework is available and working."""
    if QT_FRAMEWORK:
        return QT_FRAMEWORK
    return "fallback"

def get_qt_app():
    """Get or create Qt application instance."""
    try:
        if QT_FRAMEWORK in ["PyQt6", "PySide6"]:
            app = QApplication.instance()
            if app is None:
                app = QApplication([])
            return app
        else:
            return None
    except Exception as e:
        print(f"Failed to create Qt application: {e}")
        return None

def handle_windows_dll_issues():
    """Handle Windows-specific DLL loading issues."""
    import platform
    import os
    
    if platform.system() != "Windows":
        return True
        
    try:
        dll_paths = [
            r"C:\Windows\System32",
            r"C:\Windows\SysWOW64",
            os.path.join(os.environ.get('CONDA_PREFIX', ''), 'Library', 'bin'),
            os.path.join(os.environ.get('VIRTUAL_ENV', ''), 'Lib', 'site-packages', 'PyQt6', 'Qt6', 'bin'),
            os.path.join(os.environ.get('VIRTUAL_ENV', ''), 'Lib', 'site-packages', 'PySide6', 'Qt6', 'bin')
        ]
        
        for path in dll_paths:
            if os.path.exists(path):
                os.add_dll_directory(path)
                
        qt_plugin_paths = [
            os.path.join(os.environ.get('VIRTUAL_ENV', ''), 'Lib', 'site-packages', 'PyQt6', 'Qt6', 'plugins'),
            os.path.join(os.environ.get('VIRTUAL_ENV', ''), 'Lib', 'site-packages', 'PySide6', 'Qt6', 'plugins')
        ]
        
        for path in qt_plugin_paths:
            if os.path.exists(path):
                os.environ['QT_PLUGIN_PATH'] = path
                break
                
        return True
        
    except Exception as e:
        print(f"Windows DLL handling failed: {e}")
        return False

def enable_headless_mode():
    """Enable headless mode fallback when GUI fails."""
    try:
        import os
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
        print("Headless mode enabled - GUI will not be available")
        return True
    except Exception as e:
        print(f"Failed to enable headless mode: {e}")
        return False

qt_framework = QT_FRAMEWORK

__all__ = [
    'QT_FRAMEWORK', 'qt_framework', 'Signal', 'Qt',
    'QMainWindow', 'QWidget', 'QVBoxLayout', 'QHBoxLayout', 'QGridLayout',
    'QPushButton', 'QLabel', 'QListWidget', 'QListWidgetItem',
    'QMessageBox', 'QMenu', 'QStatusBar', 'QToolBar', 'QFileDialog',
    'QProgressBar', 'QDialog', 'QTimer', 'QUrl', 'QSpinBox',
    'QAction', 'QIcon', 'QLineEdit', 'QTextEdit', 'QComboBox', 'QCheckBox',
    'QGroupBox', 'QAbstractItemView', 'QToolButton', 'QScrollArea',
    'QDrag', 'QPixmap', 'QThread', 'QTabWidget', 'QDesktopServices',
    'QApplication', 'QCoreApplication',
    'detect_qt_framework', 'get_qt_app', 'handle_windows_dll_issues', 'enable_headless_mode'
]
