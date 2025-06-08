"""
Centralized styles for the application UI.
"""

def get_app_stylesheet():
    """
    Returns the main application stylesheet.
    """
    return """
        QMainWindow, QDialog {
            background-color: #f5f5f5;
            font-family: 'Segoe UI', Arial, sans-serif;
            color: #1a1a1a;
        }
        QLabel {
            color: #1a1a1a;
            font-size: 12px;
        }
        QPushButton {
            background-color: #3498db;  /* More accessible blue */
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
            font-size: 12px;
        }
        QPushButton:hover {
            background-color: #2980b9;  /* Darker hover state */
        }
        QPushButton:pressed {
            background-color: #1f4e79;  /* Even darker pressed state */
        }
        QPushButton:disabled {
            background-color: #bdc3c7;  /* Better disabled contrast */
            color: #7f8c8d;
        }
        QLineEdit, QTextEdit, QPlainTextEdit {
            border: 1px solid #CCCCCC;
            border-radius: 4px;
            padding: 8px;
            font-size: 12px;
            background-color: white;
            color: #1a1a1a;
        }
        QCheckBox {
            color: #1a1a1a;
        }
        QSpinBox {
            background-color: white;
            color: #1a1a1a;
            border: 1px solid #CCCCCC;
            padding: 4px;
            border-radius: 4px;
        }
        QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
            border-color: #1976D2;
        }
        QListWidget, QTreeWidget, QTableWidget {
            border: 1px solid #CCCCCC;
            border-radius: 4px;
            background-color: white;
            alternate-background-color: #f9f9f9;
        }
        QListWidget::item, QTreeWidget::item, QTableWidget::item {
            padding: 4px;
            color: #1a1a1a;
        }
        QListWidget::item:selected, QTreeWidget::item:selected, QTableWidget::item:selected {
            background-color: #1976D2;
            color: white;
        }
        QComboBox {
            border: 1px solid #CCCCCC;
            border-radius: 4px;
            padding: 8px;
            background-color: white;
            color: #1a1a1a;
        }
        QComboBox::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: center right;
            width: 20px;
            border-left: 1px solid #CCCCCC;
        }
        QComboBox QAbstractItemView {
            background-color: white;
            color: #1a1a1a;
            selection-background-color: #1976D2;
            selection-color: white;
        }
        QProgressBar {
            border: 1px solid #CCCCCC;
            border-radius: 4px;
            text-align: center;
            background-color: #EEEEEE;
            color: #1a1a1a;
        }
        QProgressBar::chunk {
            background-color: #1976D2;
            border-radius: 3px;
        }
        QGroupBox {
            border: 1px solid #CCCCCC;
            border-radius: 4px;
            margin-top: 1ex;
            padding-top: 10px;
            font-weight: bold;
            color: #1a1a1a;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 5px;
            color: #1976D2;
        }
        QTabWidget::pane {
            border: 1px solid #CCCCCC;
            border-radius: 4px;
            background-color: white;
            color: #1a1a1a;
        }
        QTabBar::tab {
            background-color: #EEEEEE;
            border: 1px solid #CCCCCC;
            border-bottom: none;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            padding: 8px 12px;
            margin-right: 2px;
            color: #1a1a1a;
        }
        QTabBar::tab:selected {
            background-color: white;
            border-bottom: 1px solid white;
            color: #1a1a1a;
        }
        QStatusBar {
            background-color: #EEEEEE;
            color: #1a1a1a;
        }
        QMenuBar {
            background-color: #EEEEEE;
            color: #1a1a1a;
        }
        QMenuBar::item {
            padding: 6px 10px;
            background-color: transparent;
            color: #1a1a1a;
        }
        QMenuBar::item:selected {
            background-color: #1976D2;
            color: white;
        }
        QMenu {
            background-color: white;
            border: 1px solid #CCCCCC;
            color: #1a1a1a;
        }
        QMenu::item {
            padding: 6px 20px;
            color: #1a1a1a;
        }
        QMenu::item:selected {
            background-color: #1976D2;
            color: white;
        }
        QScrollBar:vertical {
            border: none;
            background-color: #F5F5F5;
            width: 12px;
            margin: 0px;
        }
        QScrollBar::handle:vertical {
            background-color: #CCCCCC;
            min-height: 20px;
            border-radius: 6px;
        }
        QScrollBar::handle:vertical:hover {
            background-color: #AAAAAA;
        }
        QScrollBar:horizontal {
            border: none;
            background-color: #F5F5F5;
            height: 12px;
            margin: 0px;
        }
        QScrollBar::handle:horizontal {
            background-color: #CCCCCC;
            min-width: 20px;
            border-radius: 6px;
        }
        QScrollBar::handle:horizontal:hover {
            background-color: #AAAAAA;
        }
    """

def get_header_style():
    """
    Returns the style for header labels.
    """
    return "font-size: 18px; font-weight: bold; color: #1976D2; margin-bottom: 10px;"

def get_subheader_style():
    """
    Returns the style for subheader labels.
    """
    return "font-size: 14px; font-weight: bold; color: #1976D2; margin-top: 10px;"

def get_warning_style():
    """
    Returns the style for warning labels.
    """
    return "color: #FF5722; font-weight: bold; padding: 10px; background-color: #FFF3E0; border-radius: 4px; margin: 10px 0;"

def get_success_style():
    """
    Returns the style for success labels.
    """
    return "color: #4CAF50; font-weight: bold; padding: 10px; background-color: #E8F5E9; border-radius: 4px; margin: 10px 0;"

def get_error_style():
    """
    Returns the style for error labels.
    """
    return "color: #F44336; font-weight: bold; padding: 10px; background-color: #FFEBEE; border-radius: 4px; margin: 10px 0;"

def get_info_style():
    """
    Returns the style for info labels.
    """
    return "color: #1976D2; font-weight: bold; padding: 10px; background-color: #E3F2FD; border-radius: 4px; margin: 10px 0;"
