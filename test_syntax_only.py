#!/usr/bin/env python3
"""Quick syntax validation for all Python files."""

import ast
import sys
from pathlib import Path

def validate_python_syntax(file_path):
    """Validate Python syntax for a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        ast.parse(source, filename=str(file_path))
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error in {file_path}: {e}"
    except Exception as e:
        return False, f"Error reading {file_path}: {e}"

def main():
    """Validate syntax for all Python files in the project."""
    project_root = Path('/home/ubuntu/repos/ai-manager-app')
    python_files = list(project_root.rglob('*.py'))
    
    errors = []
    valid_count = 0
    
    for py_file in python_files:
        if 'venv' in str(py_file) or '__pycache__' in str(py_file):
            continue
            
        is_valid, error = validate_python_syntax(py_file)
        if is_valid:
            valid_count += 1
            print(f"✅ {py_file.relative_to(project_root)}")
        else:
            errors.append(error)
            print(f"❌ {error}")
    
    print(f"\nSyntax Validation Summary:")
    print(f"Valid files: {valid_count}")
    print(f"Files with errors: {len(errors)}")
    
    if errors:
        print("\nErrors found:")
        for error in errors:
            print(f"  {error}")
        return False
    else:
        print("✅ All Python files have valid syntax!")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
