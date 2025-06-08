import sys
import threading
import uvicorn
import os
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QCoreApplication
    print("Using PyQt6 for GUI")
except ImportError:
    try:
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import QCoreApplication
        print("Using PySide6 for GUI")
    except ImportError:
        print("No Qt framework available - running in headless mode only")
        QApplication = None
        QCoreApplication = None

try:
    from config import API_HOST, API_PORT
except ImportError:
    API_HOST = "127.0.0.1"
    API_PORT = 8000
from backend.database import init_db

def _detect_vram_tier():
    """Detect VRAM tier for model recommendations."""
    try:
        import torch
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if vram_gb >= 24:
                return "extreme"
            elif vram_gb >= 16:
                return "high"
            elif vram_gb >= 8:
                return "medium"
            else:
                return "low"
        else:
            return "cpu"
    except Exception:
        return "unknown"

def check_cuda_on_startup():
    """Check and auto-install CUDA on application startup."""
    try:
        from backend.cuda_installer import CUDAInstaller
        installer = CUDAInstaller()
        
        gpu_name, cuda_version = installer.detect_nvidia_gpu()
        if gpu_name and not installer.is_pytorch_cuda_installed():
            logger.info(f"NVIDIA GPU detected but CUDA not available, starting auto-setup...")
            logger.info(f"Detected GPU: {gpu_name}, optimal CUDA version: {cuda_version}")
            
            success = installer.auto_setup_cuda()
            if success:
                logger.info("CUDA auto-setup completed successfully")
            else:
                logger.warning("CUDA auto-setup failed, manual installation may be required")
        elif gpu_name:
            logger.info(f"NVIDIA GPU detected: {gpu_name}, CUDA already available")
            
            try:
                tier = _detect_vram_tier()
                logger.info(f"GPU VRAM tier detected: {tier}")
            except Exception as e:
                logger.warning(f"GPU detection failed: {e}")
        else:
            logger.info("No NVIDIA GPU detected by CUDA installer, checking AI Model Manager...")
            
            try:
                tier = _detect_vram_tier()
                if tier != "low":
                    logger.info(f"GPU detected with tier: {tier}")
                else:
                    logger.info("No NVIDIA GPU detected, skipping CUDA setup")
            except Exception as e:
                logger.warning(f"GPU detection also failed: {e}")
                logger.info("No NVIDIA GPU detected, skipping CUDA setup")
                
    except Exception as e:
        logger.warning(f"Startup CUDA check failed: {e}")

def start_api():
    """
    Start the FastAPI server in a separate thread.
    """
    from backend.api import app
    
    uvicorn.run(app, host=API_HOST, port=API_PORT)

def main():
    """
    Main entry point for the application.
    """
    parser = argparse.ArgumentParser(description='AI Project Manager')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode (API only)')
    args = parser.parse_args()
    
    check_cuda_on_startup()
    
    # Initialize database
    init_db()
    
    headless = (
        args.headless or
        QApplication is None or
        os.environ.get('HEADLESS', '').lower() in ('true', '1', 'yes') or
        os.environ.get('CI', '').lower() in ('true', '1') or
        (sys.platform.startswith('linux') and os.environ.get('DISPLAY') is None)
    )
    
    if headless:
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
        print("Running in headless mode with offscreen platform")
        print("API server running at http://{}:{}".format(API_HOST, API_PORT))
        
        # Start API server in a separate thread
        api_thread = threading.Thread(target=start_api, daemon=True)
        api_thread.start()
        
        if QCoreApplication:
            app = QCoreApplication(sys.argv)
            sys.exit(app.exec() if hasattr(app, "exec") else app.exec_())
        else:
            try:
                api_thread.join()
            except KeyboardInterrupt:
                print("Shutting down...")
                sys.exit(0)
    else:
        try:
            from gui import MainWindow
            
            # Start API server in a separate thread
            api_thread = threading.Thread(target=start_api, daemon=True)
            api_thread.start()
            
            print("Waiting for API server to start...")
            api_ready = False
            for i in range(30):  # Wait up to 30 seconds
                try:
                    import requests
                    response = requests.get(f"http://{API_HOST}:{API_PORT}/health", timeout=2)
                    if response.status_code == 200:
                        print("✅ API server is ready")
                        api_ready = True
                        break
                except:
                    pass
                import time
                time.sleep(1)
                print(f"Waiting for API server... ({i+1}/30)")
            
            if not api_ready:
                print("⚠️ Warning: API server may not be ready. GUI may not function properly.")
            
            # Start GUI in normal mode
            if QApplication:
                app = QApplication(sys.argv)
            else:
                raise ImportError("QApplication not available")
            window = MainWindow()
            window.show()
            
            # Run application
            sys.exit(app.exec() if hasattr(app, "exec") else app.exec_())
            
        except ImportError as e:
            logger.error(f"Failed to import GUI components: {e}")
            logger.info("Falling back to headless mode due to GUI import failure")
            
            os.environ['QT_QPA_PLATFORM'] = 'offscreen'
            print("Running in headless mode due to GUI import failure")
            print("API server running at http://{}:{}".format(API_HOST, API_PORT))
            
            api_thread = threading.Thread(target=start_api, daemon=True)
            api_thread.start()
            
            if QCoreApplication:
                app = QCoreApplication(sys.argv)
                sys.exit(app.exec() if hasattr(app, "exec") else app.exec_())
            else:
                try:
                    api_thread.join()
                except KeyboardInterrupt:
                    print("Shutting down...")
                    sys.exit(0)

if __name__ == "__main__":
    main()
