"""
Automatic CUDA Toolkit and PyTorch installation with UAC elevation for NVIDIA GPUs.
"""
import os
import sys
import subprocess
import requests
import platform
import logging
import ctypes
from pathlib import Path

logger = logging.getLogger(__name__)

class CUDAInstaller:
    _installation_attempted = False
    _max_attempts = 1
    def __init__(self):
        self.system = platform.system()
        self.gpu_cuda_versions = {
            "4060": "12.1",
            "4070": "12.1", 
            "4080": "12.1",
            "4090": "12.1",
            "3090": "11.8",
            "3090ti": "11.8",
            "3080": "11.8",
            "3070": "11.8",
            "3060": "11.8",
            "2080": "11.8",
            "2070": "11.8"
        }
        self.pytorch_indices = {
            "11.8": "https://download.pytorch.org/whl/cu118",
            "12.1": "https://download.pytorch.org/whl/cu121"
        }

    def is_admin(self):
        """Check if running with administrator privileges."""
        try:
            return ctypes.windll.shell32.IsUserAnAdmin()
        except:
            return False

    def run_as_admin(self, cmd_args):
        """Restart script with administrator privileges."""
        try:
            if self.system == "Windows":
                ctypes.windll.shell32.ShellExecuteW(
                    None, "runas", sys.executable, " ".join(cmd_args), None, 1
                )
                return True
        except Exception as e:
            logger.error(f"Failed to elevate privileges: {e}")
        return False

    def detect_nvidia_gpu(self):
        """Detect NVIDIA GPU and determine optimal CUDA version."""
        try:
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0).lower()
                    logger.info(f"Detected NVIDIA GPU via PyTorch: {gpu_name}")
                    
                    for model, cuda_version in self.gpu_cuda_versions.items():
                        if model in gpu_name:
                            logger.info(f"GPU {model} detected, recommending CUDA {cuda_version}")
                            return gpu_name, cuda_version
                    
                    logger.info("Unknown NVIDIA GPU, defaulting to CUDA 11.8")
                    return gpu_name, "11.8"
            except Exception as e:
                logger.debug(f"PyTorch GPU detection failed: {e}")
            
            if self.system == "Windows":
                result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    gpu_name = result.stdout.strip().lower()
                    logger.info(f"Detected NVIDIA GPU: {gpu_name}")
                    
                    for model, cuda_version in self.gpu_cuda_versions.items():
                        if model in gpu_name:
                            logger.info(f"GPU {model} detected, recommending CUDA {cuda_version}")
                            return gpu_name, cuda_version
                    
                    logger.info("Unknown NVIDIA GPU, defaulting to CUDA 11.8")
                    return gpu_name, "11.8"
                
                result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0 and 'nvidia' in result.stdout.lower():
                    return "nvidia gpu detected", "11.8"
            
            elif self.system == "Linux":
                try:
                    result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        gpu_name = result.stdout.strip().lower()
                        logger.info(f"Detected NVIDIA GPU via nvidia-smi: {gpu_name}")
                        
                        for model, cuda_version in self.gpu_cuda_versions.items():
                            if model in gpu_name:
                                logger.info(f"GPU {model} detected, recommending CUDA {cuda_version}")
                                return gpu_name, cuda_version
                        
                        logger.info("Unknown NVIDIA GPU, defaulting to CUDA 11.8")
                        return gpu_name, "11.8"
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                    logger.debug("nvidia-smi not available on Linux system")
                    
        except Exception as e:
            logger.warning(f"GPU detection failed: {e}")
        return None, None

    def is_cuda_installed(self, version):
        """Check if specific CUDA version is installed."""
        try:
            result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
            if result.returncode == 0 and version in result.stdout:
                return True
        except:
            pass
        return False

    def is_pytorch_cuda_installed(self):
        """Check if PyTorch with CUDA is installed."""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

    def download_cuda_installer(self, version):
        """Download CUDA installer for specified version."""
        cuda_urls = {
            "11.8": "https://developer.download.nvidia.com/compute/cuda/11.8.0/network_installers/cuda_11.8.0_windows_network.exe",
            "12.1": "https://developer.download.nvidia.com/compute/cuda/12.1.0/network_installers/cuda_12.1.0_windows_network.exe"
        }
        
        if version not in cuda_urls:
            logger.error(f"Unsupported CUDA version: {version}")
            return None
            
        try:
            installer_path = Path(f"cuda_{version}_installer.exe")
            logger.info(f"Downloading CUDA {version} installer...")
            
            response = requests.get(cuda_urls[version], stream=True)
            response.raise_for_status()
            
            with open(installer_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"CUDA {version} installer downloaded successfully")
            return installer_path
            
        except Exception as e:
            logger.error(f"Failed to download CUDA {version} installer: {e}")
            return None

    def install_cuda_toolkit(self, version):
        """Install CUDA toolkit with administrator privileges."""
        if not self.is_admin():
            logger.info("Requesting administrator privileges for CUDA installation...")
            return self.run_as_admin([__file__, "install_cuda", version])
            
        try:
            installer_path = self.download_cuda_installer(version)
            if not installer_path:
                return False
            
            logger.info(f"Installing CUDA {version} toolkit...")
            
            components = [
                f'nvcc_{version}', f'cuobjdump_{version}', f'nvprune_{version}',
                f'cupti_{version}', f'gpu_library_advisor_{version}', f'memcheck_{version}',
                f'nvdisasm_{version}', f'nvprof_{version}', f'visual_profiler_{version}',
                f'visual_studio_integration_{version}'
            ]
            
            result = subprocess.run([
                str(installer_path), 
                '-s',  # Silent installation
                *components
            ], timeout=1800)  # 30 minute timeout
            
            installer_path.unlink(missing_ok=True)
            
            if result.returncode == 0:
                logger.info(f"CUDA {version} toolkit installed successfully")
                return True
            else:
                logger.error(f"CUDA {version} installation failed with code {result.returncode}")
                return False
                
        except Exception as e:
            logger.error(f"CUDA {version} installation error: {e}")
            return False

    def install_pytorch_cuda(self, cuda_version):
        """Install PyTorch with CUDA support for specified version."""
        try:
            pytorch_index = self.pytorch_indices.get(cuda_version)
            if not pytorch_index:
                logger.error(f"No PyTorch index for CUDA {cuda_version}")
                return False
                
            logger.info(f"Installing PyTorch with CUDA {cuda_version} support...")
            
            subprocess.run([sys.executable, '-m', 'pip', 'uninstall', 
                          'torch', 'torchvision', 'torchaudio', '-y'], 
                          capture_output=True)
            
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', 
                'torch', 'torchvision', 'torchaudio',
                '--index-url', pytorch_index
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"PyTorch with CUDA {cuda_version} installed successfully")
                return True
            else:
                logger.error(f"PyTorch installation failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"PyTorch installation error: {e}")
            return False

    def auto_setup_cuda(self):
        """Automatically set up CUDA with optimal version for detected GPU."""
        if CUDAInstaller._installation_attempted:
            logger.warning("CUDA installation already attempted, skipping to prevent loops")
            return False
        
        if self.is_pytorch_cuda_installed():
            try:
                import torch
                if torch.cuda.is_available():
                    logger.info("PyTorch CUDA is already working, skipping installation")
                    return True
            except:
                pass
        
        gpu_name, cuda_version = self.detect_nvidia_gpu()
        if not gpu_name:
            logger.info("No NVIDIA GPU detected, skipping CUDA setup")
            return False
            
        logger.info(f"NVIDIA GPU detected: {gpu_name}, optimal CUDA version: {cuda_version}")
        CUDAInstaller._installation_attempted = True
        
        if self.is_cuda_installed(cuda_version):
            logger.info("CUDA toolkit already installed, checking PyTorch compatibility...")
            if not self.is_pytorch_cuda_installed():
                return self.install_pytorch_cuda(cuda_version)
            return True
            
        logger.info(f"CUDA {cuda_version} not found, installing...")
        if not self.install_cuda_toolkit(cuda_version):
            logger.error(f"Failed to install CUDA {cuda_version}")
            return False
        
        if not self.install_pytorch_cuda(cuda_version):
            logger.error(f"Failed to install PyTorch with CUDA {cuda_version}")
            return False
            
        logger.info(f"CUDA {cuda_version} setup completed successfully")
        return True

if __name__ == "__main__" and len(sys.argv) > 1:
    if sys.argv[1] == "install_cuda" and len(sys.argv) > 2:
        installer = CUDAInstaller()
        version = sys.argv[2]
        success = installer.install_cuda_toolkit(version)
        sys.exit(0 if success else 1)
