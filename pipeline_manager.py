<<<<<<< HEAD
#!/usr/bin/env python3
"""
Central Pipeline Manager for AI Project Manager App

This module provides a unified interface for executing all channel-specific pipelines
with support for YAML, JSON, and TXT input formats.
"""

import yaml
import json
import logging
import argparse
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.pipelines.channel_specific import (
    anime_pipeline,
    gaming_pipeline,
    manga_pipeline,
    marvel_dc_pipeline,
    superhero_pipeline,
    original_manga_pipeline,
)

PIPELINE_MAP = {
    "anime": anime_pipeline.AnimeChannelPipeline,
    "gaming": gaming_pipeline.GamingChannelPipeline,
    "manga": manga_pipeline.MangaChannelPipeline,
    "marvel_dc": marvel_dc_pipeline.MarvelDCChannelPipeline,
    "superhero": superhero_pipeline.SuperheroChannelPipeline,
    "original_manga": original_manga_pipeline.OriginalMangaChannelPipeline,
}

def setup_logging(log_file: str = "pipeline.log") -> None:
    """Set up comprehensive logging for pipeline execution."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def load_config(input_file: str) -> Dict[str, Any]:
    """
    Load configuration from YAML, JSON, or TXT input files.
    
    Args:
        input_file: Path to the input configuration file
        
    Returns:
        Dictionary containing the loaded configuration
        
    Raises:
        ValueError: If the file format is not supported
        FileNotFoundError: If the input file doesn't exist
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    file_extension = input_path.suffix.lower()
    
    try:
        if file_extension in [".yaml", ".yml"]:
            with open(input_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                logging.info(f"Loaded YAML configuration from {input_file}")
                return config
                
        elif file_extension == ".json":
            with open(input_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                logging.info(f"Loaded JSON configuration from {input_file}")
                return config
                
        elif file_extension == ".txt":
            with open(input_file, 'r', encoding='utf-8') as f:
                script_content = f.read()
                config = {
                    "script": script_content,
                    "title": input_path.stem,
                    "input_type": "text_script"
                }
                logging.info(f"Loaded TXT script from {input_file}")
                return config
                
        else:
            raise ValueError(f"Unsupported input format: {file_extension}. Supported formats: .yaml, .yml, .json, .txt")
            
    except Exception as e:
        logging.error(f"Error loading configuration from {input_file}: {str(e)}")
        raise

def validate_pipeline_config(pipeline_type: str, config: Dict[str, Any]) -> None:
    """
    Validate pipeline configuration for the specified pipeline type.
    
    Args:
        pipeline_type: The type of pipeline (anime, gaming, etc.)
        config: Configuration dictionary to validate
        
    Raises:
        ValueError: If the configuration is invalid
    """
    if pipeline_type not in PIPELINE_MAP:
        available_types = ", ".join(PIPELINE_MAP.keys())
        raise ValueError(f"Unknown pipeline type: {pipeline_type}. Available types: {available_types}")
    
    if not config:
        raise ValueError("Configuration is empty")
    
    if config.get("input_type") == "text_script" and not config.get("script"):
        raise ValueError("Text script configuration must contain 'script' content")
    
    logging.info(f"Configuration validated for {pipeline_type} pipeline")

def run_pipeline(pipeline_type: str, input_file: str, output_path: Optional[str] = None, 
                base_model: str = "stable_diffusion_1_5", lora_models: Optional[list] = None,
                **kwargs) -> str:
    """
    Execute the specified pipeline with the given configuration.
    
    Args:
        pipeline_type: Type of pipeline to run (anime, gaming, etc.)
        input_file: Path to the input configuration file
        output_path: Optional output directory path
        base_model: Base AI model to use
        lora_models: List of LoRA models to apply
        **kwargs: Additional pipeline parameters
        
    Returns:
        Path to the generated output file
        
    Raises:
        ValueError: If pipeline type or configuration is invalid
        Exception: If pipeline execution fails
    """
    logging.info(f"Starting {pipeline_type} pipeline with input: {input_file}")
    
    try:
        config = load_config(input_file)
        validate_pipeline_config(pipeline_type, config)
        
        if output_path is None:
            output_path = f"output/{pipeline_type}_{Path(input_file).stem}"
        
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        pipeline_class = PIPELINE_MAP[pipeline_type]
        
        pipeline = pipeline_class()
        
        pipeline_params = {
            "input_path": input_file,
            "output_path": str(output_dir),
            "base_model": base_model,
            "lora_models": lora_models or [],
            "lora_paths": {},
            "render_fps": kwargs.get("render_fps", 24),
            "output_fps": kwargs.get("output_fps", 24),
            "frame_interpolation_enabled": kwargs.get("frame_interpolation_enabled", True),
            "llm_model": kwargs.get("llm_model", "microsoft/DialoGPT-medium"),
            "language": kwargs.get("language", "en"),
            **kwargs
        }
        
        logging.info(f"Pipeline parameters: {pipeline_params}")
        
        output_file = pipeline.run(**pipeline_params)
        
        logging.info(f"Pipeline execution completed successfully. Output: {output_file}")
        return str(output_file)
        
    except Exception as e:
        error_msg = f"Pipeline execution failed: {str(e)}"
        logging.error(error_msg)
        raise Exception(error_msg) from e

def list_example_configs() -> Dict[str, list]:
    """
    List all available example configurations from input_formats_documentation.
    
    Returns:
        Dictionary mapping channel types to lists of example config files
    """
    examples = {}
    docs_dir = Path("input_formats_documentation")
    
    if not docs_dir.exists():
        logging.warning("input_formats_documentation directory not found")
        return examples
    
    for channel_dir in docs_dir.iterdir():
        if channel_dir.is_dir():
            channel_type = channel_dir.name
            examples[channel_type] = []
            
            examples_dir = channel_dir / "examples"
            templates_dir = channel_dir / "templates"
            
            if examples_dir.exists():
                for example_file in examples_dir.iterdir():
                    if example_file.suffix.lower() in [".yaml", ".yml", ".json", ".txt"]:
                        examples[channel_type].append(str(example_file))
            
            if templates_dir.exists():
                for template_file in templates_dir.iterdir():
                    if template_file.suffix.lower() in [".yaml", ".yml", ".json", ".txt"]:
                        examples[channel_type].append(str(template_file))
    
    return examples

def main():
    """Main entry point for command-line execution."""
    parser = argparse.ArgumentParser(
        description="Central Pipeline Manager for AI Project Manager App",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline_manager.py --pipeline anime --config input.yaml
  python pipeline_manager.py --pipeline gaming --config script.txt --output ./output
  python pipeline_manager.py --list-examples
        """
    )
    
    parser.add_argument("--pipeline", type=str, 
                       choices=list(PIPELINE_MAP.keys()),
                       help="Pipeline type to execute")
    
    parser.add_argument("--config", type=str,
                       help="Path to input configuration file (YAML, JSON, or TXT)")
    
    parser.add_argument("--output", type=str,
                       help="Output directory path (optional)")
    
    parser.add_argument("--base-model", type=str, default="stable_diffusion_1_5",
                       help="Base AI model to use")
    
    parser.add_argument("--lora-models", type=str, nargs="*",
                       help="LoRA models to apply")
    
    parser.add_argument("--render-fps", type=int, default=24,
                       help="Render frame rate")
    
    parser.add_argument("--output-fps", type=int, default=24,
                       help="Output frame rate")
    
    parser.add_argument("--language", type=str, default="en",
                       help="Language for content generation")
    
    parser.add_argument("--list-examples", action="store_true",
                       help="List all available example configurations")
    
    parser.add_argument("--log-file", type=str, default="pipeline.log",
                       help="Log file path")
    
    args = parser.parse_args()
    
    setup_logging(args.log_file)
    
    try:
        if args.list_examples:
            examples = list_example_configs()
            print("\nAvailable example configurations:")
            for channel_type, files in examples.items():
                print(f"\n{channel_type.upper()}:")
                for file_path in files:
                    print(f"  - {file_path}")
            return
        
        if not args.pipeline or not args.config:
            parser.error("Both --pipeline and --config are required (unless using --list-examples)")
        
        output_file = run_pipeline(
            pipeline_type=args.pipeline,
            input_file=args.config,
            output_path=args.output,
            base_model=args.base_model,
            lora_models=args.lora_models,
            render_fps=args.render_fps,
            output_fps=args.output_fps,
            language=args.language
        )
        
        print(f"\nPipeline execution completed successfully!")
        print(f"Output saved to: {output_file}")
        
    except Exception as e:
        logging.error(f"Pipeline manager failed: {str(e)}")
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
||||||| fb95d81
=======
#!/usr/bin/env python3
"""
Central Pipeline Manager for AI Project Manager App

This module provides a unified interface for executing all channel-specific pipelines
with support for multiple input formats and comprehensive logging.
"""

import os
import sys
import json
import yaml
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from backend.pipelines.channel_specific.anime_pipeline import AnimePipeline
    from backend.pipelines.channel_specific.gaming_pipeline import GamingPipeline
    from backend.pipelines.channel_specific.manga_pipeline import MangaPipeline
    from backend.pipelines.channel_specific.marvel_dc_pipeline import MarvelDCPipeline
    from backend.pipelines.channel_specific.superhero_pipeline import SuperheroPipeline
    from backend.pipelines.channel_specific.original_manga_pipeline import OriginalMangaPipeline
except ImportError as e:
    print(f"Warning: Could not import pipeline classes: {e}")
    print("Pipeline execution may not work properly.")

PIPELINE_MAP = {
    "anime": AnimePipeline,
    "gaming": GamingPipeline,
    "manga": MangaPipeline,
    "marvel_dc": MarvelDCPipeline,
    "superhero": SuperheroPipeline,
    "original_manga": OriginalMangaPipeline,
}

def setup_logging(log_file: Optional[str] = None, log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized - Level: {log_level}, File: {log_file or 'None'}")

def load_config(input_file: str) -> Dict[str, Any]:
    """
    Load configuration from various input formats.
    
    Args:
        input_file: Path to the input configuration file
        
    Returns:
        Dictionary containing the loaded configuration
        
    Raises:
        ValueError: If the file format is not supported
        FileNotFoundError: If the input file doesn't exist
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    file_ext = Path(input_file).suffix.lower()
    
    try:
        if file_ext in [".yaml", ".yml"]:
            with open(input_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        elif file_ext == ".json":
            with open(input_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
        elif file_ext == ".txt":
            with open(input_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                config = {
                    "script": content,
                    "title": f"Project from {Path(input_file).stem}",
                    "description": f"Generated from text file: {input_file}"
                }
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        logging.info(f"Successfully loaded configuration from {input_file}")
        return config
        
    except Exception as e:
        logging.error(f"Failed to load configuration from {input_file}: {e}")
        raise

def validate_pipeline_config(config: Dict[str, Any], pipeline_type: str) -> bool:
    """
    Validate pipeline configuration.
    
    Args:
        config: Configuration dictionary
        pipeline_type: Type of pipeline to validate for
        
    Returns:
        True if configuration is valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a dictionary")
    
    if not config:
        raise ValueError("Configuration cannot be empty")
    
    required_fields = {
        "anime": ["script"],
        "gaming": ["script"],
        "manga": ["script"],
        "marvel_dc": ["script"],
        "superhero": ["script"],
        "original_manga": ["script"],
    }
    
    if pipeline_type in required_fields:
        for field in required_fields[pipeline_type]:
            if field not in config:
                if field == "script" and not any(key in config for key in ["content", "story", "text"]):
                    raise ValueError(f"Configuration missing required field: {field}")
    
    logging.info(f"Configuration validation passed for {pipeline_type} pipeline")
    return True

def run_pipeline(pipeline_type: str, config: Dict[str, Any], **kwargs) -> None:
    """
    Execute the specified pipeline with the given configuration.
    
    Args:
        pipeline_type: Type of pipeline to run
        config: Configuration dictionary
        **kwargs: Additional parameters for pipeline execution
    """
    if pipeline_type not in PIPELINE_MAP:
        available_pipelines = ", ".join(PIPELINE_MAP.keys())
        raise ValueError(f"Unknown pipeline type: {pipeline_type}. Available: {available_pipelines}")
    
    validate_pipeline_config(config, pipeline_type)
    
    pipeline_class = PIPELINE_MAP[pipeline_type]
    
    try:
        logging.info(f"Initializing {pipeline_type} pipeline...")
        
        pipeline = pipeline_class()
        
        pipeline_params = {
            "config": config,
            **kwargs
        }
        
        pipeline_params = {k: v for k, v in pipeline_params.items() if v is not None}
        
        logging.info(f"Starting {pipeline_type} pipeline execution...")
        logging.info(f"Pipeline parameters: {list(pipeline_params.keys())}")
        
        result = pipeline.run(**pipeline_params)
        
        logging.info(f"{pipeline_type} pipeline completed successfully")
        if result:
            logging.info(f"Pipeline result: {result}")
            
    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
        logging.exception("Full traceback:")
        raise

def list_example_configs() -> Dict[str, List[str]]:
    """
    Discover and list available example configurations.
    
    Returns:
        Dictionary mapping pipeline types to lists of example config files
    """
    examples = {}
    docs_dir = Path("input_formats_documentation")
    
    if not docs_dir.exists():
        logging.warning(f"Documentation directory not found: {docs_dir}")
        return examples
    
    for channel_dir in docs_dir.iterdir():
        if channel_dir.is_dir():
            channel_type = channel_dir.name
            examples[channel_type] = []
            
            examples_dir = channel_dir / "examples"
            if examples_dir.exists():
                for example_file in examples_dir.iterdir():
                    if example_file.suffix.lower() in [".yaml", ".yml", ".json", ".txt"]:
                        examples[channel_type].append(str(example_file))
            
            templates_dir = channel_dir / "templates"
            if templates_dir.exists():
                for template_file in templates_dir.iterdir():
                    if template_file.suffix.lower() in [".yaml", ".yml", ".json", ".txt"]:
                        examples[channel_type].append(str(template_file))
    
    return examples

def print_example_configs():
    """Print available example configurations in a formatted way."""
    examples = list_example_configs()
    
    if not examples:
        print("No example configurations found.")
        return
    
    print("Available Example Configurations:")
    print("=" * 40)
    
    for channel_type, config_files in examples.items():
        print(f"\n{channel_type.upper()}:")
        if config_files:
            for config_file in config_files:
                file_path = Path(config_file)
                print(f"  - {file_path.name} ({file_path.suffix})")
        else:
            print("  - No examples found")
    
    print(f"\nTotal: {sum(len(files) for files in examples.values())} example configurations")

def main():
    """Main entry point for the pipeline manager."""
    parser = argparse.ArgumentParser(
        description="Central Pipeline Manager for AI Project Manager App",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list-examples
  %(prog)s --pipeline anime --config input.yaml
  %(prog)s --pipeline superhero --config config.json --output ./output
  %(prog)s --pipeline gaming --config script.txt --base-model stable_diffusion_xl
        """
    )
    
    parser.add_argument(
        "--pipeline", 
        choices=list(PIPELINE_MAP.keys()),
        help="Type of pipeline to execute"
    )
    
    parser.add_argument(
        "--config", 
        help="Path to input configuration file (YAML, JSON, or TXT)"
    )
    
    parser.add_argument(
        "--list-examples", 
        action="store_true",
        help="List available example configurations"
    )
    
    parser.add_argument(
        "--output", 
        help="Output directory for generated content"
    )
    
    parser.add_argument(
        "--base-model", 
        help="Base model to use for generation"
    )
    
    parser.add_argument(
        "--render-fps", 
        type=int, 
        default=24,
        help="Render FPS (default: 24)"
    )
    
    parser.add_argument(
        "--output-fps", 
        type=int, 
        default=24,
        help="Output FPS (default: 24)"
    )
    
    parser.add_argument(
        "--language", 
        default="en",
        help="Language for generation (default: en)"
    )
    
    parser.add_argument(
        "--log-file", 
        help="Log file path (default: pipeline.log)"
    )
    
    parser.add_argument(
        "--log-level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    log_file = args.log_file or "pipeline.log"
    setup_logging(log_file, args.log_level)
    
    try:
        if args.list_examples:
            print_example_configs()
            return
        
        if not args.pipeline:
            parser.error("--pipeline is required (or use --list-examples)")
        
        if not args.config:
            parser.error("--config is required for pipeline execution")
        
        logging.info(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
        
        pipeline_kwargs = {}
        
        if args.output:
            pipeline_kwargs["output_path"] = args.output
        
        if args.base_model:
            pipeline_kwargs["base_model"] = args.base_model
        
        if args.render_fps != 24:
            pipeline_kwargs["render_fps"] = args.render_fps
        
        if args.output_fps != 24:
            pipeline_kwargs["output_fps"] = args.output_fps
        
        if args.language != "en":
            pipeline_kwargs["language"] = args.language
        
        run_pipeline(args.pipeline, config, **pipeline_kwargs)
        
        logging.info("Pipeline manager execution completed successfully")
        
    except KeyboardInterrupt:
        logging.info("Pipeline execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Pipeline manager failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
>>>>>>> devin/1750013487-central-pipeline-manager
