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
