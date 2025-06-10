# AI Manager App - Input Formats Documentation

This directory contains comprehensive documentation for all supported input formats across the 6 channel pipelines in the AI Manager App.

## Channel Types Overview

| Channel | Primary Formats | Input Types | Special Features |
|---------|----------------|-------------|------------------|
| **Gaming** | JSON, TXT, Video Files | Game recordings, highlight scripts | Video processing, automatic editing |
| **Anime** | YAML, JSON | Character-driven stories, episodes | Multi-character support, voice synthesis |
| **Manga** | YAML, JSON, TXT | Scene descriptions, panels | Panel-based generation, reading flow |
| **Marvel/DC** | TXT, JSON | Superhero storylines, comics | Character powers, universe consistency |
| **Superhero** | JSON | Original superhero content | Custom powers, origin stories |
| **Original Manga** | YAML | Original universe creation | World-building, character design |

## Directory Structure

```
input_formats_documentation/
├── README.md (this file)
├── gaming/
│   ├── formats.md
│   ├── examples/
│   └── templates/
├── anime/
│   ├── formats.md
│   ├── examples/
│   └── templates/
├── manga/
│   ├── formats.md
│   ├── examples/
│   └── templates/
├── marvel_dc/
│   ├── formats.md
│   ├── examples/
│   └── templates/
├── superhero/
│   ├── formats.md
│   ├── examples/
│   └── templates/
└── original_manga/
    ├── formats.md
    ├── examples/
    └── templates/
```

## Universal Input Requirements

All channels support these common parameters:

- **Language Selection**: English, Dutch, German, French, Chinese, Japanese, Spanish
- **Quality Settings**: Low, Medium, High, Ultra (based on VRAM)
- **Output Resolution**: 1080p, 1440p, 4K (with RealESRGAN upscaling)
- **Aspect Ratio**: 16:9 (enforced across all pipelines)
- **Duration**: Configurable per scene/episode

## Getting Started

1. Choose your channel type based on content
2. Review the specific format documentation in the channel folder
3. Use the provided templates as starting points
4. Customize the input according to your project needs
5. Run the pipeline through the AI Manager App GUI or API

## API Integration

All input formats can be submitted via:
- **GUI**: Project creation dialog with format validation
- **API**: `/api/projects/create` endpoint with JSON payload
- **CLI**: Direct pipeline execution with file input

For detailed format specifications, see the individual channel documentation files.
