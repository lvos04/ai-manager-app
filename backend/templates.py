"""
Project template system for different channel types.
"""

def get_project_template(channel_type):
    """Get input template for a specific channel type."""
    templates = {
        "gaming": {
            "name": "Gaming YouTube Channel Template",
            "description": "Template for gaming video projects",
            "input_format": "JSON or TXT",
            "example_json": {
                "scenes": [
                    "Epic boss battle with dramatic lighting and realistic characters",
                    "Character exploring detailed game environment with atmospheric effects",
                    "Action sequence with dynamic camera movements and particle effects"
                ],
                "voice_style": "narrator",
                "music_genre": "epic orchestral"
            },
            "example_txt": "Scene 1: Epic boss battle with dramatic lighting\n\nScene 2: Character exploring environment\n\nScene 3: Action sequence with effects",
            "instructions": [
                "Provide either JSON with 'scenes' array or TXT with scenes separated by double newlines",
                "Each scene should describe the visual content you want",
                "Focus on realistic, gaming-appropriate descriptions",
                "Optional: Add voice_style and music_genre preferences"
            ]
        },
        "anime": {
            "name": "AI Anime Series Template", 
            "description": "Template for anime episode projects",
            "input_format": "YAML or JSON",
            "example_yaml": """characters:
  - name: "Protagonist"
    description: "Anime girl with colorful hair and determined expression"
    voice_profile: "young_female_japanese"
  - name: "Sidekick"
    description: "Anime boy with spiky hair and magical outfit"
    voice_profile: "young_male_japanese"

scenes:
  - dialogue: "We must save the world!"
    character: "Protagonist"
    location: "magical forest"
  - dialogue: "I'll help you!"
    character: "Sidekick" 
    location: "magical forest"
""",
            "instructions": [
                "Use YAML format with characters and scenes sections",
                "Define each character with name, description, and voice_profile",
                "Each scene needs dialogue, character, and location",
                "Minimum 20 pages worth of content recommended"
            ]
        },
        "marvel_dc": {
            "name": "Marvel/DC Summary Template",
            "description": "Template for comic book summary videos", 
            "input_format": "TXT or JSON",
            "example_txt": "Title: The Ultimate Superhero Battle\n\nSummary: Original superhero characters face off in an epic confrontation that spans multiple dimensions.\n\nScene 1: Hero discovers their powers\nScene 2: Villain emerges with dark plan\nScene 3: Epic final battle",
            "instructions": [
                "Provide original storyline summary (no real Marvel/DC content)",
                "Include title, summary, and scene descriptions",
                "Focus on comic book visual style elements",
                "Keep content family-friendly"
            ]
        },
        "manga": {
            "name": "Manga Channel Template",
            "description": "Template for manga-style videos",
            "input_format": "TXT or JSON", 
            "example_txt": "Manga Story: The Student's Journey\n\nScene 1: Student character in school uniform, black and white manga style\nScene 2: Dramatic revelation scene with speed lines\nScene 3: Emotional conclusion with detailed character expressions",
            "instructions": [
                "Describe scenes in manga visual style",
                "Focus on black and white aesthetic", 
                "Include typical manga elements like speed lines, dramatic angles",
                "Character expressions should be detailed and emotive"
            ]
        },
        "superhero": {
            "name": "Original Superhero Universe Template",
            "description": "Template for original superhero content",
            "input_format": "JSON",
            "example_json": {
                "universe": "Original Superhero Universe",
                "characters": [
                    {
                        "name": "Captain Lightning",
                        "powers": "electricity manipulation",
                        "costume": "blue and silver suit with lightning motifs"
                    }
                ],
                "scenes": [
                    "Captain Lightning discovers their powers in city rooftop scene",
                    "Epic battle against shadow villain in urban environment", 
                    "Heroes unite for final confrontation"
                ]
            },
            "instructions": [
                "Create original superhero characters and universe",
                "Define character powers, costumes, and personalities",
                "Build epic storylines with clear progression",
                "Focus on unique visual design elements"
            ]
        },
        "original_manga": {
            "name": "Original Manga Universe Template",
            "description": "Template for original manga universe content",
            "input_format": "YAML",
            "example_yaml": """universe: "Original Manga World"
style: "black_and_white_detailed"

characters:
  - name: "Akira"
    description: "Determined student with unique abilities"
    design_notes: "spiky black hair, school uniform, determined eyes"

scenes:
  - description: "Akira discovers hidden powers in school courtyard"
    panel_style: "dramatic close-up with speed lines"
  - description: "Training montage in mysterious location"
    panel_style: "multiple panels showing progression"
""",
            "instructions": [
                "Use YAML format for structured world-building",
                "Define unique manga universe and characters", 
                "Specify visual style and panel composition",
                "Include design notes for consistent character art"
            ]
        }
    }
    
    return templates.get(channel_type, None)

def get_all_templates():
    """Get all available project templates."""
    channel_types = ["gaming", "anime", "marvel_dc", "manga", "superhero", "original_manga"]
    return {channel: get_project_template(channel) for channel in channel_types}
