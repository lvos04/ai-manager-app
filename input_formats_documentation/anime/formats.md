# Anime Channel - Input Formats

The Anime channel creates anime-style content with character-driven stories, voice synthesis, and traditional anime aesthetics.

## Supported Input Formats

### 1. YAML Format (Recommended)
**Purpose**: Structured anime episodes with characters and scenes

```yaml
title: "Magical Academy Episode 1"
episode_number: 1
duration: 1440  # 24 minutes
style: "modern anime"

characters:
  - name: "Akira"
    description: "Main protagonist, fire magic user"
    voice_type: "young_male"
    appearance: "spiky red hair, determined eyes"
  - name: "Yuki"
    description: "Ice magic specialist, calm personality"
    voice_type: "soft_female"
    appearance: "long blue hair, elegant demeanor"

scenes:
  - number: 1
    description: "Academy entrance ceremony"
    duration: 180
    characters: ["Akira", "Yuki"]
    dialogue:
      - speaker: "Akira"
        text: "This is it! My journey begins here!"
      - speaker: "Yuki"
        text: "The academy looks even more impressive than I imagined."
    setting: "Grand hall with magical floating crystals"
    
  - number: 2
    description: "First magic class demonstration"
    duration: 240
    characters: ["Akira"]
    action: "Akira attempts fire magic spell"
    setting: "Training grounds with practice dummies"
```

### 2. JSON Format
**Purpose**: Programmatic anime content creation

```json
{
  "title": "Slice of Life Adventure",
  "genre": "slice_of_life",
  "art_style": "traditional_anime",
  "characters": [
    {
      "name": "Hana",
      "role": "protagonist",
      "personality": "cheerful, optimistic",
      "design_notes": "pink hair, school uniform"
    }
  ],
  "episodes": [
    {
      "title": "New School Day",
      "scenes": [
        {
          "description": "Morning routine in traditional Japanese home",
          "mood": "peaceful",
          "duration": 120
        }
      ]
    }
  ]
}
```

### 3. Script Format (TXT)
**Purpose**: Traditional screenplay format

```
FADE IN:

EXT. MAGIC ACADEMY - DAY

A magnificent castle-like structure floats among the clouds. Students in robes fly on magical creatures toward the entrance.

AKIRA (16), determined with spiky red hair, lands clumsily on the platform.

AKIRA
(brushing off robes)
Finally made it! Time to become the greatest mage ever!

YUKI (16), graceful with flowing blue hair, lands elegantly nearby.

YUKI
(smiling softly)
Your enthusiasm is admirable, but perhaps focus on not falling first?

CUT TO:

INT. ACADEMY HALL - CONTINUOUS

Hundreds of new students gather. Floating magical orbs provide light.
```

## Character System

### Character Definition
```yaml
characters:
  - name: "Character Name"
    age: 16
    personality: "brave, loyal, hot-headed"
    appearance: "detailed visual description"
    voice_type: "young_male/young_female/mature_male/mature_female"
    magical_abilities: ["fire_magic", "sword_combat"]
    relationships:
      - character: "Other Character"
        relationship: "best_friend"
    backstory: "Character history and motivation"
```

### Voice Synthesis Options
- **young_male**: Energetic, higher pitch
- **young_female**: Sweet, expressive
- **mature_male**: Deep, authoritative
- **mature_female**: Warm, sophisticated
- **child**: High-pitched, innocent
- **elderly**: Wise, weathered

## Scene Structure

### Basic Scene Format
```yaml
scenes:
  - number: 1
    title: "Scene Title"
    description: "Visual description of what happens"
    duration: 180  # seconds
    setting: "Location and environment details"
    mood: "happy/sad/tense/peaceful/action"
    characters: ["Character1", "Character2"]
    dialogue: [...]
    action_sequences: [...]
    camera_notes: "Close-up, wide shot, etc."
```

### Dialogue System
```yaml
dialogue:
  - speaker: "Character Name"
    text: "What they say"
    emotion: "happy/sad/angry/surprised"
    voice_direction: "whispered/shouted/normal"
  - speaker: "Narrator"
    text: "Narration text"
    type: "narration"
```

## Art Style Options

- **traditional_anime**: Classic 2D anime aesthetic
- **modern_anime**: Contemporary digital anime style
- **chibi**: Super-deformed cute style
- **realistic_anime**: More detailed, realistic proportions
- **retro_anime**: 80s/90s anime style
- **manga_style**: Black and white manga aesthetic

## Processing Pipeline

1. **Script Parsing**: Extract characters, scenes, dialogue
2. **Character Generation**: Create consistent character designs
3. **Scene Creation**: Generate anime-style backgrounds and action
4. **Voice Synthesis**: Generate character voices using Bark/XTTS
5. **Lip Sync**: Sync character mouth movements (DreamTalk for anime)
6. **Music Generation**: Create anime-style background music
7. **Final Assembly**: Combine all elements with transitions

## Output Structure

```
output/
├── final/
│   ├── anime_episode.mp4
│   ├── anime_episode_upscaled.mp4
│   └── background_music.wav
├── characters/
│   ├── character_akira.png
│   ├── character_yuki.png
│   └── character_reference_sheet.json
├── scenes/
│   ├── scene_001_academy_entrance.mp4
│   ├── scene_002_magic_class.mp4
│   └── scene_003_friendship_moment.mp4
├── audio/
│   ├── voice_akira_scene01.wav
│   ├── voice_yuki_scene01.wav
│   └── narration_scene01.wav
└── shorts/
    ├── anime_short_01.mp4 (highlight reel)
    ├── anime_short_02.mp4 (character moments)
    └── anime_short_03.mp4 (action sequences)
```

## Best Practices

1. **Character Consistency**: Define characters thoroughly before scenes
2. **Dialogue Balance**: Mix dialogue with action and visual storytelling
3. **Pacing**: Vary scene lengths for dynamic storytelling
4. **Visual Descriptions**: Provide detailed setting and action descriptions
5. **Emotional Arc**: Plan character development across scenes
6. **Cultural Elements**: Include authentic Japanese cultural elements when appropriate
