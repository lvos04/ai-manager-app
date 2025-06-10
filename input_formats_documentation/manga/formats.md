# Manga Channel - Input Formats

The Manga channel creates manga-style content with panel-based storytelling, black and white aesthetics, and traditional manga reading flow.

## Supported Input Formats

### 1. YAML Format (Recommended)
**Purpose**: Structured manga chapters with panel layouts

```yaml
title: "Urban Legends Chapter 1"
chapter_number: 1
pages: 20
style: "shounen_manga"
reading_direction: "right_to_left"

characters:
  - name: "Kenji"
    description: "High school detective"
    design: "messy black hair, sharp eyes, school uniform"
  - name: "Rei"
    description: "Mysterious transfer student"
    design: "long silver hair, gothic clothing"

pages:
  - page_number: 1
    panels:
      - panel: 1
        size: "full_page"
        description: "Establishing shot of Tokyo cityscape at night"
        text: "In the heart of Tokyo, legends come alive..."
        type: "establishing"
      
  - page_number: 2
    panels:
      - panel: 1
        size: "large"
        description: "Close-up of Kenji looking determined"
        dialogue:
          - speaker: "Kenji"
            text: "Another strange case... but this time it's different."
        emotion: "determined"
      
      - panel: 2
        size: "medium"
        description: "Rei appears in the shadows"
        dialogue:
          - speaker: "Rei"
            text: "You're getting closer to the truth."
        emotion: "mysterious"
      
      - panel: 3
        size: "small"
        description: "Kenji's surprised reaction"
        sound_effect: "GASP!"
        emotion: "shocked"
```

### 2. JSON Format
**Purpose**: Programmatic manga creation

```json
{
  "title": "School Mystery Manga",
  "genre": "mystery",
  "art_style": "shounen",
  "chapter": 1,
  "pages": [
    {
      "page_number": 1,
      "layout": "4_panel_vertical",
      "panels": [
        {
          "description": "School hallway, empty and eerie",
          "text": "After school, when everyone has gone home...",
          "mood": "suspenseful"
        },
        {
          "description": "Footsteps echoing in the corridor",
          "sound_effect": "TAP TAP TAP",
          "mood": "tense"
        }
      ]
    }
  ]
}
```

### 3. Script Format (TXT)
**Purpose**: Traditional manga script format

```
PAGE 1

PANEL 1 (Full page splash)
DESCRIPTION: Wide shot of futuristic city with flying cars and neon signs
CAPTION: "Neo-Tokyo, 2087. Where technology and tradition collide."

PAGE 2

PANEL 1 (Large, top half)
DESCRIPTION: Close-up of protagonist AKIRA (17) with cybernetic eye implant
AKIRA: "The data doesn't lie. Someone's been tampering with the city's AI."
EMOTION: Serious, determined

PANEL 2 (Medium, bottom left)
DESCRIPTION: Holographic display showing corrupted code
SFX: BZZT! BZZT!

PANEL 3 (Medium, bottom right)
DESCRIPTION: Akira's shocked expression as the display flickers
AKIRA: "This is worse than I thought..."
EMOTION: Worried
```

## Panel Layout System

### Standard Panel Sizes
- **full_page**: Single panel covering entire page (splash page)
- **large**: Takes up 50-70% of page space
- **medium**: Standard panel size, 25-40% of page
- **small**: Detail or reaction panel, 10-25% of page
- **thin**: Horizontal strip panel for transitions

### Layout Templates
```yaml
layouts:
  4_panel_vertical:
    - panel_1: large
    - panel_2: medium
    - panel_3: medium
    - panel_4: small
  
  6_panel_grid:
    - row_1: [medium, medium]
    - row_2: [small, large]
    - row_3: [medium, medium]
  
  action_sequence:
    - panel_1: large (setup)
    - panel_2: thin (motion)
    - panel_3: medium (impact)
    - panel_4: small (reaction)
```

## Manga Art Styles

- **shounen**: Action-oriented, dynamic poses, bold lines
- **shoujo**: Romance-focused, delicate lines, floral elements
- **seinen**: Mature themes, detailed artwork, realistic proportions
- **josei**: Adult female audience, sophisticated art style
- **kodomomuke**: Children's manga, simple and cute designs
- **horror**: Dark atmosphere, detailed shadows, unsettling imagery

## Text and Typography

### Dialogue Formatting
```yaml
dialogue:
  - speaker: "Character Name"
    text: "What they say"
    bubble_type: "speech/thought/shout/whisper"
    position: "top_left/center/bottom_right"
    font_size: "normal/large/small"
```

### Sound Effects (SFX)
```yaml
sound_effects:
  - text: "CRASH!"
    style: "bold_impact"
    position: "center"
  - text: "whoosh..."
    style: "motion_blur"
    position: "following_action"
```

### Text Types
- **Speech Bubbles**: Regular dialogue
- **Thought Bubbles**: Internal monologue
- **Narration Boxes**: Story exposition
- **Sound Effects**: Action sounds
- **Captions**: Time/place indicators

## Reading Flow

### Right-to-Left (Traditional Japanese)
```yaml
reading_direction: "right_to_left"
panel_order: [top_right, top_left, middle_right, middle_left, bottom_right, bottom_left]
```

### Left-to-Right (Western Style)
```yaml
reading_direction: "left_to_right"
panel_order: [top_left, top_right, middle_left, middle_right, bottom_left, bottom_right]
```

## Processing Pipeline

1. **Script Analysis**: Parse pages, panels, and dialogue
2. **Layout Generation**: Create panel arrangements
3. **Character Design**: Generate consistent manga-style characters
4. **Background Creation**: Draw detailed manga backgrounds
5. **Panel Assembly**: Combine elements into manga pages
6. **Text Integration**: Add dialogue, SFX, and narration
7. **Page Compilation**: Assemble complete manga chapter

## Output Structure

```
output/
├── final/
│   ├── manga_chapter.pdf (complete chapter)
│   ├── manga_chapter_web.jpg (web-optimized pages)
│   └── reading_guide.json
├── pages/
│   ├── page_001.jpg
│   ├── page_002.jpg
│   └── page_020.jpg
├── panels/
│   ├── page01_panel01.jpg
│   ├── page01_panel02.jpg
│   └── page01_panel03.jpg
├── characters/
│   ├── character_sheets/
│   └── reference_poses/
└── shorts/
    ├── manga_preview_01.mp4 (animated preview)
    ├── manga_preview_02.mp4 (character showcase)
    └── manga_preview_03.mp4 (action highlights)
```

## Special Features

- **Panel Transitions**: Smooth flow between panels
- **Action Lines**: Dynamic motion indicators
- **Screen Tones**: Traditional manga shading patterns
- **Character Consistency**: Maintain character designs across panels
- **Perspective Variety**: Dynamic camera angles and viewpoints
- **Emotional Expression**: Exaggerated manga-style emotions

## Best Practices

1. **Panel Pacing**: Vary panel sizes for rhythm and emphasis
2. **Visual Hierarchy**: Guide reader's eye through the page
3. **Character Expressions**: Use exaggerated emotions effectively
4. **Background Detail**: Balance detail with readability
5. **Text Placement**: Ensure dialogue doesn't obscure important art
6. **Action Sequences**: Use multiple panels to show motion
7. **Page Turns**: End pages with cliffhangers or reveals
