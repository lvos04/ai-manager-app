# Marvel/DC Channel - Input Formats

The Marvel/DC channel creates superhero content in the style of established comic universes with iconic characters, epic storylines, and classic comic book aesthetics.

## Supported Input Formats

### 1. TXT Format (Primary)
**Purpose**: Narrative-driven superhero stories

```
Title: The Last Stand of Earth's Mightiest Heroes

Setting: New York City under siege by cosmic forces

Act 1: The Threat Emerges
The sky tears open above Manhattan as interdimensional invaders pour through. Citizens flee in panic as buildings crumble under alien weaponry.

Character: Captain Marvel
Description: Blonde hair flowing, red and blue costume gleaming
Action: Flies through the portal, energy crackling around her fists
Dialogue: "Not on my watch! Earth is under my protection!"

Character: Spider-Man
Description: Classic red and blue suit, web-shooters ready
Action: Swings between buildings, evacuating civilians
Dialogue: "Hey ugly! Pick on someone with radioactive spider powers!"

Scene: Epic Battle Sequence
Duration: 180 seconds
The heroes coordinate their attack. Captain Marvel unleashes photon blasts while Spider-Man uses his agility to outmaneuver the invaders. The city becomes a battlefield of light and shadow.

Act 2: The Sacrifice
When all seems lost, the heroes must make the ultimate choice...
```

### 2. JSON Format
**Purpose**: Structured superhero universe content

```json
{
  "title": "Crisis on Infinite Earths",
  "universe": "DC",
  "threat_level": "cosmic",
  "heroes": [
    {
      "name": "Superman",
      "alias": "Clark Kent",
      "powers": ["super_strength", "flight", "heat_vision", "invulnerability"],
      "costume": "blue suit, red cape, S-shield",
      "personality": "noble, hopeful, determined"
    },
    {
      "name": "Batman",
      "alias": "Bruce Wayne", 
      "powers": ["genius_intellect", "martial_arts", "detective_skills"],
      "costume": "dark gray suit, black cape, bat symbol",
      "personality": "brooding, strategic, relentless"
    }
  ],
  "storyline": {
    "act_1": {
      "title": "The Gathering Storm",
      "scenes": [
        {
          "description": "Metropolis under attack by Brainiac's forces",
          "heroes": ["Superman"],
          "duration": 240,
          "action_level": "high"
        }
      ]
    },
    "act_2": {
      "title": "United We Stand",
      "scenes": [
        {
          "description": "Heroes unite at the Watchtower",
          "heroes": ["Superman", "Batman", "Wonder Woman"],
          "duration": 180,
          "action_level": "medium"
        }
      ]
    }
  }
}
```

### 3. Comic Script Format
**Purpose**: Traditional comic book script structure

```
PAGE 1

PANEL 1 (Splash Page)
ESTABLISHING SHOT of Gotham City at night. Rain falls on empty streets. The Bat-Signal cuts through the darkness.
CAPTION: "In the shadows of Gotham, evil never sleeps..."

PAGE 2

PANEL 1 (Wide shot)
BATMAN perches on a gargoyle overlooking the city. His cape billows in the wind.
BATMAN (thought): "Three weeks since the Joker escaped. Too quiet."

PANEL 2 (Close-up)
Batman's cowled face, eyes narrowed with determination.
BATMAN (thought): "He's planning something big."

PANEL 3 (Medium shot)
Batman's utility belt as he reaches for a communication device.
SFX: BEEP BEEP

PANEL 4 (Close-up on device)
Oracle's voice crackles through the comm.
ORACLE (radio): "Batman, we have a situation at Ace Chemicals."
```

## Character Archetypes

### Marvel Heroes
```yaml
marvel_characters:
  - name: "Iron Man"
    type: "tech_hero"
    powers: ["powered_armor", "genius_intellect", "flight"]
    personality: "witty, arrogant, heroic"
    visual_style: "red and gold armor, arc reactor"
  
  - name: "Captain America"
    type: "super_soldier"
    powers: ["enhanced_strength", "shield_mastery", "leadership"]
    personality: "noble, patriotic, inspiring"
    visual_style: "star-spangled uniform, vibranium shield"
```

### DC Heroes
```yaml
dc_characters:
  - name: "Wonder Woman"
    type: "mythological_hero"
    powers: ["super_strength", "flight", "lasso_of_truth"]
    personality: "compassionate, warrior, diplomatic"
    visual_style: "golden armor, tiara, bracers"
  
  - name: "The Flash"
    type: "speedster"
    powers: ["super_speed", "time_travel", "speed_force"]
    personality: "optimistic, scientific, quick-witted"
    visual_style: "red suit, lightning bolt emblem"
```

## Story Structure Templates

### Classic Superhero Arc
```yaml
story_structure:
  setup:
    - introduce_hero: "Establish character in normal life"
    - inciting_incident: "Threat emerges"
    - call_to_action: "Hero must respond"
  
  confrontation:
    - first_encounter: "Initial battle with villain"
    - setback: "Hero faces defeat or loss"
    - preparation: "Hero trains or gathers allies"
  
  resolution:
    - final_battle: "Epic confrontation"
    - sacrifice: "Hero makes difficult choice"
    - victory: "Threat defeated, order restored"
```

### Team-Up Format
```yaml
team_story:
  assembly:
    - individual_threats: "Each hero faces separate challenges"
    - convergence: "Threats revealed to be connected"
    - first_meeting: "Heroes encounter each other"
  
  cooperation:
    - initial_conflict: "Heroes clash due to different methods"
    - common_ground: "Shared goal identified"
    - strategy: "Plan formulated together"
  
  triumph:
    - coordinated_attack: "Heroes work as team"
    - individual_moments: "Each hero gets spotlight"
    - united_victory: "Combined powers save the day"
```

## Visual Style Guidelines

### Marvel Style
- **Color Palette**: Bright, vibrant colors
- **Art Style**: Dynamic, action-oriented
- **Composition**: Cinematic angles, dramatic lighting
- **Character Design**: Realistic proportions with heroic idealization

### DC Style
- **Color Palette**: Bold primary colors, high contrast
- **Art Style**: Iconic, mythological grandeur
- **Composition**: Symmetrical, powerful poses
- **Character Design**: Statuesque, god-like proportions

## Power System Integration

### Power Classifications
```yaml
power_types:
  physical:
    - super_strength: "Enhanced physical power"
    - super_speed: "Accelerated movement"
    - invulnerability: "Resistance to damage"
  
  energy:
    - energy_projection: "Beams, blasts, constructs"
    - energy_absorption: "Power from external sources"
    - force_fields: "Protective barriers"
  
  mental:
    - telepathy: "Mind reading and communication"
    - telekinesis: "Moving objects with mind"
    - precognition: "Seeing future events"
  
  technological:
    - powered_armor: "Mechanical enhancement"
    - gadgets: "Specialized tools and weapons"
    - ai_assistance: "Computer-aided abilities"
```

## Processing Pipeline

1. **Universe Detection**: Identify Marvel vs DC elements
2. **Character Analysis**: Extract hero/villain profiles
3. **Power Visualization**: Create appropriate power effects
4. **Scene Generation**: Build comic book style scenes
5. **Action Choreography**: Design dynamic fight sequences
6. **Dialogue Integration**: Add heroic speeches and banter
7. **Comic Assembly**: Combine into comic book format

## Output Structure

```
output/
├── final/
│   ├── superhero_comic.mp4 (animated comic)
│   ├── superhero_comic_upscaled.mp4
│   └── heroic_soundtrack.wav
├── pages/
│   ├── page_001_splash.jpg
│   ├── page_002_action.jpg
│   └── page_003_resolution.jpg
├── characters/
│   ├── hero_profiles/
│   ├── villain_profiles/
│   └── power_demonstrations/
├── scenes/
│   ├── scene_001_origin.mp4
│   ├── scene_002_conflict.mp4
│   └── scene_003_triumph.mp4
└── shorts/
    ├── hero_showcase_01.mp4
    ├── power_demonstration_02.mp4
    └── epic_moments_03.mp4
```

## Best Practices

1. **Character Consistency**: Maintain established character traits
2. **Power Logic**: Keep abilities consistent with established rules
3. **Heroic Themes**: Emphasize hope, justice, and sacrifice
4. **Visual Impact**: Create memorable, iconic moments
5. **Dialogue Style**: Use heroic, inspiring language
6. **Pacing**: Balance action with character development
7. **Universe Respect**: Honor established comic book traditions
