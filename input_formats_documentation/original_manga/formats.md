# Original Manga Channel - Input Formats

The Original Manga channel creates completely original manga universes with unique world-building, character designs, and storytelling elements not bound by existing franchises.

## Supported Input Formats

### 1. YAML Format (Primary)
**Purpose**: Comprehensive universe creation with detailed world-building

```yaml
universe: "Neon Dreams Academy"
title: "Chapter 1: Digital Awakening"
style: "cyberpunk_manga"
art_style: "detailed_black_and_white"
reading_direction: "right_to_left"

world_setting:
  location: "Neo-Tokyo Academy 2087"
  technology_level: "advanced_ai_integration"
  society: "corporate_controlled_education"
  atmosphere: "neon_lit_urban_decay"

main_characters:
  - name: "Akira Netrunner"
    age: 16
    role: "protagonist"
    background: "Talented hacker from the slums"
    personality: "rebellious, intelligent, loyal"
    appearance: "neon blue hair, cybernetic eye implant, modified school uniform"
    abilities: ["advanced_hacking", "digital_empathy", "code_visualization"]
    
  - name: "Rei Shadowbyte"
    age: 17
    role: "mysterious_ally"
    background: "Corporate heir with hidden agenda"
    personality: "calculating, protective, conflicted"
    appearance: "silver hair, elegant posture, high-tech accessories"
    abilities: ["corporate_access", "social_manipulation", "encrypted_communication"]

story_structure:
  theme: "technology_vs_humanity"
  conflict: "corporate_control_vs_individual_freedom"
  setting_rules:
    - "AI monitors all student activities"
    - "Hacking skills determine social status"
    - "Corporate families control curriculum"

scenes:
  - scene_number: 1
    description: "Akira discovers the academy's hidden surveillance network"
    panel_layout: "wide_establishing_shot"
    mood: "suspenseful_discovery"
    dialogue:
      - speaker: "Akira"
        text: "These data streams... they're watching everything we do."
        emotion: "shocked_realization"
    
  - scene_number: 2
    description: "First encounter with Rei in the digital commons"
    panel_layout: "character_introduction_sequence"
    mood: "mysterious_meeting"
    dialogue:
      - speaker: "Rei"
        text: "You're not the only one who sees through their facade."
        emotion: "cryptic_warning"
      - speaker: "Akira"
        text: "Who are you? How do you know about my hacking?"
        emotion: "suspicious_curiosity"

visual_style:
  panel_types: ["splash_page", "action_sequence", "dialogue_close_up", "environmental_wide"]
  effects: ["digital_glitch", "neon_glow", "holographic_displays", "data_streams"]
  backgrounds: ["cyberpunk_cityscape", "high_tech_classroom", "underground_server_room"]
```

### 2. JSON Format
**Purpose**: Structured original manga creation with detailed metadata

```json
{
  "title": "Spirit Walker Chronicles",
  "universe": "Modern Supernatural Japan",
  "genre": "supernatural_action",
  "art_style": "traditional_manga_with_spirit_effects",
  "chapter": 1,
  "pages": 20,
  
  "world_building": {
    "setting": "Modern Japan with hidden spirit realm",
    "magic_system": "spiritual_energy_manipulation",
    "rules": [
      "Spirits invisible to most humans",
      "Spirit walkers can cross between realms",
      "Emotional state affects spiritual power"
    ]
  },
  
  "characters": [
    {
      "name": "Yuki Spiritwalker",
      "age": 15,
      "role": "protagonist",
      "background": "Ordinary student who awakens spirit sight",
      "personality": "compassionate, determined, initially fearful",
      "appearance": "short black hair, traditional school uniform, spirit mark on forehead when powers active",
      "powers": [
        {
          "name": "Spirit Sight",
          "description": "Can see and interact with spirits",
          "visual_effect": "glowing eyes, ethereal aura"
        },
        {
          "name": "Realm Walking",
          "description": "Travel between physical and spirit worlds",
          "visual_effect": "dimensional ripples, fading transparency"
        }
      ]
    },
    {
      "name": "Kage the Shadow Guide",
      "role": "mentor_spirit",
      "background": "Ancient spirit bound to protect spirit walkers",
      "personality": "wise, protective, mysterious past",
      "appearance": "wolf-like shadow creature with glowing eyes"
    }
  ],
  
  "story_arcs": [
    {
      "arc_name": "Awakening",
      "chapters": [1, 2, 3],
      "description": "Yuki discovers her powers and learns about the spirit realm"
    }
  ],
  
  "scenes": [
    {
      "page": 1,
      "panels": [
        {
          "description": "Yuki walking home from school, normal day",
          "panel_size": "medium",
          "mood": "peaceful_ordinary"
        },
        {
          "description": "Strange shadows moving in peripheral vision",
          "panel_size": "small",
          "mood": "subtle_unease"
        },
        {
          "description": "First clear spirit sighting - ghostly figure",
          "panel_size": "large",
          "mood": "shocking_revelation"
        }
      ]
    }
  ]
}
```

### 3. TXT Format
**Purpose**: Simple scene-based original manga creation

```
Original Manga: The Clockwork Rebellion

Universe: Steampunk Victorian Era with Mechanical Uprising
Style: Detailed mechanical art with Victorian aesthetics

Character: Gearwright Eliza
- Young inventor fighting against oppressive Steam Guild
- Appearance: Goggles, leather apron, mechanical arm prosthetic
- Personality: Brilliant, rebellious, compassionate to machines

Scene 1: Eliza's workshop - Creating sentient automaton
Panel Style: Close-up of intricate gear work with dramatic lighting
Dialogue: "If they won't treat machines with respect, I'll give them consciousness to demand it."

Scene 2: Steam Guild raid on workshop
Panel Style: Dynamic action sequence with steam and mechanical effects
Action: Eliza and her automatons fighting Guild enforcers

Scene 3: Escape through underground steam tunnels
Panel Style: Wide environmental shots showing vast underground network
Mood: Tense chase sequence with steam-powered vehicles

Scene 4: Meeting with other rebels in hidden base
Panel Style: Character interaction panels with detailed mechanical backgrounds
Theme: Building alliance between humans and conscious machines

Visual Elements:
- Intricate gear and clockwork details
- Steam effects and industrial atmosphere
- Victorian clothing with mechanical modifications
- Contrast between organic and mechanical forms
```

## Universe Creation Framework

### World Building Elements
```yaml
universe_components:
  setting:
    time_period: "historical/modern/futuristic/fantasy"
    location: "specific_geographic_or_fictional_setting"
    technology_level: "primitive/industrial/modern/advanced/magical"
    society_structure: "feudal/democratic/corporate/anarchist/tribal"
    
  magic_system:
    power_source: "spiritual/technological/genetic/divine/elemental"
    limitations: "energy_cost/emotional_state/physical_toll/moral_restrictions"
    visual_effects: "glowing/particle_effects/environmental_changes"
    
  cultural_elements:
    language: "naming_conventions_and_terminology"
    customs: "social_norms_and_traditions"
    conflicts: "historical_tensions_and_current_disputes"
    values: "what_society_considers_important"
```

### Character Archetypes
```yaml
character_roles:
  protagonist_types:
    - reluctant_hero: "Ordinary person thrust into extraordinary circumstances"
    - chosen_one: "Destined individual with special powers or purpose"
    - rebel: "Fighting against oppressive system or authority"
    - seeker: "Searching for truth, power, or lost knowledge"
    
  supporting_roles:
    - mentor: "Wise guide who teaches protagonist"
    - rival: "Competitive equal who challenges protagonist"
    - ally: "Loyal friend who supports the journey"
    - love_interest: "Romantic connection that complicates story"
    
  antagonist_types:
    - tyrant: "Oppressive ruler seeking control"
    - corruptor: "Seeks to corrupt or destroy what is good"
    - mirror: "Dark reflection of protagonist's potential"
    - force_of_nature: "Unstoppable natural or supernatural threat"
```

## Art Style Guidelines

### Visual Aesthetics
```yaml
art_styles:
  traditional_manga:
    characteristics: ["black_and_white", "detailed_backgrounds", "expressive_characters"]
    panel_layouts: ["varied_sizes", "dynamic_angles", "speed_lines"]
    
  modern_digital:
    characteristics: ["clean_lines", "digital_effects", "consistent_proportions"]
    special_effects: ["glowing_elements", "particle_systems", "gradient_backgrounds"]
    
  experimental:
    characteristics: ["mixed_media", "unconventional_panels", "artistic_expression"]
    techniques: ["watercolor_effects", "sketch_style", "photorealistic_elements"]
```

### Panel Composition
```yaml
panel_types:
  establishing:
    purpose: "Set scene and mood"
    size: "large_or_full_page"
    content: "wide_environmental_shots"
    
  action:
    purpose: "Show movement and conflict"
    size: "medium_to_large"
    content: "dynamic_poses_with_motion_lines"
    
  dialogue:
    purpose: "Character interaction and development"
    size: "medium"
    content: "character_close_ups_with_speech_bubbles"
    
  reaction:
    purpose: "Emotional response and pacing"
    size: "small_to_medium"
    content: "facial_expressions_and_body_language"
    
  transition:
    purpose: "Bridge between scenes or time"
    size: "thin_horizontal_or_small"
    content: "symbolic_imagery_or_time_passage"
```

## Processing Pipeline

1. **Universe Analysis**: Extract world-building elements, rules, and themes
2. **Character Design**: Create unique, memorable character designs with consistent visual identity
3. **World Visualization**: Build detailed, consistent environments that support the story
4. **Story Structure**: Organize narrative flow, pacing, and character development arcs
5. **Art Style Application**: Apply chosen aesthetic consistently across all visual elements
6. **Panel Layout**: Create manga-style panel compositions that enhance storytelling
7. **Text Integration**: Add dialogue, narration, and sound effects with proper typography
8. **Final Assembly**: Combine all elements into cohesive manga chapters

## Output Structure

```
output/
├── final/
│   ├── original_manga_chapter.pdf (complete chapter)
│   ├── original_manga_web_format.jpg (web-optimized pages)
│   ├── universe_bible.txt (world-building reference)
│   └── character_reference_sheet.pdf
├── pages/
│   ├── page_001.jpg
│   ├── page_002.jpg
│   └── page_020.jpg
├── panels/
│   ├── page01_panel01.jpg
│   ├── page01_panel02.jpg
│   └── page01_panel03.jpg
├── characters/
│   ├── character_designs/
│   │   ├── protagonist_design.jpg
│   │   ├── supporting_cast.jpg
│   │   └── antagonist_design.jpg
│   ├── expression_sheets/
│   └── costume_variations/
├── world/
│   ├── environment_concepts/
│   ├── location_references/
│   └── cultural_elements/
└── shorts/
    ├── character_introduction_01.mp4
    ├── world_showcase_02.mp4
    └── story_preview_03.mp4
```

## Special Features

- **Original Universe Creation**: Build completely new worlds with unique rules and aesthetics
- **Character Consistency**: Maintain character designs and personalities across all scenes
- **Cultural Authenticity**: Ground fantasy elements in believable cultural frameworks
- **Visual Identity**: Develop distinctive art style that serves the story's themes
- **World Logic**: Establish and maintain consistent rules for magic, technology, and society
- **Narrative Innovation**: Create fresh takes on classic storytelling structures

## Best Practices

1. **Originality First**: Create truly unique concepts not derivative of existing works
2. **Character Depth**: Develop multi-dimensional, relatable characters with clear motivations
3. **World Logic**: Establish and consistently follow the rules of your created universe
4. **Visual Identity**: Develop distinctive art style that enhances rather than distracts from story
5. **Cultural Research**: Ground fantasy elements in real cultural understanding and respect
6. **Thematic Coherence**: Ensure all elements serve the central themes and messages
7. **Reader Engagement**: Balance familiar elements with surprising innovations
8. **Pacing Mastery**: Use panel layouts and page turns to control story rhythm
9. **Emotional Resonance**: Connect fantastical elements to universal human experiences
10. **Consistent Quality**: Maintain high standards across all visual and narrative elements
