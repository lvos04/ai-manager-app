"""
Script Expander for AI Project Manager
Automatically expands scripts to ensure minimum 20-minute episode duration using LLM.
"""

from .common_imports import *
from .ai_imports import *
import time
import re
import random

logger = logging.getLogger(__name__)

class ScriptExpander:
    """Expands scripts to meet minimum duration requirements using LLM."""
    
    def __init__(self):
        self.min_duration_minutes = 20
        self.scene_duration_estimates = {
            "dialogue": 2.0,  # minutes per scene
            "action": 1.5,
            "combat": 3.0,
            "exploration": 2.5,
            "character_development": 3.0,
            "flashback": 2.0,
            "world_building": 2.5,
            "transition": 0.5
        }
        
        self.expansion_templates = {
            "character_development": [
                "Character backstory exploration scene",
                "Character relationship development",
                "Character internal conflict moment",
                "Character skill/power development",
                "Character emotional growth scene"
            ],
            "world_building": [
                "Location exploration and description",
                "Cultural/society explanation scene",
                "History/lore exposition",
                "Environment interaction scene",
                "World rules/magic system explanation"
            ],
            "subplot": [
                "Secondary character storyline",
                "Parallel plot development",
                "Mystery/investigation subplot",
                "Romance subplot development",
                "Political intrigue subplot"
            ],
            "action_expansion": [
                "Extended combat sequence",
                "Chase scene",
                "Escape sequence",
                "Training/preparation montage",
                "Infiltration/stealth sequence"
            ],
            "emotional_beats": [
                "Quiet character moment",
                "Team bonding scene",
                "Conflict resolution",
                "Celebration/victory scene",
                "Loss/grief processing scene"
            ]
        }
    
    def analyze_script_duration(self, script_data: Dict) -> float:
        """
        Analyze script and estimate total duration.
        
        Args:
            script_data: Parsed script data
            
        Returns:
            Estimated duration in minutes
        """
        total_duration = 0.0
        
        if "scenes" in script_data:
            for scene in script_data["scenes"]:
                scene_type = scene.get("type", "dialogue").lower()
                
                base_duration = self.scene_duration_estimates.get(scene_type, 2.0)
                
                dialogue_lines = len(scene.get("dialogue", []))
                dialogue_duration = dialogue_lines * 0.1  # ~6 seconds per line
                
                description = scene.get("description", "")
                description_duration = len(description.split()) * 0.05  # ~3 seconds per word
                
                scene_duration = max(base_duration, dialogue_duration + description_duration)
                total_duration += scene_duration
        
        return total_duration
    
    def needs_expansion(self, script_data: Dict) -> bool:
        """
        Check if script needs expansion to meet minimum duration.
        
        Args:
            script_data: Parsed script data
            
        Returns:
            True if expansion is needed
        """
        current_duration = self.analyze_script_duration(script_data)
        return current_duration < self.min_duration_minutes
    
    def generate_expansion_plan(self, script_data: Dict, target_duration: float = None) -> Dict:
        """
        Generate plan for expanding script to target duration.
        
        Args:
            script_data: Current script data
            target_duration: Target duration in minutes (default: min_duration_minutes)
            
        Returns:
            Expansion plan with new scenes to add
        """
        if target_duration is None:
            target_duration = self.min_duration_minutes
        
        current_duration = self.analyze_script_duration(script_data)
        needed_duration = target_duration - current_duration
        
        if needed_duration <= 0:
            return {"expansion_needed": False, "current_duration": current_duration}
        
        existing_scenes = script_data.get("scenes", [])
        characters = script_data.get("characters", [])
        setting = script_data.get("setting", "")
        genre = script_data.get("genre", "anime")
        
        expansion_plan = {
            "expansion_needed": True,
            "current_duration": current_duration,
            "target_duration": target_duration,
            "needed_duration": needed_duration,
            "new_scenes": [],
            "insertion_points": []
        }
        
        avg_scene_duration = 2.5  # minutes
        scenes_to_add = int(needed_duration / avg_scene_duration) + 1
        
        expansion_types = self._select_expansion_types(script_data, scenes_to_add)
        
        for i, expansion_type in enumerate(expansion_types):
            new_scene = self._generate_expansion_scene(
                expansion_type=expansion_type,
                script_context=script_data,
                scene_number=len(existing_scenes) + i + 1
            )
            expansion_plan["new_scenes"].append(new_scene)
            
            insertion_point = self._find_insertion_point(existing_scenes, expansion_type, i)
            expansion_plan["insertion_points"].append(insertion_point)
        
        return expansion_plan
    
    def _select_expansion_types(self, script_data: Dict, num_scenes: int) -> List[str]:
        """Select appropriate expansion types based on script content."""
        existing_scenes = script_data.get("scenes", [])
        genre = script_data.get("genre", "anime")
        
        has_action = any("action" in scene.get("type", "").lower() or 
                        "combat" in scene.get("type", "").lower() 
                        for scene in existing_scenes)
        has_character_dev = any("character" in scene.get("description", "").lower() 
                               for scene in existing_scenes)
        has_world_building = any("world" in scene.get("description", "").lower() or
                                "location" in scene.get("description", "").lower()
                                for scene in existing_scenes)
        
        expansion_types = []
        
        for i in range(num_scenes):
            if i % 4 == 0 and not has_character_dev:
                expansion_types.append("character_development")
            elif i % 4 == 1 and not has_world_building:
                expansion_types.append("world_building")
            elif i % 4 == 2 and not has_action and genre in ["anime", "superhero", "action"]:
                expansion_types.append("action_expansion")
            elif i % 4 == 3:
                expansion_types.append("emotional_beats")
            else:
                type_cycle = ["character_development", "world_building", "subplot", "emotional_beats"]
                expansion_types.append(type_cycle[i % len(type_cycle)])
        
        return expansion_types
    
    def _generate_expansion_scene(self, expansion_type: str, script_context: Dict, 
                                scene_number: int) -> Dict:
        """Generate a new scene for expansion."""
        characters = script_context.get("characters", [])
        setting = script_context.get("setting", "")
        genre = script_context.get("genre", "anime")
        
        templates = self.expansion_templates.get(expansion_type, ["Generic scene"])
        template = random.choice(templates)
        
        if expansion_type == "character_development":
            scene = self._generate_character_development_scene(characters, setting, template)
        elif expansion_type == "world_building":
            scene = self._generate_world_building_scene(setting, genre, template)
        elif expansion_type == "action_expansion":
            scene = self._generate_action_scene(characters, setting, genre)
        elif expansion_type == "emotional_beats":
            scene = self._generate_emotional_scene(characters, setting)
        else:
            scene = self._generate_generic_scene(expansion_type, characters, setting)
        
        scene.update({
            "scene_number": scene_number,
            "expansion_type": expansion_type,
            "generated": True
        })
        
        return scene
    
    def _generate_character_development_scene(self, characters: List, setting: str, template: str) -> Dict:
        """Generate character development scene."""
        main_char = characters[0] if characters else {"name": "Protagonist"}
        
        return {
            "type": "character_development",
            "title": f"Character Development - {main_char.get('name', 'Character')}",
            "description": f"{template} featuring {main_char.get('name', 'the protagonist')} in {setting}. "
                          f"This scene explores their background, motivations, and personal growth.",
            "location": setting,
            "characters_present": [main_char.get("name", "Protagonist")],
            "dialogue": [
                {
                    "character": main_char.get("name", "Protagonist"),
                    "line": "Sometimes I wonder if I'm really cut out for this...",
                    "emotion": "contemplative"
                }
            ],
            "duration_estimate": 3.0
        }
    
    def _generate_world_building_scene(self, setting: str, genre: str, template: str) -> Dict:
        """Generate world building scene."""
        return {
            "type": "world_building",
            "title": f"World Building - {setting}",
            "description": f"{template} showcasing the rich details of {setting}. "
                          f"This scene establishes the world's rules, culture, and atmosphere in the {genre} style.",
            "location": setting,
            "characters_present": [],
            "dialogue": [],
            "world_elements": [
                "Environmental details",
                "Cultural aspects",
                "Historical context",
                "Atmospheric elements"
            ],
            "duration_estimate": 2.5
        }
    
    def _generate_action_scene(self, characters: List, setting: str, genre: str) -> Dict:
        """Generate action/combat scene."""
        participants = characters[:2] if len(characters) >= 2 else [{"name": "Hero"}, {"name": "Opponent"}]
        
        return {
            "type": "combat",
            "title": "Extended Action Sequence",
            "description": f"Intense {genre}-style combat scene in {setting} featuring dynamic action and choreography.",
            "location": setting,
            "characters_present": [char.get("name", f"Fighter{i+1}") for i, char in enumerate(participants)],
            "combat_type": "melee",
            "difficulty": "medium",
            "dialogue": [
                {
                    "character": participants[0].get("name", "Hero"),
                    "line": "I won't let you get away with this!",
                    "emotion": "determined"
                }
            ],
            "duration_estimate": 3.0
        }
    
    def _generate_emotional_scene(self, characters: List, setting: str) -> Dict:
        """Generate emotional beat scene."""
        main_char = characters[0] if characters else {"name": "Character"}
        
        return {
            "type": "emotional_beat",
            "title": "Emotional Moment",
            "description": f"Quiet character moment allowing for emotional depth and reflection in {setting}.",
            "location": setting,
            "characters_present": [main_char.get("name", "Character")],
            "dialogue": [
                {
                    "character": main_char.get("name", "Character"),
                    "line": "After everything we've been through, I finally understand what really matters.",
                    "emotion": "reflective"
                }
            ],
            "duration_estimate": 2.0
        }
    
    def _generate_generic_scene(self, expansion_type: str, characters: List, setting: str) -> Dict:
        """Generate generic scene for unknown expansion types."""
        return {
            "type": expansion_type,
            "title": f"{expansion_type.replace('_', ' ').title()} Scene",
            "description": f"Additional {expansion_type} content to enhance the story in {setting}.",
            "location": setting,
            "characters_present": [char.get("name", f"Character{i+1}") for i, char in enumerate(characters[:2])],
            "dialogue": [],
            "duration_estimate": 2.0
        }
    
    def _find_insertion_point(self, existing_scenes: List, expansion_type: str, scene_index: int) -> int:
        """Find appropriate insertion point for new scene."""
        if not existing_scenes:
            return 0
        
        if expansion_type == "character_development":
            return min(len(existing_scenes) // 3, len(existing_scenes) - 1)
        elif expansion_type == "world_building":
            return min(2, len(existing_scenes))
        elif expansion_type == "action_expansion":
            return len(existing_scenes) // 2
        elif expansion_type == "emotional_beats":
            return max(len(existing_scenes) - 2, 0)
        else:
            return (scene_index * len(existing_scenes)) // max(scene_index + 1, 1)
    
    def expand_script_with_llm(self, script_data: Dict, llm_model=None) -> Dict:
        """
        Use LLM to expand script with natural content.
        
        Args:
            script_data: Current script data
            llm_model: LLM model for generation
            
        Returns:
            Expanded script data
        """
        if not self.needs_expansion(script_data):
            return script_data
        
        expansion_plan = self.generate_expansion_plan(script_data)
        
        if not expansion_plan["expansion_needed"]:
            return script_data
        
        llm_prompt = self._create_llm_expansion_prompt(script_data, expansion_plan)
        
        try:
            if llm_model and hasattr(llm_model, 'generate'):
                expanded_content = llm_model.generate(llm_prompt, max_tokens=2000)
            else:
                expanded_content = self._rule_based_expansion(script_data, expansion_plan)
            
            expanded_script = self._integrate_expanded_content(script_data, expansion_plan, expanded_content)
            
            logger.info(f"Script expanded from {expansion_plan['current_duration']:.1f} to "
                       f"{self.analyze_script_duration(expanded_script):.1f} minutes")
            
            return expanded_script
            
        except Exception as e:
            logger.error(f"Error in LLM script expansion: {e}")
            return self._rule_based_expansion(script_data, expansion_plan)
    
    def _create_llm_expansion_prompt(self, script_data: Dict, expansion_plan: Dict) -> str:
        """Create prompt for LLM script expansion."""
        current_scenes = script_data.get("scenes", [])
        characters = script_data.get("characters", [])
        setting = script_data.get("setting", "")
        genre = script_data.get("genre", "anime")
        
        prompt = f"""
Expand this {genre} script to reach {expansion_plan['target_duration']} minutes total duration.

Current script summary:
- Setting: {setting}
- Characters: {[char.get('name', 'Unknown') for char in characters]}
- Current scenes: {len(current_scenes)}
- Current duration: {expansion_plan['current_duration']:.1f} minutes
- Needed additional duration: {expansion_plan['needed_duration']:.1f} minutes

Existing scenes:
"""
        
        for i, scene in enumerate(current_scenes[:3]):  # Show first 3 scenes for context
            prompt += f"{i+1}. {scene.get('title', 'Scene')} - {scene.get('description', '')[:100]}...\n"
        
        prompt += f"""

Please generate {len(expansion_plan['new_scenes'])} additional scenes that:
1. Maintain story continuity and character consistency
2. Add meaningful content (character development, world building, action, emotional beats)
3. Fit naturally into the existing narrative
4. Are appropriate for the {genre} genre
5. Include proper dialogue and scene descriptions

Format each new scene as:
Scene Title: [Title]
Type: [dialogue/action/combat/character_development/world_building]
Location: [Location]
Characters: [Character names]
Description: [Detailed scene description]
Dialogue: [Character dialogue with emotions]
Duration: [Estimated minutes]

Generate the additional scenes now:
"""
        
        return prompt
    
    def _rule_based_expansion(self, script_data: Dict, expansion_plan: Dict) -> Dict:
        """Fallback rule-based expansion when LLM is not available."""
        expanded_script = script_data.copy()
        
        new_scenes = []
        original_scenes = expanded_script.get("scenes", [])
        
        for i, (scene, insertion_point) in enumerate(zip(expansion_plan["new_scenes"], 
                                                        expansion_plan["insertion_points"])):
            adjusted_insertion = insertion_point + i
            new_scenes.append((adjusted_insertion, scene))
        
        new_scenes.sort(key=lambda x: x[0])
        
        final_scenes = original_scenes.copy()
        for insertion_point, scene in reversed(new_scenes):  # Insert in reverse order to maintain indices
            final_scenes.insert(min(insertion_point, len(final_scenes)), scene)
        
        expanded_script["scenes"] = final_scenes
        
        return expanded_script
    
    def _integrate_expanded_content(self, script_data: Dict, expansion_plan: Dict, 
                                  llm_content: str) -> Dict:
        """Integrate LLM-generated content into script."""
        return self._rule_based_expansion(script_data, expansion_plan)

def expand_script_if_needed(script_data: Dict, min_duration: float = 20.0, 
                          llm_model=None) -> Dict:
    """
    Expand script if it doesn't meet minimum duration requirements.
    
    Args:
        script_data: Parsed script data
        min_duration: Minimum duration in minutes
        llm_model: Optional LLM model for expansion
        
    Returns:
        Expanded script data if needed, original otherwise
    """
    expander = ScriptExpander()
    expander.min_duration_minutes = min_duration
    
    if expander.needs_expansion(script_data):
        logger.info(f"Script duration {expander.analyze_script_duration(script_data):.1f} minutes "
                   f"is below minimum {min_duration} minutes. Expanding...")
        return expander.expand_script_with_llm(script_data, llm_model)
    else:
        logger.info(f"Script duration {expander.analyze_script_duration(script_data):.1f} minutes "
                   f"meets minimum requirement of {min_duration} minutes.")
        return script_data
