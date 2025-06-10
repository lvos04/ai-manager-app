"""
Character Memory System for AI Project Manager
Maintains character consistency across multiple episodes and projects.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class CharacterMemoryManager:
    """Manages character consistency across episodes and projects."""
    
    def __init__(self, base_path: str = "characters", project_id: str = None):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        if project_id:
            self.character_db_path = self.base_path / f"character_database_{project_id}.json"
        else:
            self.character_db_path = self.base_path / "character_database.json"
        
        self.character_db = self._load_character_database()
    
    def _load_character_database(self) -> Dict[str, Any]:
        """Load the character database from disk."""
        if self.character_db_path.exists():
            try:
                with open(self.character_db_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading character database: {e}")
                return {}
        return {}
    
    def _save_character_database(self):
        """Save the character database to disk."""
        try:
            with open(self.character_db_path, 'w', encoding='utf-8') as f:
                json.dump(self.character_db, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving character database: {e}")
    
    def _generate_character_id(self, name: str, description: str) -> str:
        """Generate a unique ID for a character based on name and description."""
        content = f"{name.lower()}_{description.lower()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def register_character(self, name: str, description: str, voice_profile: str = "default", 
                          project_id: str = None, additional_data: Dict = None) -> str:
        """
        Register a character in the memory system.
        
        Args:
            name: Character name
            description: Character description/appearance
            voice_profile: Voice profile for the character
            project_id: Project this character belongs to
            additional_data: Additional character data (style, personality, etc.)
            
        Returns:
            str: Character ID for future reference
        """
        character_id = self._generate_character_id(name, description)
        
        if character_id not in self.character_db:
            self.character_db[character_id] = {
                "name": name,
                "description": description,
                "voice_profile": voice_profile,
                "projects": [project_id] if project_id else [],
                "reference_images": [],
                "design_seed": None,
                "style_parameters": {},
                "consistency_data": {},
                "animation_style": {
                    "movement_patterns": {},
                    "video_generation_params": {},
                    "preferred_models": []
                },
                "voice_characteristics": {
                    "language_profiles": {},
                    "voice_settings": {},
                    "speech_patterns": {}
                },
                "personality_traits": {
                    "dialogue_style": "",
                    "behavior_patterns": {},
                    "character_quirks": []
                },
                "design_elements": {
                    "clothing": {},
                    "accessories": {},
                    "color_schemes": {}
                },
                "created_at": str(Path().cwd()),
                "additional_data": additional_data or {}
            }
        else:
            if project_id and project_id not in self.character_db[character_id]["projects"]:
                self.character_db[character_id]["projects"].append(project_id)
        
        self._save_character_database()
        logger.info(f"Registered character: {name} (ID: {character_id})")
        return character_id
    
    def get_character(self, character_id: str) -> Optional[Dict[str, Any]]:
        """Get character data by ID."""
        return self.character_db.get(character_id)
    
    def get_character_by_name(self, name: str, project_id: str = None) -> Optional[Dict[str, Any]]:
        """Get character data by name, optionally filtered by project."""
        for char_id, char_data in self.character_db.items():
            if char_data["name"].lower() == name.lower():
                if project_id is None or project_id in char_data["projects"]:
                    return {**char_data, "character_id": char_id}
        return None
    
    def save_character_reference(self, character_id: str, image_path: str, angle: str = "front"):
        """Save a reference image for a character."""
        if character_id in self.character_db:
            ref_data = {
                "path": str(image_path),
                "angle": angle,
                "timestamp": str(Path().cwd())
            }
            self.character_db[character_id]["reference_images"].append(ref_data)
            self._save_character_database()
            logger.info(f"Saved reference image for character {character_id}: {image_path}")
    
    def set_character_seed(self, character_id: str, seed: int):
        """Set the generation seed for character consistency."""
        if character_id in self.character_db:
            self.character_db[character_id]["design_seed"] = seed
            self._save_character_database()
    
    def get_character_seed(self, character_id: str) -> Optional[int]:
        """Get the generation seed for a character."""
        char_data = self.character_db.get(character_id)
        return char_data.get("design_seed") if char_data else None
    
    def update_style_parameters(self, character_id: str, parameters: Dict[str, Any]):
        """Update style parameters for character consistency."""
        if character_id in self.character_db:
            self.character_db[character_id]["style_parameters"].update(parameters)
            self._save_character_database()
    
    def get_style_parameters(self, character_id: str) -> Dict[str, Any]:
        """Get style parameters for a character."""
        char_data = self.character_db.get(character_id)
        return char_data.get("style_parameters", {}) if char_data else {}
    
    def get_project_characters(self, project_id: str) -> List[Dict[str, Any]]:
        """Get all characters associated with a project."""
        characters = []
        for char_id, char_data in self.character_db.items():
            if project_id in char_data.get("projects", []):
                characters.append({**char_data, "character_id": char_id})
        return characters
    
    def get_character_reference_images(self, character_id: str) -> List[Dict[str, Any]]:
        """Get all reference images for a character."""
        char_data = self.character_db.get(character_id)
        return char_data.get("reference_images", []) if char_data else []
    
    def ensure_character_consistency(self, character_id: str, generation_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure character consistency by applying saved parameters.
        
        Args:
            character_id: Character ID
            generation_params: Current generation parameters
            
        Returns:
            Updated generation parameters with consistency data
        """
        char_data = self.character_db.get(character_id)
        if not char_data:
            return generation_params
        
        if char_data.get("design_seed"):
            generation_params["seed"] = char_data["design_seed"]
        
        style_params = char_data.get("style_parameters", {})
        generation_params.update(style_params)
        
        if "prompt" in generation_params:
            char_desc = char_data.get("description", "")
            generation_params["prompt"] = f"{char_desc}, {generation_params['prompt']}"
        
        logger.info(f"Applied consistency parameters for character {character_id}")
        return generation_params
    
    def update_animation_style(self, character_id: str, animation_params: Dict[str, Any]):
        """Update animation style parameters for character consistency."""
        if character_id in self.character_db:
            self.character_db[character_id]["animation_style"].update(animation_params)
            self._save_character_database()
    
    def get_animation_style(self, character_id: str) -> Dict[str, Any]:
        """Get animation style parameters for a character."""
        char_data = self.character_db.get(character_id)
        return char_data.get("animation_style", {}) if char_data else {}
    
    def update_voice_characteristics(self, character_id: str, voice_params: Dict[str, Any]):
        """Update voice characteristics for character consistency."""
        if character_id in self.character_db:
            self.character_db[character_id]["voice_characteristics"].update(voice_params)
            self._save_character_database()
    
    def get_voice_characteristics(self, character_id: str) -> Dict[str, Any]:
        """Get voice characteristics for a character."""
        char_data = self.character_db.get(character_id)
        return char_data.get("voice_characteristics", {}) if char_data else {}
    
    def update_personality_traits(self, character_id: str, personality_params: Dict[str, Any]):
        """Update personality traits for character consistency."""
        if character_id in self.character_db:
            self.character_db[character_id]["personality_traits"].update(personality_params)
            self._save_character_database()
    
    def get_personality_traits(self, character_id: str) -> Dict[str, Any]:
        """Get personality traits for a character."""
        char_data = self.character_db.get(character_id)
        return char_data.get("personality_traits", {}) if char_data else {}
    
    def update_design_elements(self, character_id: str, design_params: Dict[str, Any]):
        """Update design elements for character consistency."""
        if character_id in self.character_db:
            self.character_db[character_id]["design_elements"].update(design_params)
            self._save_character_database()
    
    def get_design_elements(self, character_id: str) -> Dict[str, Any]:
        """Get design elements for a character."""
        char_data = self.character_db.get(character_id)
        return char_data.get("design_elements", {}) if char_data else {}
    
    def ensure_comprehensive_consistency(self, character_id: str, generation_params: Dict[str, Any], 
                                       context_type: str = "video") -> Dict[str, Any]:
        """
        Ensure comprehensive character consistency across all aspects.
        
        Args:
            character_id: Character ID
            generation_params: Current generation parameters
            context_type: Type of generation (video, voice, image, etc.)
            
        Returns:
            Updated generation parameters with all consistency data
        """
        char_data = self.character_db.get(character_id)
        if not char_data:
            return generation_params
        
        if char_data.get("design_seed"):
            generation_params["seed"] = char_data["design_seed"]
        
        style_params = char_data.get("style_parameters", {})
        generation_params.update(style_params)
        
        if context_type == "video":
            animation_style = char_data.get("animation_style", {})
            if animation_style.get("video_generation_params"):
                generation_params.update(animation_style["video_generation_params"])
        
        if context_type == "voice":
            voice_chars = char_data.get("voice_characteristics", {})
            if voice_chars.get("voice_settings"):
                generation_params.update(voice_chars["voice_settings"])
        
        design_elements = char_data.get("design_elements", {})
        if design_elements and "prompt" in generation_params:
            char_desc = char_data.get("description", "")
            clothing_desc = design_elements.get("clothing", {}).get("description", "")
            accessories_desc = design_elements.get("accessories", {}).get("description", "")
            
            full_desc = f"{char_desc}"
            if clothing_desc:
                full_desc += f", {clothing_desc}"
            if accessories_desc:
                full_desc += f", {accessories_desc}"
            
            generation_params["prompt"] = f"{full_desc}, {generation_params['prompt']}"
        
        logger.info(f"Applied comprehensive consistency parameters for character {character_id}")
        return generation_params
    
    def cleanup_old_references(self, days_old: int = 30):
        """Clean up old character references to save space."""
        try:
            import time
            from datetime import datetime, timedelta
            
            cutoff_date = datetime.now() - timedelta(days=days_old)
            cutoff_timestamp = cutoff_date.timestamp()
            
            characters_to_remove = []
            
            for char_id, char_data in self.character_db.items():
                ref_images = char_data.get("reference_images", [])
                updated_refs = []
                
                for ref in ref_images:
                    ref_path = Path(ref.get("path", ""))
                    if ref_path.exists():
                        if ref_path.stat().st_mtime > cutoff_timestamp:
                            updated_refs.append(ref)
                        else:
                            try:
                                ref_path.unlink()
                                logger.info(f"Removed old reference: {ref_path}")
                            except Exception as e:
                                logger.error(f"Error removing {ref_path}: {e}")
                
                if len(updated_refs) != len(ref_images):
                    self.character_db[char_id]["reference_images"] = updated_refs
                    
                if not updated_refs and not char_data.get("projects"):
                    characters_to_remove.append(char_id)
            
            for char_id in characters_to_remove:
                del self.character_db[char_id]
                logger.info(f"Removed unused character: {char_id}")
            
            self._save_character_database()
            logger.info(f"Cleanup complete: removed {len(characters_to_remove)} characters")
            
        except Exception as e:
            logger.error(f"Error during character cleanup: {e}")

_character_memory_manager = None

def get_character_memory_manager(base_path: str = "characters", project_id: str = None) -> CharacterMemoryManager:
    """Get the character memory manager instance for a specific project."""
    global _character_memory_manager
    if _character_memory_manager is None or project_id:
        _character_memory_manager = CharacterMemoryManager(base_path, project_id)
    return _character_memory_manager
