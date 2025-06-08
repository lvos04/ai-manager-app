"""
Combat Scene Generator for AI Project Manager
Handles comprehensive combat scenes with choreography, camera effects, and VFX.
"""

from .common_imports import *
from .ai_imports import *
import time
import random

logger = logging.getLogger(__name__)

class CombatSceneGenerator:
    """Generates comprehensive combat scenes with choreography and camera effects."""
    
    def __init__(self):
        self.combat_types = {
            "melee": {
                "movements": ["punch", "kick", "block", "dodge", "grapple", "throw"],
                "camera_angles": ["close_up", "wide_shot", "over_shoulder", "low_angle", "high_angle"],
                "effects": ["impact_flash", "speed_lines", "dust_cloud", "shockwave"]
            },
            "ranged": {
                "movements": ["aim", "shoot", "reload", "take_cover", "roll", "jump"],
                "camera_angles": ["first_person", "third_person", "bullet_time", "tracking_shot"],
                "effects": ["muzzle_flash", "bullet_trail", "explosion", "debris"]
            },
            "magic": {
                "movements": ["cast", "channel", "gesture", "summon", "shield", "teleport"],
                "camera_angles": ["dramatic_low", "overhead", "spiral", "zoom_in", "pull_back"],
                "effects": ["energy_burst", "magical_aura", "spell_circle", "elemental_fx"]
            },
            "aerial": {
                "movements": ["fly", "dive", "hover", "spin", "dash", "glide"],
                "camera_angles": ["bird_eye", "chase_cam", "cockpit_view", "ground_up"],
                "effects": ["wind_trails", "sonic_boom", "altitude_blur", "sky_fx"]
            }
        }
        
        self.camera_techniques = {
            "slow_motion": {"speed": 0.3, "duration": 2.0},
            "speed_ramp": {"speed_changes": [1.0, 0.5, 2.0, 1.0]},
            "shake": {"intensity": "medium", "duration": 1.0},
            "zoom_burst": {"zoom_factor": 1.5, "duration": 0.5},
            "dutch_angle": {"tilt": 15, "duration": 3.0},
            "whip_pan": {"speed": "fast", "blur": True}
        }
    
    def generate_combat_choreography(self, combat_type: str, duration: float, 
                                   characters: List[Dict], difficulty: str = "medium") -> Dict:
        """
        Generate detailed combat choreography.
        
        Args:
            combat_type: Type of combat (melee, ranged, magic, aerial)
            duration: Duration of combat scene in seconds
            characters: List of character data
            difficulty: Combat difficulty (easy, medium, hard, epic)
            
        Returns:
            Dict containing choreography data
        """
        if combat_type not in self.combat_types:
            combat_type = "melee"
        
        combat_data = self.combat_types[combat_type]
        
        moves_per_second = {"easy": 0.5, "medium": 1.0, "hard": 1.5, "epic": 2.0}
        total_moves = int(duration * moves_per_second.get(difficulty, 1.0))
        
        choreography = {
            "combat_type": combat_type,
            "duration": duration,
            "difficulty": difficulty,
            "total_moves": total_moves,
            "sequences": []
        }
        
        time_per_move = duration / max(total_moves, 1)
        current_time = 0.0
        
        for i in range(total_moves):
            attacker = random.choice(characters) if characters else {"name": "Fighter1"}
            defender = random.choice([c for c in characters if c != attacker]) if len(characters) > 1 else {"name": "Fighter2"}
            
            movement = random.choice(combat_data["movements"])
            camera_angle = random.choice(combat_data["camera_angles"])
            effect = random.choice(combat_data["effects"])
            
            camera_technique = None
            if random.random() < 0.3:  # 30% chance for special camera technique
                camera_technique = random.choice(list(self.camera_techniques.keys()))
            
            sequence = {
                "sequence_id": i + 1,
                "start_time": current_time,
                "duration": time_per_move,
                "attacker": attacker["name"],
                "defender": defender["name"],
                "movement": movement,
                "camera_angle": camera_angle,
                "effect": effect,
                "camera_technique": camera_technique,
                "intensity": self._calculate_intensity(i, total_moves, difficulty)
            }
            
            choreography["sequences"].append(sequence)
            current_time += time_per_move
        
        return choreography
    
    def _calculate_intensity(self, sequence_num: int, total_sequences: int, difficulty: str) -> float:
        """Calculate intensity for a sequence based on position and difficulty."""
        position_factor = sequence_num / max(total_sequences - 1, 1)
        
        if position_factor < 0.3:
            base_intensity = 0.4 + (position_factor / 0.3) * 0.3  # 0.4 to 0.7
        elif position_factor < 0.8:
            base_intensity = 0.7 + ((position_factor - 0.3) / 0.5) * 0.3  # 0.7 to 1.0
        else:
            base_intensity = 1.0 - ((position_factor - 0.8) / 0.2) * 0.3  # 1.0 to 0.7
        
        difficulty_multiplier = {"easy": 0.7, "medium": 1.0, "hard": 1.2, "epic": 1.5}
        final_intensity = base_intensity * difficulty_multiplier.get(difficulty, 1.0)
        
        return min(max(final_intensity, 0.1), 1.0)
    
    def generate_camera_script(self, choreography: Dict) -> List[Dict]:
        """
        Generate detailed camera script for combat scene.
        
        Args:
            choreography: Combat choreography data
            
        Returns:
            List of camera instructions
        """
        camera_script = []
        
        for sequence in choreography["sequences"]:
            camera_instruction = {
                "sequence_id": sequence["sequence_id"],
                "start_time": sequence["start_time"],
                "duration": sequence["duration"],
                "camera_angle": sequence["camera_angle"],
                "camera_technique": sequence["camera_technique"],
                "focus_target": sequence["attacker"],
                "secondary_target": sequence["defender"],
                "movement_type": sequence["movement"],
                "intensity": sequence["intensity"]
            }
            
            if sequence["camera_angle"] == "close_up":
                camera_instruction.update({
                    "focal_length": "85mm",
                    "aperture": "f/2.8",
                    "focus_distance": "1.5m"
                })
            elif sequence["camera_angle"] == "wide_shot":
                camera_instruction.update({
                    "focal_length": "24mm",
                    "aperture": "f/5.6",
                    "focus_distance": "5m"
                })
            elif sequence["camera_angle"] == "low_angle":
                camera_instruction.update({
                    "camera_height": "0.5m",
                    "tilt_angle": "15deg_up",
                    "dramatic_effect": True
                })
            
            if sequence["camera_technique"]:
                technique_params = self.camera_techniques.get(sequence["camera_technique"], {})
                camera_instruction["technique_params"] = technique_params
            
            camera_script.append(camera_instruction)
        
        return camera_script
    
    def generate_vfx_script(self, choreography: Dict) -> List[Dict]:
        """
        Generate VFX script for combat scene.
        
        Args:
            choreography: Combat choreography data
            
        Returns:
            List of VFX instructions
        """
        vfx_script = []
        
        for sequence in choreography["sequences"]:
            vfx_instruction = {
                "sequence_id": sequence["sequence_id"],
                "start_time": sequence["start_time"],
                "duration": sequence["duration"],
                "effect_type": sequence["effect"],
                "intensity": sequence["intensity"],
                "trigger_event": sequence["movement"]
            }
            
            if sequence["effect"] == "impact_flash":
                vfx_instruction.update({
                    "flash_duration": 0.1,
                    "flash_intensity": sequence["intensity"],
                    "color": "white",
                    "bloom": True
                })
            elif sequence["effect"] == "speed_lines":
                vfx_instruction.update({
                    "line_count": int(20 * sequence["intensity"]),
                    "line_length": "screen_width",
                    "direction": "radial",
                    "fade_time": 0.3
                })
            elif sequence["effect"] == "explosion":
                vfx_instruction.update({
                    "explosion_size": sequence["intensity"] * 2.0,
                    "particle_count": int(100 * sequence["intensity"]),
                    "smoke_duration": 2.0,
                    "debris": True
                })
            elif sequence["effect"] == "energy_burst":
                vfx_instruction.update({
                    "energy_color": "blue",
                    "burst_radius": sequence["intensity"] * 3.0,
                    "particle_trail": True,
                    "glow_effect": True
                })
            
            vfx_script.append(vfx_instruction)
        
        return vfx_script
    
    def create_combat_scene_prompt(self, choreography: Dict, camera_script: List[Dict], 
                                 vfx_script: List[Dict], style: str = "anime") -> str:
        """
        Create comprehensive prompt for video generation models.
        
        Args:
            choreography: Combat choreography data
            camera_script: Camera instructions
            vfx_script: VFX instructions
            style: Visual style (anime, realistic, comic, etc.)
            
        Returns:
            Detailed prompt for video generation
        """
        style_modifiers = {
            "anime": "anime style, dynamic action, speed lines, dramatic poses",
            "realistic": "photorealistic, cinematic lighting, detailed textures",
            "comic": "comic book style, bold colors, dramatic panels",
            "manga": "manga style, black and white, detailed line art",
            "superhero": "superhero style, dramatic lighting, heroic poses"
        }
        
        base_style = style_modifiers.get(style, style_modifiers["anime"])
        
        prompt_parts = [
            f"{base_style}",
            f"{choreography['combat_type']} combat scene",
            f"duration {choreography['duration']} seconds",
            f"{choreography['difficulty']} difficulty level"
        ]
        
        characters = set()
        for seq in choreography["sequences"]:
            characters.add(seq["attacker"])
            characters.add(seq["defender"])
        
        if characters:
            prompt_parts.append(f"characters: {', '.join(characters)}")
        
        movements = set(seq["movement"] for seq in choreography["sequences"])
        effects = set(seq["effect"] for seq in choreography["sequences"])
        
        prompt_parts.append(f"movements: {', '.join(list(movements)[:3])}")
        prompt_parts.append(f"effects: {', '.join(list(effects)[:3])}")
        
        techniques = [seq["camera_technique"] for seq in choreography["sequences"] if seq["camera_technique"]]
        if techniques:
            unique_techniques = list(set(techniques))
            prompt_parts.append(f"camera techniques: {', '.join(unique_techniques[:2])}")
        
        prompt_parts.extend([
            "high quality",
            "detailed animation",
            "smooth motion",
            "16:9 aspect ratio",
            "professional cinematography"
        ])
        
        return ", ".join(prompt_parts)

def generate_combat_scene(scene_description: str, duration: float, characters: List[Dict], 
                         style: str = "anime", difficulty: str = "medium") -> Dict:
    """
    Generate a complete combat scene with choreography, camera work, and VFX.
    
    Args:
        scene_description: Description of the combat scene
        duration: Duration in seconds
        characters: List of character data
        style: Visual style
        difficulty: Combat difficulty
        
    Returns:
        Complete combat scene data
    """
    generator = CombatSceneGenerator()
    
    combat_type = "melee"  # default
    if any(word in scene_description.lower() for word in ["gun", "shoot", "bullet", "rifle"]):
        combat_type = "ranged"
    elif any(word in scene_description.lower() for word in ["magic", "spell", "energy", "power"]):
        combat_type = "magic"
    elif any(word in scene_description.lower() for word in ["fly", "aerial", "sky", "air"]):
        combat_type = "aerial"
    
    choreography = generator.generate_combat_choreography(
        combat_type=combat_type,
        duration=duration,
        characters=characters,
        difficulty=difficulty
    )
    
    camera_script = generator.generate_camera_script(choreography)
    
    vfx_script = generator.generate_vfx_script(choreography)
    
    video_prompt = generator.create_combat_scene_prompt(
        choreography=choreography,
        camera_script=camera_script,
        vfx_script=vfx_script,
        style=style
    )
    
    return {
        "scene_type": "combat",
        "combat_type": combat_type,
        "duration": duration,
        "style": style,
        "difficulty": difficulty,
        "choreography": choreography,
        "camera_script": camera_script,
        "vfx_script": vfx_script,
        "video_prompt": video_prompt,
        "characters": characters
    }
