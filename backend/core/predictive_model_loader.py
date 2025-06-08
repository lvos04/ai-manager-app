"""
Predictive Model Loading System for AI Project Manager
Analyzes usage patterns and preloads models to eliminate perceived loading times.
"""

import asyncio
import json
import logging
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import weakref

logger = logging.getLogger(__name__)

class UsagePattern:
    """Tracks model usage patterns for prediction."""
    
    def __init__(self):
        self.model_sequences = deque(maxlen=1000)  # Last 1000 model loads
        self.model_pairs = defaultdict(int)  # Model A -> Model B frequency
        self.user_preferences = defaultdict(lambda: defaultdict(int))
        self.time_patterns = defaultdict(list)  # Time-based usage patterns
        
    def record_model_usage(self, user_id: str, model_name: str, project_type: str):
        """Record a model usage event."""
        timestamp = time.time()
        
        self.model_sequences.append({
            'user_id': user_id,
            'model': model_name,
            'project_type': project_type,
            'timestamp': timestamp
        })
        
        if len(self.model_sequences) >= 2:
            prev_model = self.model_sequences[-2]['model']
            self.model_pairs[prev_model] += 1
            
        self.user_preferences[user_id][model_name] += 1
        
        hour = int(timestamp % 86400 // 3600)  # Hour of day
        self.time_patterns[hour].append(model_name)
        
    def predict_next_models(self, current_model: str, user_id: str, 
                          project_type: str, limit: int = 3) -> List[Tuple[str, float]]:
        """Predict next likely models with confidence scores."""
        predictions = defaultdict(float)
        
        total_after_current = sum(count for model, count in self.model_pairs.items() 
                                if model == current_model)
        if total_after_current > 0:
            for next_model, count in self.model_pairs.items():
                if current_model in self.model_pairs:
                    predictions[next_model] += (count / total_after_current) * 0.4
                    
        user_prefs = self.user_preferences.get(user_id, {})
        total_user_usage = sum(user_prefs.values())
        if total_user_usage > 0:
            for model, count in user_prefs.items():
                predictions[model] += (count / total_user_usage) * 0.3
                
        project_models = [entry['model'] for entry in self.model_sequences 
                         if entry['project_type'] == project_type]
        if project_models:
            for model in set(project_models):
                freq = project_models.count(model) / len(project_models)
                predictions[model] += freq * 0.2
                
        current_hour = int(time.time() % 86400 // 3600)
        hour_models = self.time_patterns.get(current_hour, [])
        if hour_models:
            for model in set(hour_models):
                freq = hour_models.count(model) / len(hour_models)
                predictions[model] += freq * 0.1
                
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return sorted_predictions[:limit]

class ModelPreloader:
    """Handles background model preloading."""
    
    def __init__(self, ai_model_manager):
        self.ai_model_manager = ai_model_manager
        self.preload_queue = asyncio.Queue()
        self.preloaded_models = weakref.WeakValueDictionary()
        self.preload_lock = threading.Lock()
        self.is_running = False
        self.preload_thread = None
        
    def start_preloading(self):
        """Start the background preloading service."""
        if not self.is_running:
            self.is_running = True
            self.preload_thread = threading.Thread(target=self._preload_worker, daemon=True)
            self.preload_thread.start()
            logger.info("Model preloader started")
            
    def stop_preloading(self):
        """Stop the background preloading service."""
        self.is_running = False
        if self.preload_thread:
            self.preload_thread.join(timeout=5)
        logger.info("Model preloader stopped")
        
    def queue_preload(self, model_name: str, priority: float = 0.5):
        """Queue a model for preloading."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.preload_queue.put((priority, model_name)))
            loop.close()
        except Exception as e:
            logger.warning(f"Failed to queue preload for {model_name}: {e}")
            
    def _preload_worker(self):
        """Background worker for model preloading."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self._preload_loop())
        except Exception as e:
            logger.error(f"Preload worker error: {e}")
        finally:
            loop.close()
            
    async def _preload_loop(self):
        """Main preloading loop."""
        while self.is_running:
            try:
                priority, model_name = await asyncio.wait_for(
                    self.preload_queue.get(), timeout=1.0
                )
                
                if model_name in self.preloaded_models:
                    continue
                    
                await self._preload_model(model_name)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in preload loop: {e}")
                
    async def _preload_model(self, model_name: str):
        """Preload a specific model."""
        try:
            logger.info(f"Preloading model: {model_name}")
            
            model = await asyncio.get_event_loop().run_in_executor(
                None, self._load_model_sync, model_name
            )
            
            if model:
                self.preloaded_models[model_name] = model
                logger.info(f"Successfully preloaded: {model_name}")
            else:
                logger.warning(f"Failed to preload: {model_name}")
                
        except Exception as e:
            logger.error(f"Error preloading {model_name}: {e}")
            
    def _load_model_sync(self, model_name: str):
        """Synchronous model loading wrapper."""
        try:
            if hasattr(self.ai_model_manager, 'load_model'):
                return self.ai_model_manager.load_model(model_name)
            else:
                from ..pipelines.ai_models import load_llm
                return load_llm()
        except Exception as e:
            logger.error(f"Sync model loading failed for {model_name}: {e}")
            return None
            
    def get_preloaded_model(self, model_name: str):
        """Get a preloaded model if available."""
        return self.preloaded_models.get(model_name)

class PredictiveModelLoader:
    """Main predictive model loading system."""
    
    def __init__(self, ai_model_manager, cache_dir: Optional[Path] = None):
        self.ai_model_manager = ai_model_manager
        self.cache_dir = cache_dir or Path("cache/predictive_loader")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.usage_pattern = UsagePattern()
        self.model_preloader = ModelPreloader(ai_model_manager)
        self.prediction_cache = {}
        self.last_prediction_time = 0
        
        self._load_usage_patterns()
        
    def start(self):
        """Start the predictive loading system."""
        self.model_preloader.start_preloading()
        logger.info("Predictive model loader started")
        
    def stop(self):
        """Stop the predictive loading system."""
        self.model_preloader.stop_preloading()
        self._save_usage_patterns()
        logger.info("Predictive model loader stopped")
        
    def record_model_load(self, user_id: str, model_name: str, project_type: str):
        """Record a model loading event and trigger predictions."""
        self.usage_pattern.record_model_usage(user_id, model_name, project_type)
        
        self._predict_and_preload(model_name, user_id, project_type)
        
    def _predict_and_preload(self, current_model: str, user_id: str, project_type: str):
        """Predict next models and queue them for preloading."""
        try:
            predictions = self.usage_pattern.predict_next_models(
                current_model, user_id, project_type, limit=3
            )
            
            for model_name, confidence in predictions:
                if confidence > 0.3:  # Only preload high-confidence predictions
                    self.model_preloader.queue_preload(model_name, confidence)
                    logger.debug(f"Queued preload: {model_name} (confidence: {confidence:.2f})")
                    
        except Exception as e:
            logger.error(f"Error in prediction and preloading: {e}")
            
    def get_model_fast(self, model_name: str, user_id: str, project_type: str):
        """Get a model with fast loading (preloaded if available)."""
        preloaded = self.model_preloader.get_preloaded_model(model_name)
        if preloaded:
            logger.info(f"Using preloaded model: {model_name}")
            self.record_model_load(user_id, model_name, project_type)
            return preloaded
            
        logger.info(f"Loading model normally: {model_name}")
        model = self._load_model_normal(model_name)
        self.record_model_load(user_id, model_name, project_type)
        return model
        
    def _load_model_normal(self, model_name: str):
        """Load model using normal method."""
        try:
            if hasattr(self.ai_model_manager, 'load_model'):
                return self.ai_model_manager.load_model(model_name)
            else:
                from ..pipelines.ai_models import load_llm
                return load_llm()
        except Exception as e:
            logger.error(f"Normal model loading failed for {model_name}: {e}")
            return None
            
    def _save_usage_patterns(self):
        """Save usage patterns to disk."""
        try:
            patterns_file = self.cache_dir / "usage_patterns.json"
            
            data = {
                'model_pairs': dict(self.usage_pattern.model_pairs),
                'user_preferences': {
                    user: dict(prefs) 
                    for user, prefs in self.usage_pattern.user_preferences.items()
                },
                'time_patterns': {
                    str(hour): models 
                    for hour, models in self.usage_pattern.time_patterns.items()
                },
                'model_sequences': list(self.usage_pattern.model_sequences)
            }
            
            with open(patterns_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Saved usage patterns to {patterns_file}")
            
        except Exception as e:
            logger.error(f"Failed to save usage patterns: {e}")
            
    def _load_usage_patterns(self):
        """Load usage patterns from disk."""
        try:
            patterns_file = self.cache_dir / "usage_patterns.json"
            
            if patterns_file.exists():
                with open(patterns_file, 'r') as f:
                    data = json.load(f)
                    
                self.usage_pattern.model_pairs.update(data.get('model_pairs', {}))
                
                for user, prefs in data.get('user_preferences', {}).items():
                    self.usage_pattern.user_preferences[user].update(prefs)
                    
                for hour, models in data.get('time_patterns', {}).items():
                    self.usage_pattern.time_patterns[int(hour)] = models
                    
                sequences = data.get('model_sequences', [])
                self.usage_pattern.model_sequences.extend(sequences)
                
                logger.info(f"Loaded usage patterns from {patterns_file}")
                
        except Exception as e:
            logger.warning(f"Failed to load usage patterns: {e}")

_predictive_loader = None

def get_model_preloader():
    """Get global model preloader instance."""
    global _predictive_loader
    if _predictive_loader is None:
        from ..pipelines.ai_models import get_model_manager
        ai_model_manager = get_model_manager()
        _predictive_loader = PredictiveModelLoader(ai_model_manager)
        _predictive_loader.start()
    return _predictive_loader

def get_predictive_loader(ai_model_manager=None):
    """Get the global predictive loader instance."""
    global _predictive_loader
    if _predictive_loader is None and ai_model_manager:
        _predictive_loader = PredictiveModelLoader(ai_model_manager)
        _predictive_loader.start()
    return _predictive_loader

def cleanup_predictive_loader():
    """Cleanup the global predictive loader."""
    global _predictive_loader
    if _predictive_loader:
        _predictive_loader.stop()
        _predictive_loader = None
