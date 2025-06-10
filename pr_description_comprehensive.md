# Comprehensive Model Configuration and Prompt Template Implementation

## Summary
This PR implements comprehensive model configuration improvements and adds prompt templates for ALL base models in the AI Project Manager app, addressing the user's request to "voeg voor alle base models de prompt templates toe" (add prompt templates for all base models).

## Major Features Added

### 1. **Centralized Prompt Template System** ✅
- Added `BASE_MODEL_PROMPT_TEMPLATES` dictionary in `backend/model_manager.py`
- Comprehensive templates for all base models: anythingv5, anything_xl, counterfeitv3, absolutereality, deliberate, stable_diffusion_1_5, stable_diffusion_xl, dreamshaper, kenshi, aam_xl_animemix, mistoon, meina_mix, orange_mix
- Each template includes optimized prefix and negative prompts specific to model type

### 2. **Anything XL Structured Format Implementation** ✅
- Implemented special structured prompt format for Anything XL: `<|quality|>, <|year|>, <|characters|>, <|tags|>`
- Applied to both `anything_xl` and `aam_xl_animemix` models
- Includes quality tags ("masterpiece, best quality") and year tags ("newest" for 2021-2024 style)

### 3. **Pipeline Prompt Optimization Updates** ✅
- Updated ALL channel pipeline `_optimize_video_prompt()` methods:
  - `anime_pipeline.py` - Enhanced with model-specific templates
  - `gaming_pipeline.py` - Added realistic model support
  - `manga_pipeline.py` - Integrated template system
  - `marvel_dc_pipeline.py` - Added comic book optimizations
  - `superhero_pipeline.py` - Enhanced superhero style prompts
  - `original_manga_pipeline.py` - Added unique manga style templates
- All methods now accept `model_name` parameter for template selection

### 4. **Enhanced Model Repository Management** ✅
- Added missing `realesrgan_anime` mapping to HF_MODEL_REPOS
- Integrated CivitAI model 9409 (Anything XL) with correct model_id and version_id
- Improved HuggingFace download function with retry logic and extended timeout (120s)
- Added comprehensive error handling and logging

### 5. **Async Pipeline Implementation** ✅
- Implemented missing `execute_pipeline_async()` method in AsyncPipelineManager
- Added proper channel-specific pipeline routing
- Enhanced error handling and logging for async execution

### 6. **GUI Model Selection Improvements** ✅
- Fixed model filtering to show ALL base models regardless of download status
- Enhanced model selection interface to display all available models
- Improved model compatibility checking

### 7. **Enhanced Scene Processing** ✅
- Improved scene extraction to support YAML, JSON, and plain text formats
- Removed hardcoded 2-scene limitation
- Added better fallback mechanisms for script parsing
- Enhanced error handling for various input formats

## Technical Implementation Details

### Prompt Template Structure
```python
BASE_MODEL_PROMPT_TEMPLATES = {
    "anything_xl": {
        "prefix": "masterpiece, best quality, newest, anime style",
        "structure": "<|quality|>, <|year|>, <|characters|>, <|tags|>",
        "negative": "nsfw, lowres, bad anatomy, bad hands, text, error..."
    },
    "absolutereality": {
        "prefix": "photorealistic, highly detailed, professional photography, 8k uhd, realistic lighting, sharp focus",
        "negative": "cartoon, anime, painting, drawing, illustration, low quality, blurry, out of focus"
    }
    # ... templates for all other models
}
```

### Pipeline Integration
- All `_optimize_video_prompt()` methods now check for model-specific templates
- Structured format models (Anything XL) receive special handling
- Channel-specific enhancements applied after template processing
- Maintains backward compatibility with existing prompt optimization

### Model Categories Covered
- **Anime Models**: anythingv5, anything_xl, counterfeitv3, kenshi, aam_xl_animemix, mistoon, meina_mix, orange_mix
- **Realistic Models**: absolutereality, deliberate
- **Base Models**: stable_diffusion_1_5, stable_diffusion_xl, dreamshaper

## Files Modified
- `backend/model_manager.py` - Added BASE_MODEL_PROMPT_TEMPLATES and CivitAI integration
- `backend/pipelines/channel_specific/anime_pipeline.py` - Enhanced prompt optimization
- `backend/pipelines/channel_specific/gaming_pipeline.py` - Added realistic model support
- `backend/pipelines/channel_specific/manga_pipeline.py` - Integrated template system
- `backend/pipelines/channel_specific/marvel_dc_pipeline.py` - Added template integration
- `backend/pipelines/channel_specific/superhero_pipeline.py` - Enhanced with templates
- `backend/pipelines/channel_specific/original_manga_pipeline.py` - Added template support
- `backend/pipelines/channel_specific/base_pipeline.py` - Enhanced scene enhancement method
- `backend/core/async_pipeline_manager.py` - Implemented missing async method
- `backend/ai_tasks.py` - Improved scene extraction logic
- `gui/project_dialogs.py` - Fixed model filtering
- `requirements.txt` - Added audiocraft dependency

## Testing
- Created comprehensive test script `test_comprehensive_models.py`
- Verified all model templates load correctly
- Tested prompt optimization across all channel types
- Confirmed model repository mappings consistency
- Validated CivitAI model integration

## Impact
- **Quality Improvement**: Each base model now uses optimized prompts for better generation quality
- **User Experience**: All models visible in GUI with proper descriptions
- **Reliability**: Enhanced error handling and retry logic for downloads
- **Performance**: Async pipeline execution for better responsiveness
- **Flexibility**: Support for multiple script formats and comprehensive scene processing

## Special Features
- **Anything XL Integration**: Full support for structured prompt format as requested
- **Multi-format Support**: YAML, JSON, and text script processing
- **Timeout Handling**: Extended timeouts and retry logic for large model downloads
- **Channel Compatibility**: Templates work across all channel types (anime, gaming, manga, etc.)

---

**Link to Devin run:** https://app.devin.ai/sessions/3c6832d57be94a668efaee9ad6bea5ff

**Requested by:** Leon van Os (lvos04@outlook.com)

This comprehensive implementation ensures that ALL base models in the AI Project Manager app have proper prompt templates and optimizations, significantly improving generation quality and user experience across all content types.
