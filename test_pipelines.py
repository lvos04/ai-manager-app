import sys
sys.path.append('backend')

try:
    from backend.pipelines.channel_specific.anime_pipeline import AnimeChannelPipeline
    from backend.pipelines.channel_specific.gaming_pipeline import GamingChannelPipeline
    from backend.pipelines.channel_specific.superhero_pipeline import SuperheroChannelPipeline
    from backend.pipelines.channel_specific.manga_pipeline import MangaChannelPipeline
    from backend.pipelines.channel_specific.marvel_dc_pipeline import MarvelDCChannelPipeline
    from backend.pipelines.channel_specific.original_manga_pipeline import OriginalMangaChannelPipeline
    from backend.core.async_pipeline_manager import AsyncPipelineManager
    print('All pipeline imports successful')
    
    anime = AnimeChannelPipeline()
    if hasattr(anime, 'execute_async'):
        print('execute_async method exists')
    else:
        print('execute_async method missing')
        
    manager = AsyncPipelineManager()
    print('AsyncPipelineManager created successfully')
    
except Exception as e:
    print(f'Import error: {e}')
    import traceback
    traceback.print_exc()
