"""Build the model gallery: scaffold examples, capture their Rerun thumbnails, check, cache, and scrape them."""

from ._build import GalleryBuild, build_from_environment
from ._cache import ScreenshotCache, cache_key
from ._capture import (
    CAPTURERS,
    Capturer,
    NativeCapturer,
    WebCapturer,
    capture_thumbnail,
    thumbnail_blueprint,
)
from ._checks import (
    encode_png,
    find_problems,
    has_bands,
    has_light_chrome,
    has_notification,
    is_blank,
    is_placeholder,
    trim_border,
    view_boxes,
)
from ._errors import GalleryError
from ._examples import (
    SCAFFOLD_MARK,
    Coverage,
    Example,
    check_budget,
    check_coverage,
    find_examples,
    gallery_interval,
    library_models,
    model_sources,
    scaffold_example,
    scaffold_examples,
)
from ._runner import RunReport, run_example, run_examples, summarize
from ._scraper import EMBED_CLASS, RerunScraper, images_markdown
from ._screenshots import finished_observers, thumbnail_of
from ._settings import BACKENDS, DEFAULT_BACKEND, Budget, CaptureSettings

__all__: list[str] = [
    'BACKENDS',
    'CAPTURERS',
    'DEFAULT_BACKEND',
    'EMBED_CLASS',
    'SCAFFOLD_MARK',
    'Budget',
    'CaptureSettings',
    'Capturer',
    'Coverage',
    'Example',
    'GalleryBuild',
    'GalleryError',
    'NativeCapturer',
    'RerunScraper',
    'RunReport',
    'ScreenshotCache',
    'WebCapturer',
    'build_from_environment',
    'cache_key',
    'capture_thumbnail',
    'check_budget',
    'check_coverage',
    'encode_png',
    'find_examples',
    'find_problems',
    'finished_observers',
    'gallery_interval',
    'has_bands',
    'has_light_chrome',
    'has_notification',
    'images_markdown',
    'is_blank',
    'is_placeholder',
    'library_models',
    'model_sources',
    'run_example',
    'run_examples',
    'scaffold_example',
    'scaffold_examples',
    'summarize',
    'thumbnail_blueprint',
    'thumbnail_of',
    'trim_border',
    'view_boxes',
]
