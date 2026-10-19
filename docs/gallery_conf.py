"""Configure `mkdocs-gallery`: the Rerun scraper before the Matplotlib one, the examples to run, and stale examples.

The Rerun scraper comes first, so the Rerun screenshot is image 1 of every example, which `mkdocs-gallery` takes as
the thumbnail by default, and the analysis figures follow it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from gallery_tools import RerunScraper, build_from_environment

BUILD = build_from_environment(Path(__file__).parent)
BUILD.mark_stale()

conf = {
    'image_scrapers': (RerunScraper(BUILD.backend, BUILD.gallery / '_cache', BUILD.settings), 'matplotlib'),
    'filename_pattern': BUILD.filename_pattern(),
    'reset_modules': ('matplotlib',),
    'thumbnail_size': (400, 280),
    'show_memory': False,
    'min_reported_time': 1,
}
