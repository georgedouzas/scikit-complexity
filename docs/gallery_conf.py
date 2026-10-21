"""Configure `mkdocs-gallery`: the Rerun scraper before the Matplotlib one, the examples to run, and stale examples.

The Rerun scraper comes first, so the Rerun screenshot is image 1 of every example, which `mkdocs-gallery` takes as
the thumbnail by default, and the analysis figures follow it.
"""

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from gallery_tools import RerunScraper, build_from_environment


def quiet_show(gallery_conf: dict[str, Any], file: Path) -> None:
    """Make `plt.show` do nothing while an example runs, since the build draws on Agg and scrapes the figures itself.

    The examples end with `plt.show()` so they open windows when run directly; on Agg it would warn into the output.

    Args:
        gallery_conf:
            The gallery configuration.

        file:
            The example about to run.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    plt.show = lambda *_, **__: None


BUILD = build_from_environment(Path(__file__).parent)
BUILD.mark_stale()

conf = {
    'image_scrapers': (RerunScraper(BUILD.backend, BUILD.gallery / '_cache', BUILD.settings), 'matplotlib'),
    'filename_pattern': BUILD.filename_pattern(),
    'reset_modules': ('matplotlib', quiet_show),
    'thumbnail_size': (400, 280),
    'show_memory': False,
    'min_reported_time': 1,
}
