"""Check after the gallery is generated that every example it ran has a screenshot as its thumbnail."""

import logging
import sys
import warnings
from pathlib import Path
from typing import Any

from mkdocs.exceptions import PluginError
from mkdocs_gallery import glr_path_static

sys.path.insert(0, str(Path(__file__).parent))

from gallery_tools import build_from_environment, check_coverage

# mkdocs-gallery 0.10.4, its latest release, still reads docstrings through `ast.Str` and `.s`, which Python 3.12
# deprecates, so every parsed example logs two warnings. Silence only those, only from mkdocs-gallery.
for message in (r'ast\.Str is deprecated', r'Attribute s is deprecated'):
    warnings.filterwarnings('ignore', message=message, category=DeprecationWarning, module=r'mkdocs_gallery\.')

LOGGER = logging.getLogger('mkdocs.plugins.skcomplex_gallery')


def on_pre_build(config: Any, **kwargs: Any) -> None:  # noqa: ANN401
    """Fail the build on a run example without a screenshot, and report the rest of the coverage.

    Args:
        config:
            The documentation configuration.

        **kwargs:
            The other arguments of the event.

    Raises:
        PluginError:
            A run example failed or has no screenshot, or an example is an orphan.
    """
    build = build_from_environment(Path(__file__).parent)
    errors = build.failed_pages()
    if errors:
        names = ', '.join(example.path.stem for example in errors)
        error_msg = f'These gallery examples failed and show an error on their page: {names}.'
        raise PluginError(error_msg)
    failing, waiting = build.missing_screenshots(Path(glr_path_static()) / 'no_image.png')
    if failing:
        names = ', '.join(example.path.stem for example in failing)
        error_msg = f'These gallery examples ran but have no screenshot as their thumbnail: {names}.'
        raise PluginError(error_msg)
    coverage = check_coverage(build.examples)
    if coverage.orphans:
        error_msg = f'These gallery examples name no registered model or are misplaced: {", ".join(coverage.orphans)}.'
        raise PluginError(error_msg)
    LOGGER.info(
        'Gallery coverage: %d models without an example, %d examples not run and shown with the placeholder.',
        len(coverage.missing),
        len(waiting),
    )
