"""Check after the gallery is generated that every example it ran has a screenshot as its thumbnail."""

import functools
import logging
import re
import warnings
from importlib.resources import files
from pathlib import Path

from gallery_tools import build_from_environment, check_coverage, find_examples
from mkdocs.config.defaults import MkDocsConfig
from mkdocs.exceptions import PluginError
from mkdocs.structure.pages import Page
from mkdocs_gallery import glr_path_static

from skcomplex.models import describe

# mkdocs-gallery 0.10.4, its latest release, still reads docstrings through `ast.Str` and `.s`, which Python 3.12
# deprecates, so every parsed example logs two warnings. Silence only those, only from mkdocs-gallery.
for message in (r'ast\.Str is deprecated', r'Attribute s is deprecated'):
    warnings.filterwarnings('ignore', message=message, category=DeprecationWarning, module=r'mkdocs_gallery\.')

LOGGER = logging.getLogger('mkdocs.plugins.skcomplex_gallery')
GALLERY_PAGES = 'generated/gallery/'
RUNNING_TIME = '**Total running time of the script:**'
EMPTY_SUBSECTION = re.compile(r'^## [^\n]*\n\s*<div class="mkd-glr-clear"></div>\s*<!-- \S+ -->\n*', re.MULTILINE)


def _without_empty_sections(nav: list[object]) -> list[object]:
    """Return a navigation without the sections that hold no page, those of the subfields a sample build leaves out."""
    kept: list[object] = []
    for item in nav:
        if isinstance(item, dict):
            sections = {
                title: _without_empty_sections(pages) if isinstance(pages, list) else pages
                for title, pages in item.items()
            }
            section = {title: pages for title, pages in sections.items() if pages != []}
            if section:
                kept.append(section)
        else:
            kept.append(item)
    return kept


@functools.cache
def _models_of_examples() -> dict[str, str]:
    """Return the model of every example, by the name of its script."""
    docs = Path(str(files('gallery_tools'))).parent
    return {example.path.stem: example.model for example in find_examples(docs / 'examples')}


def on_page_markdown(markdown: str, page: Page, **kwargs: object) -> str:
    """Remove the gallery subsections that hold no example, and add its catalog entry to the page of an example.

    Args:
        markdown:
            The Markdown of the page.

        page:
            The page.

        **kwargs:
            The other arguments of the event.

    Returns:
        markdown:
            The Markdown without the empty subsections, with the description and references of the model of an example.
    """
    model = _models_of_examples().get(Path(page.file.src_uri).stem)
    if model is None or not page.file.src_uri.startswith(GALLERY_PAGES):
        return EMPTY_SUBSECTION.sub('', markdown)
    entry = describe(model)
    references = '\n'.join(f'- {reference}' for reference in entry.references)
    head, marker, tail = markdown.partition(RUNNING_TIME)
    return f'{head}## About the model\n\n{entry.description}\n\n## References\n\n{references}\n\n{marker}{tail}'


def on_pre_build(config: MkDocsConfig, **kwargs: object) -> None:
    """Fail the build on a failed example, a run example without a screenshot, or an orphan, and on any gap if strict.

    The checksums of the failed examples and of those without a screenshot are removed first, so the next build runs
    them again.

    Args:
        config:
            The documentation configuration.

        **kwargs:
            The other arguments of the event.

    Raises:
        PluginError:
            A run example failed or has no screenshot, or an example is an orphan, or, in a strict build, a registered
            model has no example or, when every example runs, an example has no screenshot.
    """
    config['nav'] = _without_empty_sections(config['nav'])
    build = build_from_environment(Path(str(files('gallery_tools'))).parent)
    errors = build.failed_pages()
    build.forget(errors)
    if errors:
        names = ', '.join(example.path.stem for example in errors)
        error_msg = f'These gallery examples failed and show an error on their page: {names}.'
        raise PluginError(error_msg)
    failing, waiting = build.missing_screenshots(Path(glr_path_static()) / 'no_image.png')
    build.forget(failing)
    if failing:
        names = ', '.join(example.path.stem for example in failing)
        error_msg = f'These gallery examples ran but have no screenshot as their thumbnail: {names}.'
        raise PluginError(error_msg)
    coverage = check_coverage(build.examples)
    if coverage.orphans:
        error_msg = f'These gallery examples name no registered model or are misplaced: {", ".join(coverage.orphans)}.'
        raise PluginError(error_msg)
    problems = coverage.problems([example.path.stem for example in waiting] if build.run_all else [])
    if problems and config.get('strict'):
        raise PluginError(' '.join(problems))
    for problem in problems:
        LOGGER.warning(problem)
