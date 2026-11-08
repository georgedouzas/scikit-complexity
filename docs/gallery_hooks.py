"""Check after the gallery is generated that every example it ran has a screenshot as its thumbnail."""

import ast
import contextlib
import functools
import inspect
import io
import logging
import pprint
import re
import sys
import warnings
from importlib.machinery import SourceFileLoader
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType

import numpy as np
from gallery_tools import build_from_environment, check_coverage, find_examples, library_systems
from mkdocs.config.defaults import MkDocsConfig
from mkdocs.exceptions import PluginError
from mkdocs.plugins import event_priority
from mkdocs.structure.files import Files
from mkdocs.structure.pages import Page
from mkdocs_gallery import glr_path_static

from skcomplex.core.base import BaseConfig

# mkdocs-gallery 0.10.4, its latest release, still reads docstrings through `ast.Str` and `.s`, which Python 3.12
# deprecates, so every parsed example logs two warnings. Silence only those, only from mkdocs-gallery.
for message in (r'ast\.Str is deprecated', r'Attribute s is deprecated'):
    warnings.filterwarnings('ignore', message=message, category=DeprecationWarning, module=r'mkdocs_gallery\.')

LOGGER = logging.getLogger('mkdocs.plugins.skcomplex_gallery')
GALLERY_PAGES = 'generated/gallery/'
SYSTEMS_NAVIGATION = 'systems/SUMMARY.md'
USER_GUIDE = 'overview/user_guide/'
PARTS = 'parts/'
SHOW_EXPRESSION = '_show_expression'
PYTHON_BLOCK = re.compile(r'^```python(?P<continuation> continuation)?\n(?P<code>.*?)^```$', re.MULTILINE | re.DOTALL)
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
def _systems_of_examples() -> dict[str, str]:
    """Return the system of every example, by the name of its script."""
    docs = Path(str(files('gallery_tools'))).parent
    return {example.path.stem: example.system for example in find_examples(docs / 'examples')}


def _show_expression(value: object) -> None:
    """Print the value of an expression of a block, as a Python prompt does, except for a step that returns its object.

    Args:
        value:
            The value of the expression.
    """
    if value is None or isinstance(value, BaseConfig):
        return
    with np.printoptions(legacy='1.25'):
        text = value if isinstance(value, str) else pprint.pformat(value, width=118, compact=True, sort_dicts=False)
    sys.stdout.write(f'{text}\n')


def _with_shown_expressions(code: str) -> str:
    """Return the source of a block with the value of every expression statement at its top level shown.

    Args:
        code:
            The source of the block.

    Returns:
        code:
            The source that shows the expressions.
    """
    tree = ast.parse(code)
    for statement in tree.body:
        if isinstance(statement, ast.Expr):
            statement.value = ast.Call(ast.Name(SHOW_EXPRESSION, ast.Load()), [statement.value], [])
    return ast.unparse(tree)


def _with_executed_code(markdown: str, page: str) -> str:
    """Return the Markdown of a guide or parts page with the values of the expressions of its Python blocks below them.

    A block runs in a module of its own, registered as a real one because observables pickle the functions of a block,
    and a block that continues another runs in the module of that block. The source stays plain, so it reads and tests
    as it is.

    Args:
        markdown:
            The Markdown of the page.

        page:
            The path of the page, relative to the documentation.

    Returns:
        markdown:
            The Markdown with a code block of its output after each Python block that shows a value.

    Raises:
        PluginError:
            A block fails, so the page cannot show a result that its code does not give.
    """
    prefix = re.sub(r'\W', '_', page.removesuffix('.md'))
    modules = [ModuleType(f'{prefix}_block_0')]

    def executed(block: re.Match[str]) -> str:
        if not block.group('continuation'):
            modules.append(ModuleType(f'{prefix}_block_{len(modules)}'))
        module = modules[-1]
        sys.modules[module.__name__] = module
        vars(module)[SHOW_EXPRESSION] = _show_expression
        output = io.StringIO()
        with TemporaryDirectory() as folder, contextlib.chdir(folder), contextlib.redirect_stdout(output):
            path = Path(folder) / 'block.py'
            path.write_text(_with_shown_expressions(block.group('code')))
            try:
                SourceFileLoader(module.__name__, str(path)).exec_module(module)
            except Exception as error:
                line = markdown[: block.start()].count('\n') + 1
                error_msg = f'The Python block at line {line} of {page} failed: {error}'
                raise PluginError(error_msg) from error
        printed = output.getvalue().rstrip()
        return f'{block.group(0)}\n\n```\n{printed}\n```' if printed else block.group(0)

    return PYTHON_BLOCK.sub(executed, markdown)


def on_page_markdown(markdown: str, page: Page, **kwargs: object) -> str:
    """Run the blocks of a guide or parts page, or tidy a gallery page and describe the system of an example page.

    Args:
        markdown:
            The Markdown of the page.

        page:
            The page.

        **kwargs:
            The other arguments of the event.

    Returns:
        markdown:
            The Markdown without the empty subsections, with the description and references of the system of an example.
    """
    if page.file.src_uri.startswith((USER_GUIDE, PARTS)):
        return _with_executed_code(markdown, page.file.src_uri)
    system = _systems_of_examples().get(Path(page.file.src_uri).stem)
    if system is None or not page.file.src_uri.startswith(GALLERY_PAGES):
        return EMPTY_SUBSECTION.sub('', markdown)
    system_class = library_systems()[system]
    description = (inspect.getdoc(system_class) or '').split('\n', maxsplit=1)[0]
    references = '\n'.join(f'- {reference}' for reference in system_class.references)
    head, marker, tail = markdown.partition(RUNNING_TIME)
    return f'{head}## About the system\n\n{description}\n\n## References\n\n{references}\n\n{marker}{tail}'


@event_priority(-200)
def on_files(files: Files, **kwargs: object) -> Files:
    """Remove the navigation file of the systems, so the build does not publish it as a page.

    The low priority runs this after `literate-nav`, which reads the file at priority -100.

    Args:
        files:
            The files of the documentation.

        **kwargs:
            The other arguments of the event.

    Returns:
        files:
            The files without the navigation file of the systems.
    """
    files.remove(files.src_uris[SYSTEMS_NAVIGATION])
    return files


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
            system has no example or, when every example runs, an example has no screenshot.
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
        error_msg = f'These gallery examples name no registered system or are misplaced: {", ".join(coverage.orphans)}.'
        raise PluginError(error_msg)
    problems = coverage.problems([example.path.stem for example in waiting] if build.run_all else [])
    if problems and config.get('strict'):
        raise PluginError(' '.join(problems))
    for problem in problems:
        LOGGER.warning(problem)
