"""Scrape the Rerun screenshots of a gallery example for `mkdocs-gallery`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import shutil
import tempfile
import time
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skcomplex.core.compile import Problem
from skcomplex.core.viz import RerunObserver

from ._cache import ScreenshotCache
from ._errors import GalleryError
from ._examples import check_budget
from ._screenshots import finished_observers, thumbnail_of
from ._settings import Budget, CaptureSettings

EMBED_CLASS = 'skcomplex-rerun'


@dataclass
class _Redirect:
    """Folder an example runs in, and the folder of its run that its relative recording paths resolve against."""

    source: Path | None = None
    target: Path | None = None


_REDIRECT = _Redirect()
_ON_START = RerunObserver.on_start


def _on_start(observer: RerunObserver, problem: Problem) -> None:
    """Start an observer, moving the relative recording path of an example that runs in its folder to its run."""
    path, target = observer.save_path, _REDIRECT.target
    in_example = Path.cwd().resolve() == _REDIRECT.source
    if target is not None and path is not None and not Path(path).is_absolute() and in_example:
        observer.save_path = target / path
        observer.save_path.parent.mkdir(parents=True, exist_ok=True)
    _ON_START(observer, problem)


def images_markdown(paths: list[Path], script: Any) -> str:  # noqa: ANN401
    """Return the Markdown of images of an example page, one below the other, in the order of the paths.

    Each image is rendered by `figure_md_or_html` of `mkdocs-gallery` with a single path, since its branch for several
    paths leaves a raw HTML list with Markdown inside that is never rendered.

    Args:
        paths:
            The image files, in the images folder of the example's gallery.

        script:
            The gallery script, with its gallery.

    Returns:
        markdown:
            The images as Markdown, separated by blank lines.
    """
    from mkdocs_gallery.scrapers import figure_md_or_html  # noqa: PLC0415  # optional: only the docs build has it

    return '\n\n'.join(figure_md_or_html([path], script) for path in paths)


class RerunScraper:
    """Image scraper of `mkdocs-gallery` that returns the Rerun screenshot of an example's primary view.

    After each code block, the scraper looks for an observer that finished saving a recording. It takes the screenshot
    from the cache or captures it, and copies it to the example's images. The configuration runs this scraper before the
    Matplotlib one, so the screenshot is the first image of the example and becomes its gallery thumbnail. The scraper
    publishes the recording and its blueprint next to the screenshot for the interactive viewer, and writes the cache
    key of the screenshot next to the example's page, so a later build knows which screenshot the page shows.

    The configuration runs `begin` before every example. It starts the clock of the whole script, whose run time
    without the captures must stay within the budget at every code block, and it gives the run a temporary folder of
    its own: an observer started in the example's folder with a relative `save_path` writes its recording there, so
    concurrent builds of one example never take each other's recording.

    Args:
        backend:
            The capture backend, `'native'` or `'web'`.

        cache:
            The folder of the thumbnail cache.

        settings:
            The capture settings, or `None` for the defaults.

        budget:
            The budget of every example, or `None` for the default budget.

    Examples:
        >>> from pathlib import Path
        >>> from gallery_tools import RerunScraper
        >>> RerunScraper('native', Path('cache'))
        RerunScraper(backend='native')
    """

    def __init__(
        self,
        backend: str,
        cache: Path,
        settings: CaptureSettings | None = None,
        budget: Budget | None = None,
    ) -> None:
        self.backend = backend
        self.cache = cache
        self.settings = settings or CaptureSettings()
        self.budget = budget or Budget()
        self._seen: weakref.WeakSet[RerunObserver] = weakref.WeakSet()
        self._start: float | None = None
        self._captured = 0.0
        self._folder: Path | None = None

    def __repr__(self) -> str:
        """Return the backend, which `mkdocs-gallery` shows in its configuration."""
        return f'RerunScraper(backend={self.backend!r})'

    def begin(self, gallery_conf: dict[str, Any], file: Path | None) -> None:
        """Start the clock and the recording folder of an example, as a module resetter of `mkdocs-gallery`.

        Args:
            gallery_conf:
                The gallery configuration.

            file:
                The example about to run, or `None` when no example runs.
        """
        if file is None:
            return
        if self._folder is not None:
            shutil.rmtree(self._folder, ignore_errors=True)
        self._folder = Path(tempfile.mkdtemp(prefix=f'{Path(file).stem}-'))
        _REDIRECT.source, _REDIRECT.target = Path(file).resolve().parent, self._folder
        RerunObserver.on_start = _on_start  # type: ignore[method-assign]
        self._start, self._captured = time.perf_counter(), 0.0

    def _seconds(self) -> float | None:
        """Return the run time of the current example without its captures, or `None` when no example began."""
        return None if self._start is None else time.perf_counter() - self._start - self._captured

    def _publish(self, observer: RerunObserver, folder: Path, stem: str) -> Path:
        """Move the recording and its blueprint next to the example's images, and return the recording."""
        recording = Path(str(observer.save_path)).absolute()
        target = folder / f'{stem}.rrd'
        shutil.move(recording, target)
        blueprint = recording.with_suffix('.rbl')
        if blueprint.exists():
            shutil.move(blueprint, target.with_suffix('.rbl'))
        return target

    def __call__(self, block: Any, script: Any) -> str:  # noqa: ANN401
        """Return the Markdown of the screenshot of the recording a code block finished, or nothing.

        Args:
            block:
                The code block, as `mkdocs-gallery` passes it.

            script:
                The gallery script, with its variables, its image paths, and its gallery.

        Returns:
            markdown:
                The screenshot and the link that opens the recording in the web viewer.

        Raises:
            GalleryError:
                The example names no model, exceeds its budget, its whole script included, or its capture fails its
                checks.
        """
        seconds = self._seconds()
        if seconds is not None:
            check_budget(Path(script.src_py_file).stem, 0, 0, seconds, self.budget)
        variables = script.run_vars.example_globals
        observers = [observer for observer in finished_observers(variables) if observer not in self._seen]
        if not observers:
            return ''
        for observer in observers:
            self._seen.add(observer)
        observer, example = observers[-1], Path(script.src_py_file)
        stamp = Path(script.dwnld_py_file).with_suffix('.py.key')
        stamp.unlink(missing_ok=True)
        model = variables.get('MODEL')
        if not isinstance(model, str):
            error_msg = f'{example.stem} sets no `MODEL` constant. Name the model it runs.'
            raise GalleryError(error_msg)
        start = time.perf_counter()
        thumbnail, _ = thumbnail_of(
            example,
            model,
            observer,
            ScreenshotCache(self.cache),
            backend=self.backend,
            settings=self.settings,
            budget=self.budget,
            seconds=seconds,
        )
        self._captured += time.perf_counter() - start
        path = Path(next(script.run_vars.image_path_iterator))
        shutil.copyfile(thumbnail, path)
        recording = self._publish(observer, path.parent, example.stem)
        stamp.write_text(thumbnail.parent.name)
        link = f'[Open the recording in the Rerun viewer]({path.parent.name}/{recording.name}){{ .{EMBED_CLASS} }}'
        return f'{images_markdown([path], script)}\n\n{link}\n'
