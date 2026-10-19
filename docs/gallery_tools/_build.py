"""Select the examples a documentation build runs, mark stale ones, and check the thumbnails it made."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from ._cache import ScreenshotCache, cache_key
from ._checks import is_placeholder
from ._errors import GalleryError
from ._examples import Example, find_examples, model_sources
from ._settings import BACKENDS, DEFAULT_BACKEND, CaptureSettings

SELECTION_VARIABLE = 'SKCOMPLEX_GALLERY'
BACKEND_VARIABLE = 'SKCOMPLEX_GALLERY_BACKEND'
RUN_ALL = 'all'
NOTHING = r'^$'
ERROR_BLOCK = 'mkd-glr-script-err-disp'


@dataclass(frozen=True)
class GalleryBuild:
    """Examples of one documentation build, with the folders it reads and writes.

    Args:
        examples:
            The root of the example scripts, with one folder per field.

        gallery:
            The root of the generated gallery, with one folder per field.

        run_all:
            Whether to run the scaffolded examples too, or only the curated ones.

        backend:
            The capture backend.

        settings:
            The capture settings.
    """

    examples: Path
    gallery: Path
    run_all: bool = False
    backend: str = DEFAULT_BACKEND
    settings: CaptureSettings = field(default_factory=CaptureSettings)

    @classmethod
    def from_environment(cls, examples: Path, gallery: Path, environment: Mapping[str, str]) -> 'GalleryBuild':
        """Return the build that environment variables select.

        `SKCOMPLEX_GALLERY=all` runs every example, and `SKCOMPLEX_GALLERY_BACKEND` names the capture backend.

        Args:
            examples:
                The root of the example scripts.

            gallery:
                The root of the generated gallery.

            environment:
                The environment variables.

        Returns:
            build:
                The build.

        Raises:
            GalleryError:
                The backend is unknown.

        Examples:
            >>> from pathlib import Path
            >>> from gallery_tools import GalleryBuild
            >>> GalleryBuild.from_environment(Path('e'), Path('g'), {'SKCOMPLEX_GALLERY': 'all'}).run_all
            True
        """
        backend = environment.get(BACKEND_VARIABLE, DEFAULT_BACKEND)
        if backend not in BACKENDS:
            error_msg = f'{BACKEND_VARIABLE} = {backend!r} is not one of {", ".join(BACKENDS)}.'
            raise GalleryError(error_msg)
        return cls(examples, gallery, environment.get(SELECTION_VARIABLE, '') == RUN_ALL, backend)

    @property
    def cache(self) -> ScreenshotCache:
        """Screenshot cache of the build."""
        return ScreenshotCache(self.gallery / '_cache')

    def selected(self) -> list[Example]:
        """Return the examples the build runs: the curated ones, or all of them.

        Returns:
            examples:
                The examples to run.
        """
        return [example for example in find_examples(self.examples) if self.run_all or not example.scaffolded]

    def filename_pattern(self) -> str:
        """Return the `mkdocs-gallery` pattern of the scripts to run.

        Returns:
            pattern:
                A regular expression that matches the paths of the selected examples.
        """
        names = [re.escape(example.path.name) for example in self.selected()]
        return f'/({"|".join(names)})$' if names else NOTHING

    def generated_dir(self, example: Example) -> Path:
        """Return the folder of the gallery pages of an example.

        Args:
            example:
                The example.

        Returns:
            folder:
                The folder of its page, its images, and its thumbnail.
        """
        return self.gallery / example.path.parent.relative_to(self.examples)

    def thumbnail(self, example: Example) -> Path:
        """Return the thumbnail `mkdocs-gallery` makes for an example.

        Args:
            example:
                The example.

        Returns:
            path:
                The path of the thumbnail.
        """
        return self.generated_dir(example) / 'images' / 'thumb' / f'mkd_glr_{example.path.stem}_thumb.png'

    def screenshot(self, example: Example) -> Path:
        """Return the screenshot of the primary view of an example, the first image of its page.

        Args:
            example:
                The example.

        Returns:
            path:
                The path of the screenshot, which exists once the example has run.
        """
        return self.generated_dir(example) / 'images' / f'mkd_glr_{example.path.stem}_001.png'

    def stamp(self, example: Example) -> Path:
        """Return the file that holds the cache key of the screenshots on an example's gallery page.

        Args:
            example:
                The example.

        Returns:
            path:
                The path of the stamp, next to the checksum file of `mkdocs-gallery`.
        """
        return self.generated_dir(example) / f'{example.path.name}.key'

    def mark_stale(self) -> list[Example]:
        """Make `mkdocs-gallery` run again the selected examples whose page does not show the current screenshots.

        `mkdocs-gallery` skips a script whose source has not changed since its last run. A change of the model's code,
        of Rerun, of the backend, or of the capture settings changes the cache key without changing the script, so the
        checksum file of an example is removed when its screenshots are not cached or its page holds another key.

        Returns:
            examples:
                The examples marked as stale.
        """
        stale = []
        for example in self.selected():
            key = cache_key(example.path, model_sources(example.model), self.backend, self.settings)
            stamp = self.stamp(example)
            if self.cache.get(key) is not None and stamp.exists() and stamp.read_text() == key:
                continue
            checksum = self.generated_dir(example) / f'{example.path.name}.md5'
            checksum.unlink(missing_ok=True)
            stale.append(example)
        return stale

    def failed_pages(self) -> list[Example]:
        """Return the selected examples whose gallery page shows an error.

        Returns:
            examples:
                The examples whose page holds the error block of `mkdocs-gallery`.
        """
        failed = []
        for example in self.selected():
            page = self.generated_dir(example) / f'{example.path.stem}.md'
            if page.exists() and ERROR_BLOCK in page.read_text():
                failed.append(example)
        return failed

    def missing_screenshots(self, placeholder: Path) -> tuple[list[Example], list[Example]]:
        """Return the examples whose thumbnail is missing or is the placeholder image.

        Args:
            placeholder:
                The placeholder image of `mkdocs-gallery`.

        Returns:
            examples:
                The selected examples without a screenshot, which fail the build, and the other examples without one.
        """
        selected = {example.path for example in self.selected()}
        failing: list[Example] = []
        waiting: list[Example] = []
        for example in find_examples(self.examples):
            thumbnail = self.thumbnail(example)
            if thumbnail.exists() and not is_placeholder(thumbnail, placeholder):
                continue
            (failing if example.path in selected else waiting).append(example)
        return failing, waiting


def build_from_environment(docs: Path) -> GalleryBuild:
    """Return the gallery build of a documentation folder, as the environment selects it.

    Args:
        docs:
            The documentation folder, with `examples` and `generated/gallery`.

    Returns:
        build:
            The build.
    """
    return GalleryBuild.from_environment(docs / 'examples', docs / 'generated' / 'gallery', os.environ)
