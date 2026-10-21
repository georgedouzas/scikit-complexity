"""Cache thumbnails under a key computed from the example, its model's and the core's code, Rerun, and the settings."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import dataclasses
import hashlib
import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import rerun as rr

from ._settings import CaptureSettings

THUMBNAIL = 'thumbnail.png'
METADATA = 'metadata.json'


def cache_key(example: Path, sources: Sequence[Path], backend: str, settings: CaptureSettings) -> str:
    """Return the key of the screenshots of an example.

    The key is the SHA-256 of the example source, the sources of the subfield package of its model and of the core
    package, the installed Rerun version, the backend, and the capture settings.

    Args:
        example:
            The path of the example script.

        sources:
            The source files of the subfield package of the example's model and of the core package, as
            `model_sources` returns them.

        backend:
            The capture backend.

        settings:
            The capture settings.

    Returns:
        key:
            The hexadecimal digest.

    Examples:
        >>> from pathlib import Path
        >>> from gallery_tools import CaptureSettings, cache_key, model_sources
        >>> example = Path('docs/examples/mathematics/dynamical_systems/plot_lorenz.py')
        >>> len(cache_key(example, model_sources('lorenz'), 'native', CaptureSettings()))
        64
    """
    digest = hashlib.sha256()
    digest.update(example.read_bytes())
    for path in sources:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    digest.update(rr.__version__.encode())
    digest.update(backend.encode())
    digest.update(json.dumps(dataclasses.asdict(settings), sort_keys=True).encode())
    return digest.hexdigest()


class ScreenshotCache:
    """Folder of cached thumbnails, one subfolder per key.

    Args:
        folder:
            The root of the cache.
    """

    def __init__(self, folder: Path) -> None:
        self.folder = folder

    def get(self, key: str) -> Path | None:
        """Return the cached thumbnail of a key, or `None` when it is missing.

        Args:
            key:
                The cache key.

        Returns:
            thumbnail:
                The cached thumbnail.
        """
        thumbnail = self.folder / key / THUMBNAIL
        return thumbnail if thumbnail.is_file() else None

    def put(self, key: str, thumbnail: Path, metadata: dict[str, Any]) -> Path:
        """Store a thumbnail under a key, with metadata on the run that produced it.

        Args:
            key:
                The cache key.

            thumbnail:
                The thumbnail.

            metadata:
                The example, the model, the backend, the number of frames, and the run time.

        Returns:
            thumbnail:
                The cached copy.
        """
        entry = self.folder / key
        entry.mkdir(parents=True, exist_ok=True)
        if thumbnail.resolve() != (entry / THUMBNAIL).resolve():
            shutil.copyfile(thumbnail, entry / THUMBNAIL)
        (entry / METADATA).write_text(json.dumps(metadata, indent=2, sort_keys=True))
        return entry / THUMBNAIL
