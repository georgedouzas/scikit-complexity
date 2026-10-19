"""Turn the finished Rerun observer of an example into a checked and cached thumbnail."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import rerun as rr

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver

from ._cache import ScreenshotCache, cache_key
from ._capture import capture_thumbnail
from ._errors import GalleryError
from ._examples import check_budget, model_sources
from ._settings import Budget, CaptureSettings


def finished_observers(variables: Mapping[str, Any]) -> list[RerunObserver]:
    """Return the Rerun observers of an example's variables that saved a recording with a blueprint.

    Observers are found as variables and as observers of simulations.

    Args:
        variables:
            The global variables of the example.

    Returns:
        observers:
            The observers that logged at least one frame to a file and sent a blueprint.

    Examples:
        >>> from gallery_tools import finished_observers
        >>> from skcomplex.core.viz import RerunObserver
        >>> finished_observers({'observer': RerunObserver(save_path='run.rrd', blueprint='default')})
        []
    """
    candidates = [value for value in variables.values() if isinstance(value, RerunObserver)]
    for value in variables.values():
        if isinstance(value, Simulation):
            candidates += [item for item in value.observers if isinstance(item, RerunObserver)]
    found: list[RerunObserver] = []
    for observer in candidates:
        if observer in found or observer.save_path is None or getattr(observer, 'logged_frames_', 0) == 0:
            continue
        if getattr(observer, 'blueprint_', None) is None:
            error_msg = (
                'The example saves a recording without a blueprint. Pass `blueprint=\'default\'` or a blueprint.'
            )
            raise GalleryError(error_msg)
        found.append(observer)
    return found


def thumbnail_of(
    example: Path,
    model: str,
    observer: RerunObserver,
    cache: ScreenshotCache,
    *,
    backend: str,
    settings: CaptureSettings,
    budget: Budget,
    seconds: float | None = None,
) -> tuple[Path, bool]:
    """Return the thumbnail of an example from the cache, or capture, check, and cache it.

    Args:
        example:
            The path of the example script.

        model:
            The name of the model the example runs.

        observer:
            The finished observer that saved the example's recording.

        cache:
            The screenshot cache.

        backend:
            The capture backend.

        settings:
            The capture settings.

        budget:
            The budget the run must stay within.

        seconds:
            The run time of the example, or `None` when it was not measured.

    Returns:
        thumbnail:
            The cached thumbnail, and whether it came from the cache.

    Raises:
        GalleryError:
            The run exceeds the budget, or the capture fails its checks.
    """
    recording = Path(str(observer.save_path)).absolute()
    frames = int(observer.logged_frames_)
    check_budget(example.stem, frames, recording.stat().st_size, seconds, budget)
    key = cache_key(example, model_sources(model), backend, settings)
    cached = cache.get(key)
    if cached is not None:
        return cached, True
    blueprint = observer.blueprint_
    if blueprint is None:
        error_msg = f'{example.stem} sends no blueprint. Pass `blueprint=\'default\'` or a blueprint to its observer.'
        raise GalleryError(error_msg)
    with tempfile.TemporaryDirectory() as folder:
        try:
            thumbnail = capture_thumbnail(
                recording,
                blueprint,
                observer.application_id,
                Path(folder),
                backend=backend,
                settings=settings,
            )
        except GalleryError as error:
            error_msg = f'{example.stem}: {error}'
            raise GalleryError(error_msg) from error
        metadata = {
            'example': example.name,
            'model': model,
            'backend': backend,
            'frames': frames,
            'bytes': recording.stat().st_size,
            'seconds': seconds,
            'rerun': rr.__version__,
        }
        return cache.put(key, thumbnail, metadata), False
