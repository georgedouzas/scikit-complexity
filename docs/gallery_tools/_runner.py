"""Run gallery examples in parallel processes, capture their screenshots into the cache, and report the runs."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import multiprocessing
import os
import runpy
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skcomplex.core.viz import RerunObserver

from ._build import GalleryBuild
from ._errors import GalleryError
from ._examples import Example
from ._screenshots import finished_observers, thumbnail_of
from ._settings import Budget

SLOWEST = 10


@dataclass(frozen=True)
class RunReport:
    """Report of the run of one example.

    Args:
        example:
            The name of the example script.

        seconds:
            The run time of the example, without the capture.

        frames:
            The number of logged frames.

        size:
            The size in bytes of the recording.

        cached:
            Whether the screenshots came from the cache.

        error:
            The error that failed the example, or `None`.
    """

    example: str
    seconds: float = 0.0
    frames: int = 0
    size: int = 0
    cached: bool = False
    error: str | None = None


def _last_observer(example: Example, variables: dict[str, Any]) -> RerunObserver:
    """Return the last observer of an example that saved a recording."""
    observers = finished_observers(variables)
    if not observers:
        error_msg = f'{example.path.stem} saves no recording. Give its RerunObserver a `save_path`.'
        raise GalleryError(error_msg)
    return observers[-1]


def run_example(example: Example, build: GalleryBuild, budget: Budget) -> RunReport:
    """Run an example in a temporary folder, and capture or take from the cache its thumbnail.

    Matplotlib draws with its non-interactive backend, so the analysis figures of the example open no window.

    Args:
        example:
            The example.

        build:
            The gallery build, with the cache, the backend, and the capture settings.

        budget:
            The budget of the example.

    Returns:
        report:
            The report of the run, with the error that failed it.
    """
    os.environ['MPLBACKEND'] = 'agg'
    start = time.perf_counter()
    with tempfile.TemporaryDirectory() as folder:
        os.chdir(folder)
        try:
            variables = runpy.run_path(str(example.path.absolute()), run_name='__main__')
            seconds = time.perf_counter() - start
            observer = _last_observer(example, variables)
            frames, size = int(observer.logged_frames_), Path(str(observer.save_path)).stat().st_size
            _, cached = thumbnail_of(
                example.path.absolute(),
                example.model,
                observer,
                build.cache,
                backend=build.backend,
                settings=build.settings,
                budget=budget,
                seconds=seconds,
            )
        except Exception as error:  # noqa: BLE001  # the report names any error of the example, which is user code
            return RunReport(example.path.name, time.perf_counter() - start, error=f'{type(error).__name__}: {error}')
    return RunReport(example.path.name, seconds, frames, size, cached)


def run_examples(examples: list[Example], build: GalleryBuild, budget: Budget, jobs: int) -> list[RunReport]:
    """Run examples in parallel processes and return their reports in the order of the examples.

    Args:
        examples:
            The examples.

        build:
            The gallery build.

        budget:
            The budget of every example.

        jobs:
            The number of processes.

    Returns:
        reports:
            One report per example.
    """
    context = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=jobs, mp_context=context) as executor:
        futures = [executor.submit(run_example, example, build, budget) for example in examples]
        return [future.result() for future in futures]


def summarize(reports: list[RunReport]) -> str:
    """Return a summary of runs: the slowest examples with their times, and the failures.

    Args:
        reports:
            The reports of the runs.

    Returns:
        summary:
            The lines of the summary.

    Examples:
        >>> from gallery_tools import RunReport, summarize
        >>> print(summarize([RunReport('plot_lorenz.py', 1.5, 251, 9000)]))
        1 examples, 0 failed, 0 from the cache.
        Slowest examples:
          plot_lorenz.py: 1.5 s, 251 frames, 9000 bytes
    """
    failed = [report for report in reports if report.error is not None]
    lines = [
        f'{len(reports)} examples, {len(failed)} failed, {sum(report.cached for report in reports)} from the cache.',
    ]
    lines.append('Slowest examples:')
    for report in sorted(reports, key=lambda report: -report.seconds)[:SLOWEST]:
        lines.append(f'  {report.example}: {report.seconds:.1f} s, {report.frames} frames, {report.size} bytes')
    if failed:
        lines.append('Failures:')
        lines += [f'  {report.example}: {report.error}' for report in failed]
    return '\n'.join(lines)
