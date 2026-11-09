"""Scaffold, find, and check the gallery examples.

There is one script per ready-made system, within a budget.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import ast
import functools
import importlib
import inspect
import math
import re
import textwrap
import tomllib
from collections import Counter
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

from skcomplex.core.components.systems import BaseSystem
from skcomplex.systems import all_systems

from ._errors import GalleryError
from ._settings import Budget

SCAFFOLD_MARK = 'This example is scaffolded from the catalog and awaits curation.'
EXAMPLE_PREFIX = 'plot_'
CLASS_WORDS = re.compile(r'(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])')
STEPWISE_DT = 1.0
CONTINUOUS_DT = 0.01
DEFAULT_UNTIL = 10.0
TIME_TOLERANCE = 1e-9
LINE_LENGTH = 120
README = 'README.md'
SCAFFOLD = '''"""
# {title}

{caption}
"""

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from {module} import {class_name}

SYSTEM = '{name}'

observer = RerunObserver(SYSTEM, save_path=f'{{SYSTEM}}.rrd', blueprint='default', interval={interval})
trajectory = simulate({class_name}({parameters}), observers=[observer])
'''


@dataclass(frozen=True)
class Example:
    """Gallery example of one system.

    Args:
        path:
            The path of the script.

        system:
            The name of the system the script sets in its `SYSTEM` constant.

        scaffolded:
            Whether the script is still the scaffold the generator wrote.
    """

    path: Path
    system: str
    scaffolded: bool


@dataclass(frozen=True)
class Coverage:
    """Comparison of the examples with the ready-made systems.

    Args:
        missing:
            The ready-made systems without an example.

        orphans:
            The examples that name no ready-made system, name a system already named by another example, or sit outside
            the folder of their system's field.
    """

    missing: list[str] = field(default_factory=list)
    orphans: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        """Whether every system has exactly one example in its place."""
        return not self.missing and not self.orphans

    def problems(self, without_screenshot: list[str]) -> list[str]:
        """Return the gaps that a strict documentation build reports as errors.

        Args:
            without_screenshot:
                The examples shown with the placeholder instead of a screenshot.

        Returns:
            problems:
                One sentence per gap: the systems without an example, and the examples without a screenshot.

        Examples:
            >>> from gallery_tools import Coverage
            >>> for problem in Coverage(missing=['lorenz']).problems(['plot_sir']):
            ...     print(problem)
            These ready-made systems have no gallery example: lorenz.
            These gallery examples have no screenshot: plot_sir.
        """
        problems = []
        if self.missing:
            problems.append(f'These ready-made systems have no gallery example: {", ".join(self.missing)}.')
        if without_screenshot:
            problems.append(f'These gallery examples have no screenshot: {", ".join(without_screenshot)}.')
        return problems


def _system_of(path: Path) -> str | None:
    """Return the value of the `SYSTEM` constant of a script.

    It returns `None` when the script sets none.
    """
    for node in ast.parse(path.read_text()).body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
            continue
        if any(isinstance(target, ast.Name) and target.id == 'SYSTEM' for target in node.targets):
            return str(node.value.value)
    return None


@functools.cache
def library_systems() -> dict[str, type[BaseSystem]]:
    """Return the class of every ready-made system of the library by its gallery name.

    The gallery name of a class is its name without `System` in snake case, with an acronym kept as one word, so
    `GrayScottSystem` is `gray_scott` and `BrusselatorODESystem` is `brusselator_ode`.

    Returns:
        systems:
            The class of every ready-made system, by gallery name.

    Examples:
        >>> from gallery_tools import library_systems
        >>> library_systems()['lotka_volterra'].__name__
        'LotkaVolterraSystem'
    """
    return {CLASS_WORDS.sub('_', name.removesuffix('System')).lower(): cls for name, cls in all_systems()}


def find_examples(folder: Path) -> list[Example]:
    """Return the gallery examples under a folder, sorted by path.

    Args:
        folder:
            The root of the examples, with one folder per field and subfield.

    Returns:
        examples:
            The scripts named `plot_<name>.py` that set a `SYSTEM` constant.

    Examples:
        >>> from pathlib import Path
        >>> from gallery_tools import find_examples
        >>> examples = find_examples(Path('docs/examples'))
        >>> next(example for example in examples if example.system == 'lorenz').path.parent.name
        'dynamical_systems'
    """
    examples = []
    for path in sorted(folder.rglob(f'{EXAMPLE_PREFIX}*.py')):
        system = _system_of(path)
        if system is not None:
            examples.append(Example(path, system, SCAFFOLD_MARK in ' '.join(path.read_text().split())))
    return examples


def check_coverage(folder: Path, systems: dict[str, type[BaseSystem]] | None = None) -> Coverage:
    """Compare the examples under a folder with the ready-made systems.

    Args:
        folder:
            The root of the examples.

        systems:
            The class of every ready-made system by gallery name, or `None` for the systems of the library.

    Returns:
        coverage:
            The systems without an example and the examples that are orphans.

    Examples:
        >>> from pathlib import Path
        >>> from gallery_tools import check_coverage
        >>> check_coverage(Path('docs/examples')).complete
        True
    """
    systems = library_systems() if systems is None else systems
    fields = {name: system.field for name, system in systems.items()}
    examples = find_examples(folder)
    counts = Counter(example.system for example in examples)
    orphans = []
    for example in examples:
        place = example.path.parent.relative_to(folder).as_posix()
        if example.system not in fields or counts[example.system] > 1 or place != fields[example.system]:
            orphans.append(example.path.relative_to(folder).as_posix())
    missing = sorted(set(fields) - set(counts))
    return Coverage(missing, orphans)


def _titles(inventory: Path) -> dict[str, str]:
    """Return the title of every system of the inventory."""
    data = tomllib.loads(inventory.read_text())
    data.pop('sources', None)
    data.pop('primitives', None)
    data.pop('vocabulary', None)
    return {
        name: entry['title']
        for subfields in data.values()
        for systems in subfields.values()
        for name, entry in systems.items()
    }


def gallery_interval(until: float, dt: float, max_frames: int) -> int:
    """Return the logging interval that keeps a run within a number of frames.

    The interval also logs the last frame of the run.

    Args:
        until:
            The simulated time of the run.

        dt:
            The simulated time between frames.

        max_frames:
            The largest number of logged frames, the first one included.

    Returns:
        interval:
            The smallest interval that keeps the frames within the limit, preferring one that divides the number of
            steps so the final state is logged.

    Examples:
        >>> from gallery_tools import gallery_interval
        >>> gallery_interval(50.0, 0.1, 300), gallery_interval(10.0, 1.0, 300)
        (2, 1)
    """
    steps = max(1, math.ceil(until / dt - TIME_TOLERANCE))
    smallest = max(1, math.ceil(steps / (max_frames - 1)))
    for interval in range(smallest, 2 * smallest + 1):
        if steps % interval == 0:
            return interval
    return smallest


def _settings_of(system: type[BaseSystem]) -> tuple[float, float]:
    """Return the simulated time and the time between frames of a system class.

    They are those of the tuned run of the class.
    """
    built = system()
    settings = built.get_run_settings()
    until = DEFAULT_UNTIL if settings['until'] is None else settings['until']
    dt = settings['dt']
    if dt is None:
        dt = STEPWISE_DT if 'discrete' in built.get_tags().time else CONTINUOUS_DT
    if not isinstance(until, int | float) or not isinstance(dt, int | float):
        error_msg = f'The tuned run of {system.__name__} takes numbers as `until` and `dt`, not {until!r} and {dt!r}.'
        raise GalleryError(error_msg)
    return float(until), float(dt)


def _description_of(system: type[BaseSystem]) -> str:
    """Return the description of a system class, the first line of its docstring."""
    return (inspect.getdoc(system) or '').split('\n', maxsplit=1)[0]


def scaffold_example(name: str, system: type[BaseSystem], title: str, budget: Budget | None = None) -> str:
    """Return the scaffold of the example of a system.

    It holds its title, its caption, its run, and its blueprint.

    Args:
        name:
            The gallery name of the system.

        system:
            The class of the system.

        title:
            The title of the example.

        budget:
            The budget whose frame limit sets the logging interval, or `None` for the default budget.

    Returns:
        script:
            The source of the example.

    Examples:
        >>> from gallery_tools import scaffold_example
        >>> from skcomplex.systems.mathematics.dynamical_systems import LorenzSystem
        >>> print(scaffold_example('lorenz', LorenzSystem, 'Lorenz').splitlines()[-1])
        trajectory = simulate(LorenzSystem(), observers=[observer])
    """
    budget = budget or Budget()
    interval = gallery_interval(*_settings_of(system), budget.max_frames)
    seeded = 'random_state' in inspect.signature(system).parameters
    caption = f'{textwrap.fill(_description_of(system).rstrip(".") + ".", LINE_LENGTH)}\n\n{SCAFFOLD_MARK}'
    return SCAFFOLD.format(
        title=title,
        caption=caption,
        module=system.__module__.rpartition('.')[0],
        class_name=system.__name__,
        name=name,
        interval=interval,
        parameters='random_state=0' if seeded else '',
    )


def _title(name: str) -> str:
    """Return the title of a field or a subfield."""
    return name.replace('_', ' ').capitalize()


def _write_readmes(folder: Path, field_path: str) -> None:
    """Write the gallery headers of a field and a subfield that have none."""
    field_name, subfield = field_path.split('/')
    headers = {
        folder
        / field_name
        / README: f'# {_title(field_name)}\n\nExamples of the systems of `skcomplex.systems.{field_name}`.\n',
        folder / field_path / README: f'## {_title(subfield)}\n',
    }
    for path, text in headers.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)


def scaffold_examples(folder: Path, inventory: Path, *, overwrite: bool = False) -> list[Path]:
    """Write the scaffold of the example of every ready-made system that has none.

    The scaffolds come with the gallery headers.

    Args:
        folder:
            The root of the examples.

        inventory:
            The inventory of systems, which gives the titles.

        overwrite:
            Whether to rewrite examples that are still scaffolds.

    Returns:
        paths:
            The written scripts.
    """
    titles = _titles(inventory)
    existing = {example.system: example for example in find_examples(folder)}
    written = []
    for name, system in library_systems().items():
        example = existing.get(name)
        if example is not None and not (overwrite and example.scaffolded):
            continue
        _write_readmes(folder, system.field)
        path = folder / system.field / f'{EXAMPLE_PREFIX}{name}.py'
        title = titles.get(name, _title(name))
        path.write_text(scaffold_example(name, system, title))
        written.append(path)
    return written


def check_budget(name: str, frames: int, size: int, seconds: float | None, budget: Budget) -> None:
    """Raise an error when an example ran beyond its budget.

    Args:
        name:
            The name of the example.

        frames:
            The number of logged frames.

        size:
            The size in bytes of the recording.

        seconds:
            The run time, or `None` when it was not measured.

        budget:
            The budget.

    Raises:
        GalleryError:
            A limit of the budget is exceeded, named with the example.

    Examples:
        >>> from gallery_tools import Budget, check_budget
        >>> check_budget('plot_lorenz', 301, 1000, 1.0, Budget())
        Traceback (most recent call last):
        ...
        gallery_tools._errors.GalleryError: plot_lorenz exceeds the budget: 301 frames, more than 300.
    """
    exceeded = []
    if frames > budget.max_frames:
        exceeded.append(f'{frames} frames, more than {budget.max_frames}')
    if size > budget.max_bytes:
        exceeded.append(f'a recording of {size} bytes, more than {budget.max_bytes}')
    if seconds is not None and seconds > budget.max_seconds:
        exceeded.append(f'a run of {seconds:.1f} seconds, more than {budget.max_seconds:.0f}')
    if exceeded:
        error_msg = f'{name} exceeds the budget: {"; ".join(exceeded)}.'
        raise GalleryError(error_msg)


def system_sources(system: str) -> list[Path]:
    """Return the source files that can change the screenshots of a system's example.

    The system is a ready-made system. These are the files of the subfield package of the system, then the files of the
    core package, which compiles, runs, and logs every system.

    Args:
        system:
            The gallery name of the system.

    Returns:
        paths:
            The Python files of the subfield package, sorted, followed by those of the core, sorted.

    Examples:
        >>> from gallery_tools import system_sources
        >>> sources = system_sources('lotka_volterra')
        >>> sources[0].parent.name, sources[-1].parent.name
        ('ecology', 'core')
    """
    package = importlib.import_module('skcomplex.systems.' + library_systems()[system].field.replace('/', '.'))
    return [
        *sorted(Path(str(files(package.__name__))).rglob('*.py')),
        *sorted(Path(str(files('skcomplex.core'))).rglob('*.py')),
    ]
