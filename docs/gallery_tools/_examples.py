"""Scaffold, find, and check the gallery examples: one script per registered model, within a budget."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import ast
import importlib
import inspect
import math
import textwrap
import tomllib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from skcomplex.models import ModelEntry, catalog, describe

from ._errors import GalleryError
from ._settings import Budget

SCAFFOLD_MARK = 'This example is scaffolded from the catalog and awaits curation.'
EXAMPLE_PREFIX = 'plot_'
STEPWISE_FORMS = {'discrete map', 'agent-based model', 'lattice dynamics', 'Monte Carlo sampling'}
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

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = '{name}'

observer = RerunObserver(MODEL, save_path=f'{{MODEL}}.rrd', blueprint='default', interval={interval})
simulation = run(MODEL, settings={{'observers': [observer], 'random_state': 0}}{parameters})
'''


@dataclass(frozen=True)
class Example:
    """Gallery example of one model.

    Args:
        path:
            The path of the script.

        model:
            The name of the model the script sets in its `MODEL` constant.

        scaffolded:
            Whether the script is still the scaffold the generator wrote.
    """

    path: Path
    model: str
    scaffolded: bool


@dataclass(frozen=True)
class Coverage:
    """Comparison of the examples with the registered models.

    Args:
        missing:
            The registered models without an example.

        orphans:
            The examples that name no registered model, name a model already named by another example, or sit outside
            the folder of their model's field.
    """

    missing: list[str] = field(default_factory=list)
    orphans: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        """Whether every model has exactly one example in its place."""
        return not self.missing and not self.orphans


def _model_of(path: Path) -> str | None:
    """Return the value of the `MODEL` constant of a script, or `None` when it sets none."""
    for node in ast.parse(path.read_text()).body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
            continue
        if any(isinstance(target, ast.Name) and target.id == 'MODEL' for target in node.targets):
            return str(node.value.value)
    return None


def library_models() -> list[ModelEntry]:
    """Return the registered models of the library, without models registered by other code such as tests.

    Returns:
        entries:
            The registered models whose factory lives in `skcomplex.models`.

    Examples:
        >>> from gallery_tools import library_models
        >>> 'lotka_volterra' in [entry.name for entry in library_models()]
        True
    """
    return [entry for entry in catalog() if entry.factory.__module__.startswith('skcomplex.models.')]


def find_examples(folder: Path) -> list[Example]:
    """Return the gallery examples under a folder, sorted by path.

    Args:
        folder:
            The root of the examples, with one folder per field and subfield.

    Returns:
        examples:
            The scripts named `plot_<name>.py` that set a `MODEL` constant.

    Examples:
        >>> from pathlib import Path
        >>> from gallery_tools import find_examples
        >>> examples = find_examples(Path('docs/examples'))
        >>> next(example for example in examples if example.model == 'lorenz').path.parent.name
        'dynamical_systems'
    """
    examples = []
    for path in sorted(folder.rglob(f'{EXAMPLE_PREFIX}*.py')):
        model = _model_of(path)
        if model is not None:
            examples.append(Example(path, model, SCAFFOLD_MARK in ' '.join(path.read_text().split())))
    return examples


def check_coverage(folder: Path, entries: list[ModelEntry] | None = None) -> Coverage:
    """Compare the examples under a folder with the registered models.

    Args:
        folder:
            The root of the examples.

        entries:
            The registered models, or `None` for the models of the library.

    Returns:
        coverage:
            The models without an example and the examples that are orphans.

    Examples:
        >>> from pathlib import Path
        >>> from gallery_tools import check_coverage
        >>> check_coverage(Path('docs/examples')).complete
        True
    """
    entries = library_models() if entries is None else entries
    fields = {entry.name: entry.field for entry in entries}
    examples = find_examples(folder)
    counts = Counter(example.model for example in examples)
    orphans = []
    for example in examples:
        place = example.path.parent.relative_to(folder).as_posix()
        if example.model not in fields or counts[example.model] > 1 or place != fields[example.model]:
            orphans.append(example.path.relative_to(folder).as_posix())
    missing = sorted(set(fields) - set(counts))
    return Coverage(missing, orphans)


def _titles(inventory: Path) -> dict[str, str]:
    """Return the title of every model of the inventory."""
    data = tomllib.loads(inventory.read_text())
    data.pop('sources', None)
    return {
        name: entry['title']
        for subfields in data.values()
        for models in subfields.values()
        for name, entry in models.items()
    }


def gallery_interval(until: float, dt: float, max_frames: int) -> int:
    """Return the logging interval that keeps a run within a number of frames and logs its last frame.

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


def _settings_of(entry: ModelEntry) -> tuple[float, float]:
    """Return the simulated time and the time between frames of a model's default run."""
    until = float(entry.defaults.get('until', DEFAULT_UNTIL))
    dt = entry.defaults.get('dt')
    if dt is None:
        dt = STEPWISE_DT if entry.formalism in STEPWISE_FORMS else CONTINUOUS_DT
    return until, float(dt)


def scaffold_example(entry: ModelEntry, title: str, budget: Budget | None = None) -> str:
    """Return the scaffold of the example of a model: its title, its caption, its run, and its blueprint.

    Args:
        entry:
            The registered model.

        title:
            The title of the example.

        budget:
            The budget whose frame limit sets the logging interval, or `None` for the default budget.

    Returns:
        script:
            The source of the example.

    Examples:
        >>> from gallery_tools import scaffold_example
        >>> from skcomplex.models import describe
        >>> print(scaffold_example(describe('lorenz'), 'Lorenz').splitlines()[-1])
        simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
    """
    budget = budget or Budget()
    interval = gallery_interval(*_settings_of(entry), budget.max_frames)
    seeded = 'random_state' in inspect.signature(entry.factory).parameters
    caption = f'{textwrap.fill(entry.description.rstrip(".") + ".", LINE_LENGTH)}\n\n{SCAFFOLD_MARK}'
    return SCAFFOLD.format(
        title=title,
        caption=caption,
        name=entry.name,
        interval=interval,
        parameters=', random_state=0' if seeded else '',
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
        / README: f'# {_title(field_name)}\n\nExamples of the models of `skcomplex.models.{field_name}`.\n',
        folder / field_path / README: f'## {_title(subfield)}\n',
    }
    for path, text in headers.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)


def scaffold_examples(folder: Path, inventory: Path, *, overwrite: bool = False) -> list[Path]:
    """Write the scaffold of the example of every registered model that has none, with the gallery headers.

    Args:
        folder:
            The root of the examples.

        inventory:
            The inventory of models, which gives the titles.

        overwrite:
            Whether to rewrite examples that are still scaffolds.

    Returns:
        paths:
            The written scripts.
    """
    titles = _titles(inventory)
    existing = {example.model: example for example in find_examples(folder)}
    written = []
    for entry in library_models():
        example = existing.get(entry.name)
        if example is not None and not (overwrite and example.scaffolded):
            continue
        _write_readmes(folder, entry.field)
        path = folder / entry.field / f'{EXAMPLE_PREFIX}{entry.name}.py'
        title = titles.get(entry.name, _title(entry.name))
        path.write_text(scaffold_example(entry, title))
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


def model_sources(model: str) -> list[Path]:
    """Return the source files of the subfield package of a registered model.

    Args:
        model:
            The name of the model.

    Returns:
        paths:
            The Python files of the package, sorted.

    Examples:
        >>> from gallery_tools import model_sources
        >>> model_sources('lotka_volterra')[0].parent.name
        'ecology'
    """
    package = importlib.import_module('skcomplex.models.' + describe(model).field.replace('/', '.'))
    return sorted(Path(str(package.__file__)).parent.rglob('*.py'))
