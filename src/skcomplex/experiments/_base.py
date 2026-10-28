"""Define the bases of designs, which give rows of values, and of experiments, which run a model many times."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from ..core.errors import ValidationError
from ..core.specs import ObservableSpec
from ..core.system import System
from ..engine.compile import compile_system
from ..engine.problems import BaseProblem, contains
from ..engine.runtime import MODELS, Simulation, Trajectory
from ..engine.utils import BaseEstimator
from ._ensembles import _replicate

type Source = str | Callable[..., System] | System
type Factory = Callable[..., System] | System
type Row = dict[str, object]
type Bounds = Mapping[str, object]
_PROBLEMS: dict[tuple[object, ...], BaseProblem] = {}
"""Compiled problems by source, factory arguments, and observables, compiled once per process."""


class BaseDesign(BaseEstimator, ABC):
    """Base of designs, which give rows of values by factory argument or dotted path.

    A design stores its settings, such as the number of rows, in `__init__`. Preparing it on a mapping from name to
    what it samples sets its rows, and running it returns them:

    - a sampled design reads a pair `(low, high)` or a SciPy distribution per name;
    - a grid design reads a sequence of values per name;
    - a rows design reads nothing, because its rows are its setting.

    A child implements `_prepare(bounds)`, which sets `rows_`.

    Attributes:
        rows_:
            One mapping from factory argument or dotted path to value per row, set by `prepare`.

    Examples:
        >>> from skcomplex.experiments import BaseDesign
        >>> class MidpointDesign(BaseDesign):
        ...     def _prepare(self, bounds=None):
        ...         self.rows_ = [{name: sum(bound) / 2 for name, bound in (bounds or {}).items()}]
        >>> MidpointDesign().check().prepare({'r': (0.0, 1.0)}).run()
        [{'r': 0.5}]
    """

    def _check_size(self, n: object) -> None:
        """Raise an error for a number of rows that is not a nonnegative integer."""
        if isinstance(n, bool) or not isinstance(n, int | np.integer) or n < 0:
            error_msg = f'The number of rows of a design must be a nonnegative integer, not {n!r}.'
            raise ValidationError(error_msg)

    def _check_bounds(self, bounds: Bounds | None, *, distributions: bool = True) -> dict[str, object]:
        """Return the bounds by name, raising an error for one that is neither a pair nor an allowed distribution."""
        bounds = dict(bounds or {})
        for name, bound in bounds.items():
            pair = isinstance(bound, Sequence) and not isinstance(bound, str) and len(bound) == 2  # noqa: PLR2004
            if pair or (distributions and hasattr(bound, 'ppf')):
                continue
            kinds = 'a pair (low, high) or a SciPy distribution' if distributions else 'a pair (low, high)'
            error_msg = f'The bounds of {name} must be {kinds}, not {bound!r}.'
            raise ValidationError(error_msg)
        return bounds

    def _scale(self, bounds: Bounds, uniform: npt.NDArray[np.float64]) -> list[Row]:
        """Return rows of uniform samples in the unit cube mapped to pairs of bounds or through quantile functions."""
        columns = {}
        for index, (name, bound) in enumerate(bounds.items()):
            if hasattr(bound, 'ppf'):
                columns[name] = np.asarray(bound.ppf(uniform[:, index]), dtype=float)
            else:
                low, high = (float(item) for item in bound)  # type: ignore[attr-defined]
                columns[name] = low + uniform[:, index] * (high - low)
        return [{name: float(column[row]) for name, column in columns.items()} for row in range(uniform.shape[0])]

    @abstractmethod
    def _prepare(self, bounds: Bounds | None = None) -> None:
        """Set the rows of the design in `rows_` from the mapping of what it samples."""

    def _run(self) -> list[Row]:
        """Return a copy of the prepared rows."""
        return [dict(row) for row in self.rows_]


class BaseExperiment(BaseEstimator, ABC):
    """Base of experiments, which run a model, given by a registered name, a factory, or a system, many times.

    An experiment stores its source and settings in `__init__`, validates them in `check`, resolves the source and
    what it runs in `prepare`, and runs in `run`, which returns the experiment with its results in the attributes that
    end in `_`. A child implements `_prepare` and `_run`, and reads the model with the private methods of the base.

    Examples:
        >>> from skcomplex.experiments import BaseExperiment
        >>> from skcomplex.models.biology.ecology import logistic_growth
        >>> class DefaultsExperiment(BaseExperiment):
        ...     def __init__(self, source):
        ...         self.source = source
        ...
        ...     def _prepare(self):
        ...         self.factory_, self.defaults_ = self._resolve_source(self.source)
        ...
        ...     def _run(self):
        ...         return self
        >>> DefaultsExperiment('logistic_growth').check().prepare().run().defaults_['until']
        20.0
    """

    @abstractmethod
    def _prepare(self) -> None:
        """Resolve the source and what the experiment runs, and set the attributes that end in `_`."""

    @abstractmethod
    def _run(self) -> 'BaseExperiment':
        """Run the experiment, set its results as attributes that end in `_`, and return it."""

    @staticmethod
    def _resolve_source(source: Source) -> tuple[Factory, dict[str, object]]:
        """Return the factory or system of a model source and the default run settings of a registered model."""
        if isinstance(source, str):
            entry = MODELS.resolve(source)
            return entry.factory, dict(entry.defaults)
        if isinstance(source, System) or callable(source):
            return source, {}
        error_msg = f'A model source is a registered model name, a factory, or a system, not {source!r}.'
        raise ValidationError(error_msg)

    @staticmethod
    def _run_replicated(
        source: Source,
        values: Mapping[str, Sequence[object]],
        until: float | None,
        dt: float | None,
        random_state: int | None,
    ) -> tuple[Trajectory, System]:
        """Return one run of a source's system replicated once per value of every path, and the original system."""
        factory, defaults = BaseExperiment._resolve_source(source)
        system = factory if isinstance(factory, System) else factory()
        size = len(next(iter(values.values())))
        options = {key: value for key, value in defaults.items() if key in ('solver', 'backend')}
        step = dt if dt is not None else defaults.get('dt', 1.0)
        end = until if until is not None else defaults.get('until', 10.0)
        simulation = Simulation(_replicate(system, size, values), dt=step, random_state=random_state, **options)
        return simulation.run(until=end).trajectory_, system

    @staticmethod
    def _compile(
        source: Factory,
        arguments: Mapping[str, object],
        system: System | None = None,
        observables: Sequence[ObservableSpec] | str | None = None,
    ) -> BaseProblem:
        """Return the compiled problem of a source with factory arguments, compiled once per process and arguments.

        The observables are given ones, the name of the registered model whose standard observables to use, or `None`.
        """
        if system is not None:
            key: tuple[object, ...] = ('system', id(system))
            built = system
        elif isinstance(source, System):
            if arguments:
                error_msg = (
                    'A system source takes dotted paths of states and parameters, '
                    f'not the arguments {sorted(arguments)}.'
                )
                raise ValidationError(error_msg)
            key, built = ('system', id(source)), source
        else:
            key = (source, tuple(sorted((name, repr(value)) for name, value in arguments.items())))
            built = None
        key = (*key, observables if isinstance(observables, str | None) else tuple(map(id, observables)))
        if key not in _PROBLEMS:
            built = built if built is not None else source(**arguments)
            if isinstance(observables, str):
                specs = MODELS.resolve(observables).observables_of(built)
            else:
                specs = list(observables or ())
            _PROBLEMS[key] = compile_system(built, observables=specs)
        return _PROBLEMS[key]

    @staticmethod
    def _with_values[P: BaseProblem](problem: P, paths: Mapping[str, object]) -> P:
        """Return a copy of a problem whose initial states and parameters take values by dotted path."""
        states, parameters = problem.initial_state.copy(), problem.initial_parameters.copy()
        for name, value in paths.items():
            layout, vector = (problem.states, states) if problem.states.has(name) else (problem.parameters, parameters)
            if not layout.has(name):
                names = ', '.join([*problem.states.names, *problem.parameters.names])
                error_msg = f'{name} is neither a state nor a parameter of the model. Available paths are: {names}.'
                raise ValidationError(error_msg)
            entry = layout.entry(name)
            if not contains(entry.domain, value):
                error_msg = f'{name} = {value!r} is not in its domain {entry.domain}.'
                raise ValidationError(error_msg)
            vector[entry.indices] = entry.encode(value)
        return dataclasses.replace(problem, initial_state=states, initial_parameters=parameters)
