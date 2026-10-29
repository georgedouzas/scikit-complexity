"""Define the bases of designs, which give rows of values, and of experiments, which run a model many times."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from ..base import BaseConfig
from ..core.elements import Element, GroupElement
from ..core.laws import (
    BoundaryLaw,
    CrossingLaw,
    EmissionLaw,
    EquationLaw,
    JumpLaw,
    NoiseLaw,
    RateLaw,
    TransferLaw,
    TransitionLaw,
    UpdateLaw,
)
from ..core.system import System
from ..engine._utils import declared_variables
from ..engine.compile import compile_system
from ..engine.compile._probe import probe_laws
from ..engine.problems import BaseProblemSpec
from ..engine.runtime import Simulation, Trajectory
from ..errors import ExperimentError, ValidationError
from ..specs import ObservableSpec
from ..types import Bounds, Row, VariableKind

type Source = Callable[..., System] | System
_PROBLEMS: dict[tuple[object, ...], BaseProblemSpec] = {}
"""Compiled problems by source, factory arguments, and observables, compiled once per process."""
PLAIN_LAWS = (
    RateLaw,
    NoiseLaw,
    JumpLaw,
    UpdateLaw,
    BoundaryLaw,
    CrossingLaw,
    TransitionLaw,
    EmissionLaw,
    TransferLaw,
    EquationLaw,
)
"""Laws of elements that a group of members of the same type runs as one vectorised simulation."""
BOUND_LENGTH = 2
"""Length of a pair `(low, high)` of bounds."""


class BaseDesign(BaseConfig[Bounds, None], ABC):
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
        >>> MidpointDesign().check().check().prepare({'r': (0.0, 1.0)}).run()
        [{'r': 0.5}]
    """

    rows_: list[Row]

    def _check_size(self, n: object) -> None:
        """Raise an error for a number of rows that is not a nonnegative integer."""
        if isinstance(n, bool) or not isinstance(n, int | np.integer) or n < 0:
            error_msg = f'The number of rows of a design must be a nonnegative integer, not {n!r}.'
            raise ValidationError(error_msg)

    def _check_bounds(self, bounds: Bounds | None, *, distributions: bool = True) -> dict[str, object]:
        """Return the bounds by name, raising an error for one that is neither a pair nor an allowed distribution."""
        bounds = dict(bounds or {})
        for name, bound in bounds.items():
            pair = isinstance(bound, Sequence) and not isinstance(bound, str) and len(bound) == BOUND_LENGTH
            if pair or (distributions and hasattr(bound, 'ppf')):
                continue
            kinds = 'a pair (low, high) or a SciPy distribution' if distributions else 'a pair (low, high)'
            error_msg = f'The bounds of {name} must be {kinds}, not {bound!r}.'
            raise ValidationError(error_msg)
        return bounds

    def _check_pairs(self, bounds: Bounds | None) -> dict[str, tuple[float, float]]:
        """Return the pairs `(low, high)` of bounds by name, raising an error for a bound that is not a pair."""
        pairs = {}
        for name, bound in self._check_bounds(bounds, distributions=False).items():
            if isinstance(bound, Sequence):
                low, high = (float(item) for item in bound)
                pairs[name] = (low, high)
        return pairs

    def _scale(self, bounds: Bounds, uniform: npt.NDArray[np.float64]) -> list[Row]:
        """Return rows of uniform samples in the unit cube mapped to pairs of bounds or through quantile functions."""
        columns = {}
        for index, (name, bound) in enumerate(bounds.items()):
            if hasattr(bound, 'ppf'):
                columns[name] = np.asarray(bound.ppf(uniform[:, index]), dtype=float)
            elif isinstance(bound, Sequence):
                low, high = (float(item) for item in bound)
                columns[name] = low + uniform[:, index] * (high - low)
        return [{name: float(column[row]) for name, column in columns.items()} for row in range(uniform.shape[0])]

    @abstractmethod
    def _prepare(self, bounds: Bounds | None = None) -> None:
        """Set the rows of the design in `rows_` from the mapping of what it samples."""

    def _run(self, inputs: None = None) -> list[Row]:
        """Return a copy of the prepared rows."""
        return [dict(row) for row in self.rows_]


class BaseExperiment(BaseConfig[None, None], ABC):
    """Base of experiments, which run a model, given by a factory of systems or a system, many times.

    An experiment stores its source and settings in `__init__`, validates them in `check`, resolves the source and
    what it runs in `prepare`, and runs in `run`, which returns the experiment with its results in the attributes that
    end in `_`. A child implements `_prepare` and `_run`, and reads the model with the private methods of the base. A
    model of the library is passed as its factory, or as a system that `skcomplex.models.create` returns, since
    experiments do not import the model library.

    Examples:
        >>> from skcomplex.experiments import BaseExperiment
        >>> from skcomplex.models.biology.ecology import logistic_growth
        >>> class SystemExperiment(BaseExperiment):
        ...     def __init__(self, source):
        ...         self.source = source
        ...
        ...     def _prepare(self, context):
        ...         self.factory_ = self._check_source(self.source)
        ...
        ...     def _run(self, inputs):
        ...         return self
        >>> len(SystemExperiment(logistic_growth).check().prepare().run().factory_().elements)
        1
    """

    @abstractmethod
    def _prepare(self, context: None = None) -> None:
        """Resolve the source and what the experiment runs, and set the attributes that end in `_`."""

    @abstractmethod
    def _run(self, inputs: None = None) -> 'BaseExperiment':
        """Run the experiment, set its results as attributes that end in `_`, and return it."""

    @staticmethod
    def _check_source(source: object) -> Source:
        """Return a model source, a factory of systems or a system, after checking it.

        Args:
            source:
                The source to check.

        Returns:
            source:
                The source, a factory of systems or a system.

        Raises:
            ExperimentError:
                The source is a model name or neither a factory nor a system.
        """
        if isinstance(source, str):
            error_msg = (
                f'An experiment takes a system or a factory of systems, not the model name {source!r}. Pass the '
                f"model's factory, or the system that `skcomplex.models.create({source!r})` returns."
            )
            raise ExperimentError(error_msg)
        if isinstance(source, System) or callable(source):
            return source
        error_msg = f'A model source is a factory of systems or a system, not {source!r}.'
        raise ExperimentError(error_msg)

    @staticmethod
    def _run_replicated(
        source: Source,
        values: Mapping[str, npt.ArrayLike],
        until: float | None,
        dt: float | None,
        random_state: int | None,
    ) -> tuple[Trajectory, System]:
        """Return one run of a source's system replicated once per value of every path, and the original system."""
        factory = BaseExperiment._check_source(source)
        system = factory if isinstance(factory, System) else factory()
        size = len(np.asarray(next(iter(values.values()))))
        step = dt if dt is not None else 1.0
        end = until if until is not None else 10.0
        replicated = BaseExperiment._replicate(system, size, values)
        simulation = Simulation(replicated, dt=step, random_state=random_state).check()
        return simulation.run(until=end).trajectory_, system

    @staticmethod
    def _compile(
        source: Source,
        arguments: Mapping[str, object],
        system: System | None = None,
        observables: Sequence[ObservableSpec] | None = None,
    ) -> BaseProblemSpec:
        """Return the compiled problem of a source with factory arguments, compiled once per process and arguments."""
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
        key = (*key, tuple(map(id, observables or ())))
        if key not in _PROBLEMS:
            built = built if built is not None else source(**arguments)
            _PROBLEMS[key] = compile_system(built, observables=list(observables or ()))
        return _PROBLEMS[key]

    @staticmethod
    def _with_values[P: BaseProblemSpec](problem: P, paths: Mapping[str, object]) -> P:
        """Return a copy of a problem whose initial states and parameters take values by dotted path."""
        states, parameters = problem.initial_state.copy(), problem.initial_parameters.copy()
        for name, value in paths.items():
            layout, vector = (problem.states, states) if problem.states.has(name) else (problem.parameters, parameters)
            if not layout.has(name):
                names = ', '.join([*problem.states.names, *problem.parameters.names])
                error_msg = f'{name} is neither a state nor a parameter of the model. Available paths are: {names}.'
                raise ValidationError(error_msg)
            entry = layout.entry(name)
            if not entry.contains(value):
                error_msg = f'{name} = {value!r} is not in its domain {entry.domain}.'
                raise ValidationError(error_msg)
            vector[entry.indices] = entry.encode(value)
        return dataclasses.replace(problem, initial_state=states, initial_parameters=parameters)

    @staticmethod
    def _has_formalism_laws(element: Element) -> bool:
        """Check whether an element follows laws other than the plain ones, such as the rules of agents."""
        return any(not isinstance(law, PLAIN_LAWS) for law in probe_laws(element) or [])

    @staticmethod
    def _check_plain(system: System) -> None:
        """Raise an error unless a system holds single elements alone."""
        plain = all(
            isinstance(item, Element) and not BaseExperiment._has_formalism_laws(item) for item in system.elements
        )
        if system.interactions or system.systems or not plain:
            error_msg = (
                f'Only systems of plain single elements replicate, but {system.label!r} has interactions, child '
                'systems, groups, fields, or agents. Build the ensemble as a GroupElement with per-member values '
                'instead.'
            )
            raise ValidationError(error_msg)

    @staticmethod
    def _value_names(element: Element) -> list[str]:
        """Return the names of the states and parameters an element holds, with derivatives of higher order states."""
        return [
            name
            for name, (variable, _) in declared_variables(element).items()
            if variable.kind != VariableKind.ROLE and hasattr(element, name)
        ]

    @staticmethod
    def _member_values(
        values: Mapping[str, npt.ArrayLike],
        system: System,
        size: int,
    ) -> dict[str, dict[str, object]]:
        """Return the per-member values of every element by its label, checked against the elements and the size."""
        by_label: dict[str, dict[str, object]] = {element.label: {} for element in system.elements}
        names = {element.label: set(BaseExperiment._value_names(element)) for element in system.elements}
        for path, value in values.items():
            label, _, name = path.partition('.')
            if label not in names or name not in names[label]:
                available = ', '.join(f'{item}.{field}' for item, fields in names.items() for field in sorted(fields))
                error_msg = f'{path} is not a state or parameter of the system. Available paths are: {available}.'
                raise ValidationError(error_msg)
            array = np.asarray(value)
            if array.ndim == 0 or array.shape[0] != size:
                error_msg = (
                    f'The values of {path} have shape {array.shape}, but they need one row per member, {size} rows.'
                )
                raise ValidationError(error_msg)
            by_label[label][name] = array
        return by_label

    @staticmethod
    def _replicate(system: System, size: int, values: Mapping[str, npt.ArrayLike] | None = None) -> System:
        """Return a system whose single elements become groups of members of the same types and labels.

        Every member starts from the values of its element, except the dotted paths given in `values`, which take one
        value per member. A dotted path such as `map.x` names the same state in the replicated system, where it holds
        one value per member, so many initial conditions or parameter values run as one vectorised simulation.

        Args:
            system:
                The system of single elements, without interactions or child systems.

            size:
                The number of members of every group.

            values:
                A mapping from dotted path of a state or parameter to its per-member values, one row per member.

        Returns:
            system:
                The replicated system, with the observables of the original.

        Raises:
            ValidationError:
                The system has interactions, child systems, groups, fields, or agents, a path is not a state or
                parameter, or its values do not have one row per member.

        Examples:
            >>> import numpy as np
            >>> from skcomplex.experiments import BaseExperiment
            >>> from skcomplex.models.mathematics.dynamical_systems import logistic_map
            >>> ensemble = BaseExperiment._replicate(logistic_map(r=3.7), 4, {'map.x': np.linspace(0.1, 0.4, 4)})
            >>> group = ensemble.elements[0]
            >>> group.label, group.element.__name__, group.size
            ('map', 'LogisticMapElement', 4)
        """
        BaseExperiment._check_plain(system)
        by_label = BaseExperiment._member_values(values or {}, system, size)
        groups = [
            GroupElement(
                element.label,
                element=type(element),
                size=size,
                values={
                    **{name: getattr(element, name) for name in BaseExperiment._value_names(element)},
                    **by_label[element.label],
                },
                visual=element.visual,
            )
            for element in system.elements
        ]
        return System(system.label, elements=groups, space=system.space, time=system.time)
