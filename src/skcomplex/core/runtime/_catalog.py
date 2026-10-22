"""Register ready-made models as factories and list, describe, create, and run them by name."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import dataclasses
import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..base import Registry, ValidationError
from ..compile import compile_system
from ..model import Observable, System
from ._simulations import Simulation

type Factory = Callable[..., System]

RUN_SETTINGS = ('until', 'dt', 'solver', 'backend', 'clock', 'random_state', 'checked', 'initial_state')


@dataclass(frozen=True)
class ModelEntry:
    """Registered model: a factory with its metadata and default run settings.

    Args:
        name:
            The name the model is created by.

        factory:
            The function that returns the model's system.

        field:
            The field path of the model, its science and subfield, such as `biology/ecology`.

        description:
            One sentence on what the model shows.

        tags:
            Keywords for filtering.

        references:
            The sources of the model.

        defaults:
            The default run settings: `until` and any `Simulation` argument such as `dt` or `solver`.

        observables:
            A function of the model's system that returns its standard observables, or `None` for none.

        initial_state:
            A function of the factory's arguments that returns the initial values `run` gives the simulation, by dotted
            state name, or `None` for the components' own.
    """

    name: str
    factory: Factory = dataclasses.field(repr=False)
    field: str
    description: str
    tags: tuple[str, ...] = ()
    references: tuple[str, ...] = ()
    defaults: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    observables: Callable[[System], Sequence[Observable]] | None = dataclasses.field(default=None, repr=False)
    initial_state: Callable[..., Mapping[str, Any] | None] | None = dataclasses.field(default=None, repr=False)

    def initial_state_of(self, **parameters: Any) -> dict[str, Any]:  # noqa: ANN401
        """Return the initial values `run` gives the simulation for arguments of the factory.

        Args:
            **parameters:
                Arguments of the factory that replace its defaults.

        Returns:
            initial_state:
                The initial values by dotted state name, empty for the components' own.
        """
        if self.initial_state is None:
            return {}
        return dict(self.initial_state(**{**self.parameters, **parameters}) or {})

    def observables_of(self, system: System) -> list[Observable]:
        """Return the standard observables of a system of the model, which `run` records.

        Args:
            system:
                A system the model's factory returned.

        Returns:
            observables:
                The observables, labelled by their path from the root.
        """
        return [] if self.observables is None else list(self.observables(system))

    @property
    def parameters(self) -> dict[str, Any]:
        """Parameters of the factory with their defaults."""
        return {
            name: parameter.default
            for name, parameter in inspect.signature(self.factory).parameters.items()
            if parameter.default is not inspect.Parameter.empty
        }

    @property
    def formalism(self) -> str:
        """Mathematical form of the model with its default parameters, inferred by compiling it once."""
        if self.name not in _FORMALISMS:
            _FORMALISMS[self.name] = compile_system(self.factory()).form.value
        return _FORMALISMS[self.name]


MODELS = Registry[ModelEntry]('model')
_FORMALISMS: dict[str, str] = {}


def register_model(
    name: str | None = None,
    *,
    field: str,
    description: str | None = None,
    tags: Sequence[str] = (),
    references: Sequence[str] = (),
    defaults: Mapping[str, Any] | None = None,
    observables: Callable[[System], Sequence[Observable]] | None = None,
    initial_state: Callable[..., Mapping[str, Any] | None] | None = None,
) -> Callable[[Factory], Factory]:
    """Return a decorator that registers a factory as a ready-made model.

    Args:
        name:
            The name of the model, or `None` for the factory's name.

        field:
            The field path of the model, its science and subfield, such as `biology/ecology`.

        description:
            One sentence on what the model shows, or `None` for the first line of the factory's docstring.

        tags:
            Keywords for filtering.

        references:
            The sources of the model.

        defaults:
            The default run settings: `until` and any `Simulation` argument such as `dt` or `solver`.

        observables:
            A function of the model's system that returns its standard observables, which `run` records, or `None`.

        initial_state:
            A function of the factory's arguments that returns the initial values `run` gives the simulation, by dotted
            state name, or `None`.

    Returns:
        decorator:
            The decorator, which returns the factory unchanged.

    Raises:
        ValidationError:
            A default run setting is unknown, or a factory argument has no default.

    Examples:
        >>> from skcomplex.core.model import System
        >>> from skcomplex.models.biology.ecology import Population
        >>> from skcomplex.core.runtime import create, register_model
        >>> @register_model(field='examples/demo', references=('This docstring',), defaults={'until': 5})
        ... def constant_population(n: float = 3.0) -> System:
        ...     '''Return one population that does not change.'''
        ...     return System(components=[Population('a', n=n)])
        >>> create('constant_population').components[0].n
        3.0
    """

    def decorator(factory: Factory) -> Factory:
        unknown = set(defaults or {}) - set(RUN_SETTINGS)
        if unknown:
            error_msg = f'Unknown run settings {sorted(unknown)} for {factory.__name__}. Use {", ".join(RUN_SETTINGS)}.'
            raise ValidationError(error_msg)
        required = [
            parameter.name
            for parameter in inspect.signature(factory).parameters.values()
            if parameter.default is inspect.Parameter.empty
            and parameter.kind not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
        ]
        if required:
            error_msg = f'The factory {factory.__name__} needs defaults for {", ".join(required)} to be registered.'
            raise ValidationError(error_msg)
        summary = description or (inspect.getdoc(factory) or '').split('\n')[0]
        if description is None and summary.startswith('Return '):
            summary = summary.removeprefix('Return ')
            summary = summary[0].upper() + summary[1:]
        entry = ModelEntry(
            name or factory.__name__,
            factory,
            field,
            summary,
            tuple(tags),
            tuple(references),
            dict(defaults or {}),
            observables,
            initial_state,
        )
        MODELS.register(entry.name, entry)
        return factory

    return decorator


def catalog(field: str | None = None, tag: str | None = None) -> list[ModelEntry]:
    """Return the registered models, optionally of one field or with one tag, sorted by field and name.

    Args:
        field:
            The field, such as `biology`, or the field path, such as `biology/ecology`, to keep, or `None` for every
            field.

        tag:
            The tag to keep, or `None` for every model.

    Returns:
        entries:
            The registered models.

    Examples:
        >>> from skcomplex.models import catalog
        >>> 'lotka_volterra' in [entry.name for entry in catalog(field='biology/ecology')]
        True
        >>> sorted({entry.field for entry in catalog(field='chemistry')})
        ['chemistry/kinetics', 'chemistry/pattern_formation']
    """
    entries = [MODELS.resolve(name) for name in MODELS]
    return sorted(
        (
            entry
            for entry in entries
            if (field is None or entry.field == field or entry.field.startswith(f'{field}/'))
            and (tag is None or tag in entry.tags)
        ),
        key=lambda entry: (entry.field, entry.name),
    )


def describe(name: str) -> ModelEntry:
    """Return a registered model by name, with its parameters, metadata, and formalism.

    Args:
        name:
            The name of the model.

    Returns:
        entry:
            The registered model.

    Raises:
        KeyError:
            No model has the name.
    """
    return MODELS.resolve(name)


def create(name: str, **parameters: Any) -> System:  # noqa: ANN401
    """Return the system of a registered model, built by its factory.

    Args:
        name:
            The name of the model.

        **parameters:
            Arguments of the factory that replace its defaults.

    Returns:
        system:
            The model's system.

    Raises:
        KeyError:
            No model has the name.
    """
    return MODELS.resolve(name).factory(**parameters)


def run(name: str, *, settings: Mapping[str, Any] | None = None, **parameters: Any) -> Simulation:  # noqa: ANN401
    """Create a registered model and simulate it with its default run settings.

    Args:
        name:
            The name of the model.

        settings:
            Run settings that replace the model's defaults: `until` and any `Simulation` argument. The run is seeded
            with `random_state=0` unless the settings give another seed, or `None` for a fresh run.

        **parameters:
            Arguments of the factory that replace its defaults.

    Returns:
        simulation:
            The finished simulation, whose `trajectory_` holds the run.

    Raises:
        KeyError:
            No model has the name.
    """
    entry = MODELS.resolve(name)
    chosen = {'random_state': 0, **entry.defaults, **(settings or {})}
    until = chosen.pop('until', 10.0)
    system = create(name, **parameters)
    initial = {**entry.initial_state_of(**parameters), **dict(chosen.pop('initial_state', None) or {})}
    simulation = Simulation(system, observables=entry.observables_of(system), initial_state=initial, **chosen)
    return simulation.run(until=until)
