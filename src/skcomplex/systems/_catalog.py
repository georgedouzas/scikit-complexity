"""Register the systems of the library, and list, describe, and create the system or the simulation of them by name."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping, Sequence
from importlib.resources import files
from pathlib import Path

from ..core.components.system import System
from ..core.engine.runtime import Simulation
from ..core.errors import ArgumentError, ValidationError
from ..core.specs import ObservableSpec
from ._system_catalog import SystemCatalog
from ._system_entry import Factory, SystemEntry
from ._system_listing import SystemListing

SYSTEMS = SystemCatalog(Path(str(files('skcomplex.systems'))))
"""The catalog of the systems of the library and of the systems registered at run time."""


def register_system(
    name: str | None = None,
    *,
    field: str,
    description: str | None = None,
    tags: Sequence[str] = (),
    references: Sequence[str] = (),
    defaults: Mapping[str, object] | None = None,
    settings: Callable[..., Mapping[str, object]] | None = None,
    observables: Callable[[System], Sequence[ObservableSpec]] | None = None,
    initial_state: Callable[..., Mapping[str, object] | None] | None = None,
) -> Callable[[Factory], Factory]:
    """Return a decorator that registers a factory as a ready-made system.

    Args:
        name:
            The name of the system, or `None` for the factory's name.

        field:
            The field path of the system, its science and subfield, such as `biology/ecology`.

        description:
            One sentence on what the system shows, or `None` for the first line of the factory's docstring.

        tags:
            Keywords for filtering.

        references:
            The sources of the system.

        defaults:
            The default run settings: `until` and any `Simulation` argument such as `dt` or `solver`.

        settings:
            A function of the factory's arguments that returns the run settings that replace `defaults` for these
            arguments, or `None`.

        observables:
            A function of the system's system that returns its standard observables, which `run` records, or `None`.

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
        >>> from typing import Annotated
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.specs import StateSpec
        >>> from skcomplex.core.components.system import System
        >>> from skcomplex.systems import create_system, register_system
        >>> class PopulationElement(Element):
        ...     def __init__(self, label, n: Annotated[float, StateSpec()] = 1.0):
        ...         super().__init__(label)
        ...         self.n = n
        >>> @register_system(field='examples/demo', references=('This docstring',), defaults={'until': 5})
        ... def constant_population(n: float = 3.0) -> System:
        ...     '''Return one population that does not change.'''
        ...     return System(elements=[PopulationElement('a', n=n)])
        >>> create_system('constant_population').elements[0].n
        3.0
    """

    def decorator(factory: Factory) -> Factory:
        entry = SystemEntry(
            name or factory.__name__,
            factory,
            field,
            description or '',
            tuple(tags),
            tuple(references),
            dict(defaults or {}),
            observables,
            initial_state,
            settings,
        )
        SYSTEMS.register(entry._validated())
        return factory

    return decorator


def catalog(field: str | None = None, tag: str | None = None) -> list[SystemListing]:
    """Return the listed systems, optionally of one field or with one tag, sorted by field and name.

    The listing reads the registration of every system of the library from its source, and imports no system module.
    Systems registered at run time outside the library are listed too.

    Args:
        field:
            The field, such as `biology`, or the field path, such as `biology/ecology`, to keep, or `None` for every
            field.

        tag:
            The tag to keep, or `None` for every system.

    Returns:
        listings:
            The listed systems.

    Raises:
        RegistryError:
            A system registers its name, field, description, tags, or references with a value computed at run time.

    Examples:
        >>> from skcomplex.systems import catalog
        >>> 'lotka_volterra' in [listing.name for listing in catalog(field='biology/ecology')]
        True
        >>> sorted({listing.field for listing in catalog(field='chemistry')})
        ['chemistry/kinetics', 'chemistry/pattern_formation']
    """
    return SYSTEMS.listings(field, tag)


def describe(name: str) -> SystemEntry:
    """Return a system by name, with its factory, parameters, metadata, and default run settings.

    It imports the system's module and nothing else of the library.

    Args:
        name:
            The name of the system.

    Returns:
        entry:
            The registered system.

    Raises:
        RegistryError:
            No system has the name.

    Examples:
        >>> from skcomplex.systems import describe
        >>> describe('lotka_volterra').field
        'biology/ecology'
    """
    return SYSTEMS.entry(name)


def create_system(name: str, **parameters: object) -> System:
    """Return the system of a system, built by its factory.

    It imports the system's module and nothing else of the library.

    Args:
        name:
            The name of the system.

        **parameters:
            Arguments of the factory that replace its defaults.

    Returns:
        system:
            The system's system.

    Raises:
        RegistryError:
            No system has the name.

    Examples:
        >>> from skcomplex.systems import create_system
        >>> len(create_system('lotka_volterra').elements)
        2
    """
    return SYSTEMS.entry(name).factory(**parameters)


def create_simulation(name: str, **parameters: object) -> Simulation:
    """Create the simulation of a system with its default run settings, ready to prepare and run.

    It imports the system's module and nothing else of the library. The simulation holds the system as its `system`,
    the system's default `until`, time step, solver, observables and initial state, and the seed `random_state=0`.
    Change any of them with `set_params` before preparing it.

    Args:
        name:
            The name of the system.

        **parameters:
            Arguments of the factory that replace its defaults.

    Returns:
        simulation:
            The unprepared simulation of the system.

    Raises:
        RegistryError:
            No system has the name.

        ValidationError:
            The system's setting `initial_state` is not a mapping.

        ArgumentError:
            A default setting of the system is not an argument of `Simulation`.

    Examples:
        >>> from skcomplex.systems import create_simulation
        >>> simulation = create_simulation('logistic_growth')
        >>> simulation.until
        20.0
        >>> simulation.prepare().run().time_
        20.0
    """
    entry = SYSTEMS.entry(name)
    chosen: dict[str, object] = {'random_state': 0, 'until': 10.0, **entry.settings_of(**parameters)}
    initial_state = chosen.pop('initial_state', None) or {}
    if not isinstance(initial_state, Mapping):
        error_msg = f'The run setting `initial_state` of {name} is {initial_state!r}. Give it as a mapping.'
        raise ValidationError(error_msg)
    system = entry.factory(**parameters)
    initial = {**entry.initial_state_of(**parameters), **initial_state}
    simulation = Simulation(system, observables=entry.observables_of(system), initial_state=initial)
    unknown = sorted(set(chosen) - set(simulation.get_params(deep=False)))
    if unknown:
        error_msg = f'The system {name} has a run setting {unknown[0]!r}, which is not an argument of `Simulation`.'
        raise ArgumentError(error_msg)
    return simulation.set_params(**chosen)
