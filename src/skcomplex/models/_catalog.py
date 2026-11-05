"""Register the models of the library, and list, describe, create, and run them by name."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping, Sequence
from importlib.resources import files
from pathlib import Path

from ..core.system import System
from ..engine.runtime import Simulation
from ..errors import ArgumentError, ValidationError
from ..specs import ObservableSpec
from ._model_catalog import ModelCatalog
from ._model_entry import Factory, ModelEntry
from ._model_listing import ModelListing

MODELS = ModelCatalog(Path(str(files('skcomplex.models'))))
"""The catalog of the models of the library and of the models registered at run time."""


def register_model(
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

        settings:
            A function of the factory's arguments that returns the run settings that replace `defaults` for these
            arguments, or `None`.

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
        >>> from typing import Annotated
        >>> from skcomplex.core.elements import Element
        >>> from skcomplex.specs import StateSpec
        >>> from skcomplex.core.system import System
        >>> from skcomplex.models import create, register_model
        >>> class PopulationElement(Element):
        ...     def __init__(self, label, n: Annotated[float, StateSpec()] = 1.0):
        ...         super().__init__(label)
        ...         self.n = n
        >>> @register_model(field='examples/demo', references=('This docstring',), defaults={'until': 5})
        ... def constant_population(n: float = 3.0) -> System:
        ...     '''Return one population that does not change.'''
        ...     return System(elements=[PopulationElement('a', n=n)])
        >>> create('constant_population').elements[0].n
        3.0
    """

    def decorator(factory: Factory) -> Factory:
        entry = ModelEntry(
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
        MODELS.register(entry._validated())
        return factory

    return decorator


def catalog(field: str | None = None, tag: str | None = None) -> list[ModelListing]:
    """Return the listed models, optionally of one field or with one tag, sorted by field and name.

    The listing reads the registration of every model of the library from its source, and imports no model module.
    Models registered at run time outside the library are listed too.

    Args:
        field:
            The field, such as `biology`, or the field path, such as `biology/ecology`, to keep, or `None` for every
            field.

        tag:
            The tag to keep, or `None` for every model.

    Returns:
        listings:
            The listed models.

    Raises:
        RegistryError:
            A model registers its name, field, description, tags, or references with a value computed at run time.

    Examples:
        >>> from skcomplex.models import catalog
        >>> 'lotka_volterra' in [listing.name for listing in catalog(field='biology/ecology')]
        True
        >>> sorted({listing.field for listing in catalog(field='chemistry')})
        ['chemistry/kinetics', 'chemistry/pattern_formation']
    """
    return MODELS.listings(field, tag)


def describe(name: str) -> ModelEntry:
    """Return a model by name, with its factory, parameters, metadata, and default run settings.

    It imports the model's module and nothing else of the library.

    Args:
        name:
            The name of the model.

    Returns:
        entry:
            The registered model.

    Raises:
        RegistryError:
            No model has the name.

    Examples:
        >>> from skcomplex.models import describe
        >>> describe('lotka_volterra').field
        'biology/ecology'
    """
    return MODELS.entry(name)


def create(name: str, **parameters: object) -> System:
    """Return the system of a model, built by its factory.

    It imports the model's module and nothing else of the library.

    Args:
        name:
            The name of the model.

        **parameters:
            Arguments of the factory that replace its defaults.

    Returns:
        system:
            The model's system.

    Raises:
        RegistryError:
            No model has the name.

    Examples:
        >>> from skcomplex.models import create
        >>> len(create('lotka_volterra').elements)
        2
    """
    return MODELS.entry(name).factory(**parameters)


def run(name: str, *, settings: Mapping[str, object] | None = None, **parameters: object) -> Simulation:
    """Create a model and simulate it with its default run settings.

    It imports the model's module and nothing else of the library.

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
        RegistryError:
            No model has the name.

        ValidationError:
            The setting `until` is not a number, or `initial_state` is not a mapping.

        ArgumentError:
            A setting is neither `until` nor an argument of `Simulation`.
    """
    entry = MODELS.entry(name)
    chosen: dict[str, object] = {'random_state': 0, **entry.settings_of(**parameters), **(settings or {})}
    until = chosen.pop('until', 10.0)
    if not isinstance(until, int | float):
        error_msg = f'The run setting `until` of {name} is {until!r}. Give it as a number.'
        raise ValidationError(error_msg)
    initial_state = chosen.pop('initial_state', None) or {}
    if not isinstance(initial_state, Mapping):
        error_msg = f'The run setting `initial_state` of {name} is {initial_state!r}. Give it as a mapping.'
        raise ValidationError(error_msg)
    system = entry.factory(**parameters)
    initial = {**entry.initial_state_of(**parameters), **initial_state}
    simulation = Simulation(system, observables=entry.observables_of(system), initial_state=initial)
    unknown = sorted(set(chosen) - set(simulation.get_params(deep=False)))
    if unknown:
        error_msg = f'No run setting named {unknown[0]!r}. Give `until` or an argument of `Simulation`.'
        raise ArgumentError(error_msg)
    return simulation.set_params(**chosen).prepare().run(until=float(until))
