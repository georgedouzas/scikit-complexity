"""Define the registered model: a factory with its metadata and default run settings."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import dataclasses
import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from ..core.errors import ValidationError
from ..core.specs import ObservableSpec
from ..core.system import System
from ..engine.compile import compile_system

type Factory = Callable[..., System]

RUN_SETTINGS = ('until', 'dt', 'solver', 'backend', 'clock', 'random_state', 'checked', 'invariants', 'initial_state')
_FORMALISMS: dict[str, str] = {}
"""Mathematical form of every model compiled so far, by name."""


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
            state name, or `None` for the elements' own.
    """

    name: str
    factory: Factory = dataclasses.field(repr=False)
    field: str
    description: str
    tags: tuple[str, ...] = ()
    references: tuple[str, ...] = ()
    defaults: Mapping[str, object] = dataclasses.field(default_factory=dict)
    observables: Callable[[System], Sequence[ObservableSpec]] | None = dataclasses.field(default=None, repr=False)
    initial_state: Callable[..., Mapping[str, object] | None] | None = dataclasses.field(default=None, repr=False)

    def check(self) -> 'ModelEntry':
        """Validate that the default run settings are known and every argument of the factory has a default.

        Returns:
            entry:
                The entry itself.

        Raises:
            ValidationError:
                A default run setting is unknown, or a factory argument has no default.
        """
        unknown = set(self.defaults) - set(RUN_SETTINGS)
        if unknown:
            error_msg = f'Unknown run settings {sorted(unknown)} for {self.name}. Use {", ".join(RUN_SETTINGS)}.'
            raise ValidationError(error_msg)
        required = [
            parameter.name
            for parameter in inspect.signature(self.factory).parameters.values()
            if parameter.default is inspect.Parameter.empty
            and parameter.kind not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
        ]
        if required:
            names = ', '.join(required)
            error_msg = f'The factory {self.factory.__name__} needs defaults for {names} to be registered.'
            raise ValidationError(error_msg)
        return self

    def initial_state_of(self, **parameters: object) -> dict[str, object]:
        """Return the initial values `run` gives the simulation for arguments of the factory.

        Args:
            **parameters:
                Arguments of the factory that replace its defaults.

        Returns:
            initial_state:
                The initial values by dotted state name, empty for the elements' own.
        """
        if self.initial_state is None:
            return {}
        return dict(self.initial_state(**{**self.parameters, **parameters}) or {})

    def observables_of(self, system: System) -> list[ObservableSpec]:
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
    def parameters(self) -> dict[str, object]:
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
