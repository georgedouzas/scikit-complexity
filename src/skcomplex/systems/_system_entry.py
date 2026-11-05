"""Define the registered system: a factory with its metadata and default run settings."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import dataclasses
import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from ..core.components.system import System
from ..core.engine.compile import BaseCompiler, compile_system
from ..core.errors import ValidationError
from ..core.specs import ObservableSpec
from .types import Factory

RUN_SETTINGS = (
    'until',
    'dt',
    'compiler',
    'solver',
    'backend',
    'clock',
    'random_state',
    'checked',
    'invariants',
    'initial_state',
)
_FORMALISMS: dict[str, str] = {}
"""Mathematical form of every system compiled so far, by name."""


@dataclass(frozen=True)
class SystemEntry:
    """Registered system: a factory with its metadata and default run settings.

    Args:
        name:
            The name the system is created by.

        factory:
            The factory that builds the system.

        field:
            The field path of the system, its science and subfield, such as `biology/ecology`.

        description:
            One sentence on what the system shows.

        tags:
            Keywords for filtering.

        references:
            The sources of the system.

        defaults:
            The default run settings: `until` and any `Simulation` argument such as `dt` or `solver`.

        observables:
            A function of the built system that returns its standard observables, or `None` for none.

        initial_state:
            A function of the factory's arguments that returns the initial values `run` gives the simulation, by dotted
            state name, or `None` for the elements' own.

        settings:
            A function of the factory's arguments that returns the run settings that replace `defaults` for these
            arguments, or `None`.
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
    settings: Callable[..., Mapping[str, object]] | None = dataclasses.field(default=None, repr=False)

    def _validated(self) -> 'SystemEntry':
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

    def settings_of(self, **parameters: object) -> dict[str, object]:
        """Return the default run settings for arguments of the factory.

        Args:
            **parameters:
                Arguments of the factory that replace its defaults.

        Returns:
            settings:
                The run settings, `defaults` replaced by what `settings` returns for the arguments.
        """
        if self.settings is None:
            return dict(self.defaults)
        return {**self.defaults, **self.settings(**{**self.parameters, **parameters})}

    def observables_of(self, system: System) -> list[ObservableSpec]:
        """Return the standard observables of a system of the system, which `run` records.

        Args:
            system:
                A system the system's factory returned.

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
        """Mathematical form of the system with its default parameters, inferred by compiling it once."""
        if self.name not in _FORMALISMS:
            compiler = self.defaults.get('compiler')
            if compiler is not None and not isinstance(compiler, BaseCompiler):
                error_msg = f'The default compiler of {self.name} is {compiler!r}, which is not a compiler.'
                raise ValidationError(error_msg)
            _FORMALISMS[self.name] = str(compile_system(self.factory(), compiler=compiler).form)
        return _FORMALISMS[self.name]
