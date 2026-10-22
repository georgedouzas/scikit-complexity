"""Define systems, which compose components and interactions on a space, in a time."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar

from sklearn.base import BaseEstimator

from ..base import ComponentKind
from ._components import Element, Group, Interaction
from ._spaces import Space
from ._time import Time

LEGACY_ARGUMENTS = {
    'elements': 'Pass elements and groups in `components`.',
    'systems': 'Pass nested systems in `components`.',
    'observables': 'Pass observables to `Simulation(observables=...)`, or declare them on the catalog entry.',
    'history': 'Declare the history on the delayed state with `state(history=...)`.',
    'noise': 'Give the convention to each noise law with `Noise(..., convention=...)`.',
    'phases': 'Declare the phases as the schedule of the time, `DiscreteTime(schedule=Phases([...]))`.',
}


class System(BaseEstimator):
    """Composition of components and the interactions between them, on a space, in a time.

    A system is defined by its components, its interactions, its space, and its time, and by nothing about how it
    runs. Components are elements, groups, and nested systems, so systems nest to any depth. An interaction refers to a
    component by its path relative to the system that holds the interaction, so `forest/foxes` names the element
    `foxes` of the nested system `forest`.

    Args:
        label:
            The label of the system. It names the system when it is nested in another one.

        components:
            The elements, groups, and nested systems.

        interactions:
            The interactions between the components.

        space:
            The substrate the components live on, which gives vector states their dimension. Nested systems without a
            space use it too.

        time:
            The time the system evolves in, `ContinuousTime`, `DiscreteTime` with its schedule, or `StaticTime`, or
            `None` to infer it from the laws.

    Raises:
        TypeError:
            An argument of the former signature is given, such as `elements` or `observables`. The message names the
            argument's new home.

    Examples:
        >>> from skcomplex.core.model import System
        >>> from skcomplex.models.biology.ecology import LogisticGrowth, Population
        >>> system = System(
        ...     components=[Population('rabbits', n=10)],
        ...     interactions=[LogisticGrowth('growth', target='rabbits', r=1.0, K=100)],
        ... )
        >>> system
        System(components=1, interactions=1)
    """

    component_kind: ClassVar[ComponentKind] = ComponentKind.SYSTEM

    def __init__(
        self,
        label: str = 'system',
        *,
        components: Sequence['Element | Group | System'] = (),
        interactions: Sequence[Interaction] = (),
        space: Space | None = None,
        time: Time | None = None,
        **legacy: Any,  # noqa: ANN401
    ) -> None:
        if legacy:
            homes = ' '.join(LEGACY_ARGUMENTS.get(name, f'`{name}` is not an argument of System.') for name in legacy)
            error_msg = f'System got the arguments {", ".join(map(repr, legacy))}, which it no longer takes. {homes}'
            raise TypeError(error_msg)
        self.label = label
        self.components = components
        self.interactions = interactions
        self.space = space
        self.time = time

    @property
    def elements(self) -> tuple['Element | Group', ...]:
        """Return the elements and groups among the components, in order."""
        return tuple(component for component in self.components if not isinstance(component, System))

    @property
    def systems(self) -> tuple['System', ...]:
        """Return the nested systems among the components, in order."""
        return tuple(component for component in self.components if isinstance(component, System))

    def __repr__(self, N_CHAR_MAX: int = 700) -> str:
        """Return the label and the number of components and interactions."""
        label = '' if self.label == 'system' else f'{self.label!r}, '
        return f'System({label}components={len(self.components)}, interactions={len(self.interactions)})'


class MesaSystem(System):
    """Existing Mesa model wrapped as a system, so it runs with the framework's runtime and visualization.

    The model is built as `model(**parameters, rng=seed)` and stepped once per unit of time. Its declared parameters
    change live through `Simulation.update_parameters`, which sets the model attribute of the same name.

    Args:
        label:
            The label of the system, which prefixes its parameter and reporter names.

        model:
            The Mesa model class.

        parameters:
            The constructor arguments of the model, which are also its live parameters.

        model_reporters:
            A mapping from reporter name to a function of the model that returns a number.

        agent_reporters:
            A mapping from variable name to an agent attribute name or a function of an agent.

        agents:
            The label of the group that holds the model's agents.

    Examples:
        >>> import mesa
        >>> from skcomplex.core.model import MesaSystem
        >>> MesaSystem('market', model=mesa.Model).agents
        'agents'
    """

    def __init__(
        self,
        label: str = 'system',
        *,
        model: type,
        parameters: Mapping[str, Any] | None = None,
        model_reporters: Mapping[str, Callable[[Any], float]] | None = None,
        agent_reporters: Mapping[str, str | Callable[[Any], float]] | None = None,
        agents: str = 'agents',
    ) -> None:
        super().__init__(label)
        self.model = model
        self.parameters = parameters
        self.model_reporters = model_reporters
        self.agent_reporters = agent_reporters
        self.agents = agents

    def __repr__(self, N_CHAR_MAX: int = 700) -> str:
        """Return the label and the wrapped model class."""
        return f'MesaSystem({self.label!r}, model={self.model.__name__})'
