"""Define systems, which compose elements, interactions, and subsystems."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from sklearn.base import BaseEstimator

from ._components import Element, Group, Interaction
from ._spaces import Space


class System(BaseEstimator):
    """Composition of elements, interactions, and child systems.

    An interaction refers to elements by their path relative to the system that holds the interaction, so
    `forest/foxes` names the element `foxes` of the child system `forest`.

    Args:
        label:
            The label of the system. It names the system when it is the child of another one.

        elements:
            The elements and element groups of the system.

        interactions:
            The interactions of the system.

        systems:
            The child systems.

        space:
            The space that gives vector states their dimension. Child systems without a space use it too.

    Examples:
        >>> from skcomplex.model import System
        >>> from skcomplex.models.ecology import LogisticGrowth, Population
        >>> system = System(
        ...     elements=[Population('rabbits', n=10)],
        ...     interactions=[LogisticGrowth('growth', target='rabbits', r=1.0, K=100)],
        ... )
        >>> system
        System(elements=1, interactions=1, systems=0)
    """

    def __init__(
        self,
        label: str = 'system',
        *,
        elements: Sequence[Element | Group] = (),
        interactions: Sequence[Interaction] = (),
        systems: Sequence['System'] = (),
        space: Space | None = None,
    ) -> None:
        self.label = label
        self.elements = elements
        self.interactions = interactions
        self.systems = systems
        self.space = space

    def __repr__(self, N_CHAR_MAX: int = 700) -> str:
        """Return the label and the number of children of each kind."""
        label = '' if self.label == 'system' else f'{self.label!r}, '
        return (
            f'System({label}elements={len(self.elements)}, interactions={len(self.interactions)}, '
            f'systems={len(self.systems)})'
        )


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
        >>> from skcomplex.model import MesaSystem
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
