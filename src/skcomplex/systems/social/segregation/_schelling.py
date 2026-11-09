"""Segregation systems: residential sorting of agents."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import MovementInteraction
from ....core.components.spaces import LatticeSpace
from ....core.components.systems import BaseSystem, System
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ....core.engine.solvers import AgentView, SystemView
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec


class ResidentView(AgentView):
    """View of a resident of the Schelling system.

    Its states and parameters are attributes.
    """

    kind: float
    threshold: float


class ResidentElement(Element):
    """Resident of the Schelling system.

    It moves to an empty cell while too few neighbours share its kind.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        kind:
            The kind of the resident, 0 or 1.

        threshold:
            The smallest share of neighbours of the same kind that keeps a resident in place.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.social.segregation import ResidentElement
        >>> group = GroupElement('residents', element=ResidentElement, size=10, values={'kind': 1.0})
        >>> group.label, group.element.__name__, group.size
        ('residents', 'ResidentElement', 10)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'kind', 'color_range': (0.0, 1.0)}),
        *,
        kind: Annotated[float, StateSpec(domain=NN)] = 0.0,
        threshold: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.kind = kind
        self.threshold = threshold

    @classmethod
    def _get_view(cls) -> type[ResidentView]:
        """Return the class of the views of the agents.

        Their attributes are the states and parameters.
        """
        return ResidentView


class RelocationInteraction(MovementInteraction):
    """Relocation of the Schelling system.

    Every step every resident, in random order, moves to a random empty cell when the share of its neighbours of its own
    kind is below its threshold. With `choices`, the Big Sort of Bishop's hypothesis, every resident moves with the
    probability `probability` whatever its neighbours, to the cell with the highest share of neighbours of its own kind
    among `choices` random empty cells, the first of them on ties.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of residents.

        probability:
            The probability that an unhappy resident moves in a step, or that any resident moves under the Big Sort.

        choices:
            The number of random empty cells a moving resident compares under the Big Sort, or `None` for Schelling's
            rule.

        phase:
            The phase of the step the relocation runs in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.systems.social.segregation import RelocationInteraction
        >>> RelocationInteraction('relocation', target='residents').target
        'residents'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        probability: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 1.0,
        choices: int | None = None,
        phase: str | None = None,
    ) -> None:
        super().__init__(label, target=target, probability=probability, phase=phase)
        self.choices = choices

    def _get_destination(self, agent: AgentViewProtocol, system: SystemViewProtocol, occupied: set[object]) -> object:
        """Return a random empty cell for a resident with too few similar neighbours.

        It is `None` to stay.
        """
        if not isinstance(agent, ResidentView) or not isinstance(system, SystemView):
            return None
        if self.choices is not None:
            return self._get_sorted_destination(agent, system)
        neighbors = [item for item in agent.neighbors() if isinstance(item, ResidentView)]
        if not neighbors:
            return None
        similar = sum(neighbor.kind == agent.kind for neighbor in neighbors) / len(neighbors)
        return system.random_empty_cell() if similar < agent.threshold else None

    def _get_sorted_destination(self, resident: ResidentView, system: SystemView) -> object:
        """Return the empty cell with the highest share of similar neighbours.

        It is chosen among random empty cells.
        """
        storage = system.storage
        if storage.get('time') != system.time:
            storage['time'] = system.time
            storage['kinds'] = {item.location_key: item.kind for item in system.agents(self.target, ResidentView)}
        kinds = storage['kinds']
        if not isinstance(kinds, dict):
            return None
        width, height = system.dimensions or (0, 0)
        best, best_share = None, -1.0
        for _ in range(int(self.choices or 0)):
            cell = system.random_empty_cell()
            column, row = (int(item) for item in np.asarray(cell).ravel())
            around = [
                kinds[column + dx, row + dy]
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
                if (dx or dy)
                and 0 <= column + dx < width
                and 0 <= row + dy < height
                and (column + dx, row + dy) in kinds
            ]
            share = sum(kind == resident.kind for kind in around) / len(around) if around else 0.0
            if share > best_share:
                best, best_share = cell, share
        if best is not None:
            kinds.pop(resident.location_key, None)
            kinds[tuple(int(item) for item in np.asarray(best).ravel())] = resident.kind
        return best


class SchellingSystem(BaseSystem):
    """A Schelling segregation system on a grid with at most one resident per cell.

    Every step every resident whose share of neighbours of its own kind is below the threshold moves to a random empty
    cell. With `choices`, the Big Sort of Downey's Exercise 9-1 after Bishop's hypothesis, residents move for any
    reason, a share `moving` of them every step, and choose among `choices` random empty cells the one with the highest
    share of neighbours of their own kind, so the threshold plays no part. The system has the group `residents`.

    Args:
        width:
            The number of columns.

        height:
            The number of rows.

        density:
            The share of cells with a resident.

        minority:
            The share of residents of kind 1.

        threshold:
            The smallest share of similar neighbours that keeps a resident in place.

        choices:
            The number of random empty cells a moving resident compares under the Big Sort, or `None` for Schelling's
            rule.

        moving:
            The probability that a resident moves in a step under the Big Sort.

        random_state:
            The seed or generator of the kinds.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.social.segregation import SchellingSystem
        >>> SchellingSystem(10, 10, density=0.5).prepare().system_.elements[0].prepare().size_
        50
    """

    field = 'social/segregation'
    topics = ('segregation', 'agents', 'grid')
    references = ('Schelling 1971', 'Downey 2018')
    until = 50

    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        *,
        density: float = 0.8,
        minority: float = 0.3,
        threshold: float = 0.5,
        choices: int | None = None,
        moving: float = 0.1,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.width = width
        self.height = height
        self.density = density
        self.minority = minority
        self.threshold = threshold
        self.choices = choices
        self.moving = moving
        self.random_state = random_state

    def _build_system(self) -> System:
        """Return the parts of the system built from its model parameters."""
        rng = np.random.default_rng(self.random_state)
        size = int(self.width * self.height * self.density)
        kinds = (rng.random(size) < self.minority).astype(float)
        residents = GroupElement(
            'residents',
            element=ResidentElement,
            size=size,
            values={'kind': kinds, 'threshold': self.threshold},
        )
        return System(
            self.label,
            elements=[residents],
            interactions=[
                (
                    RelocationInteraction('relocation', target='residents')
                    if self.choices is None
                    else RelocationInteraction(
                        'relocation',
                        target='residents',
                        probability=self.moving,
                        choices=self.choices,
                    )
                ),
            ],
            space=LatticeSpace((self.width, self.height), boundary='fixed', capacity=1),
        )
