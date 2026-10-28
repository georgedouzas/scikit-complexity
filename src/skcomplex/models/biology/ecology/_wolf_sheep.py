"""Ecology models: population growth, predation, and invasion fronts."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement
from ....core.laws import BaseLaw, RuleLaw
from ....core.spaces import LatticeSpace
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ....core.system import System
from ....engine.solvers import AgentView, ModelView
from ..._catalog import register_model

GROWN_SHARE = 0.5


def _wander(agent: AgentView, model: ModelView) -> None:
    """Move an animal to a random neighbouring cell and spend one unit of energy."""
    cells = agent.neighbor_cells()
    agent.move_to(cells[int(model.rng.integers(len(cells)))])
    agent.energy -= 1


def _live(agent: AgentView, model: ModelView, reproduction: float) -> None:
    """Remove an animal without energy, or split its energy with an offspring at a reproduction rate."""
    if agent.energy < 0:
        agent.remove()
    elif model.rng.random() < reproduction:
        agent.energy /= 2
        agent.create(energy=agent.energy)


class GrassElement(Element):
    """Grass patch that regrows a fixed number of steps after it is eaten.

    Args:
        label:
            The label of the agent or group.

        grown:
            Whether the grass is grown, 1 or 0.

        countdown:
            The steps left until the grass is grown again.

        regrowth:
            The number of steps the grass takes to regrow.

    Examples:
        >>> from skcomplex.models.biology.ecology import GrassElement
        >>> GroupElement('grass', element=GrassElement, size=4, values={'grown': 1.0}).size
        4
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec(
            'agents',
            {'color_by': 'grown', 'color_range': (0.0, 1.0), 'radius': 0.5},
        ),
        *,
        grown: Annotated[float, StateSpec(domain=NN)] = 1.0,
        countdown: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        regrowth: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 30.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.grown = grown
        self.countdown = countdown
        self.regrowth = regrowth

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the regrowth rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.regrow)]

    def regrow(self, agent: AgentView, model: ModelView) -> None:
        """Count down eaten grass and grow it back when the countdown ends.

        Args:
            agent:
                The patch.

            model:
                The model.
        """
        del model
        if agent.grown:
            return
        agent.countdown -= 1
        if agent.countdown <= 0:
            agent.grown = 1.0
            agent.countdown = agent.regrowth


class SheepElement(Element):
    """Sheep that wanders, eats grown grass, reproduces, and dies without energy.

    Args:
        label:
            The label of the agent or group.

        energy:
            The initial energy.

        gain:
            The energy a sheep gains from grown grass.

        reproduction:
            The probability of reproducing at every step.

    Examples:
        >>> from skcomplex.models.biology.ecology import SheepElement
        >>> GroupElement('sheep', element=SheepElement, size=100).size
        100
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'radius': 0.3}),
        *,
        energy: Annotated[float, StateSpec()] = 8.0,
        gain: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 4.0,
        reproduction: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.04,
    ) -> None:
        super().__init__(label, visual=visual)
        self.energy = energy
        self.gain = gain
        self.reproduction = reproduction

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the life rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.graze)]

    def graze(self, agent: AgentView, model: ModelView) -> None:
        """Wander, eat the grass of the new cell when it is grown, then reproduce or die.

        Args:
            agent:
                The sheep.

            model:
                The model.
        """
        _wander(agent, model)
        for patch in agent.agents_here('grass'):
            if patch.grown:
                patch.grown = 0.0
                patch.countdown = patch.regrowth
                agent.energy += agent.gain
        _live(agent, model, agent.reproduction)


class WolfElement(Element):
    """Wolf that wanders, eats a sheep of its cell, reproduces, and dies without energy.

    Args:
        label:
            The label of the agent or group.

        energy:
            The initial energy.

        gain:
            The energy a wolf gains from a sheep.

        reproduction:
            The probability of reproducing at every step.

    Examples:
        >>> from skcomplex.models.biology.ecology import WolfElement
        >>> GroupElement('wolves', element=WolfElement, size=25).size
        25
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'radius': 0.35}),
        *,
        energy: Annotated[float, StateSpec()] = 20.0,
        gain: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 20.0,
        reproduction: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
    ) -> None:
        super().__init__(label, visual=visual)
        self.energy = energy
        self.gain = gain
        self.reproduction = reproduction

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the life rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.hunt)]

    def hunt(self, agent: AgentView, model: ModelView) -> None:
        """Wander, eat one sheep of the new cell when there is one, then reproduce or die.

        Args:
            agent:
                The wolf.

            model:
                The model.
        """
        _wander(agent, model)
        prey = agent.agents_here('sheep')
        if prey:
            prey[int(model.rng.integers(len(prey)))].remove()
            agent.energy += agent.gain
        _live(agent, model, agent.reproduction)


@register_model(
    field='biology/ecology',
    tags=('predator-prey', 'agents', 'grid'),
    references=('Wilensky 1997',),
    defaults={'until': 100},
)
def wolf_sheep(
    width: int = 20,
    height: int = 20,
    *,
    sheep: int = 100,
    wolves: int = 25,
    regrowth: float = 30.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the wolf and sheep predation model with one grass patch per cell.

    Args:
        width:
            The number of columns.

        height:
            The number of rows.

        sheep:
            The initial number of sheep.

        wolves:
            The initial number of wolves.

        regrowth:
            The number of steps grass takes to regrow.

        random_state:
            The seed or generator of the initial grass state.

    Returns:
        system:
            The model, with the groups `grass`, `sheep`, and `wolves`.

    Examples:
        >>> from skcomplex.models.biology.ecology import wolf_sheep
        >>> [group.label for group in wolf_sheep(10, 10).elements]
        ['grass', 'sheep', 'wolves']
    """
    rng = np.random.default_rng(random_state)
    cells = [(column, row) for column in range(width) for row in range(height)]
    grown = (rng.random(len(cells)) < GROWN_SHARE).astype(float)
    countdown = np.where(grown > 0, regrowth, rng.uniform(0, regrowth, len(cells)))
    grass = GroupElement(
        'grass',
        element=GrassElement,
        size=len(cells),
        values={'grown': grown, 'countdown': countdown, 'regrowth': regrowth},
        variable=True,
        location=cells,
    )
    return System(
        elements=[
            grass,
            GroupElement('sheep', element=SheepElement, size=sheep, variable=True),
            GroupElement('wolves', element=WolfElement, size=wolves, variable=True),
        ],
        space=LatticeSpace((width, height), boundary='periodic'),
    )
