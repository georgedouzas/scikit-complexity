"""Model wolves and sheep that move, eat, reproduce, and die on a grid of regrowing grass."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

import numpy as np

from ...core import Domain, VisualSpec, parameter, state
from ...model import AgentType, GridSpace, Law, Rule, Symbols, System
from ...solvers import AgentView, ModelView

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


class Grass(AgentType):
    """Grass patch that regrows a fixed number of steps after it is eaten.

    Args:
        label:
            The label of the agent type or group.

        location:
            The cell of the patch, or one cell per patch of a group.

        grown:
            Whether the grass is grown, 1 or 0.

        countdown:
            The steps left until the grass is grown again.

        regrowth:
            The number of steps the grass takes to regrow.

    Examples:
        >>> from skcomplex.models.ecology import Grass
        >>> Grass.group('grass', 4, grown=1.0).size
        4
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'color_by': 'grown', 'color_range': (0.0, 1.0), 'radius': 0.5})
    grown: float = state(1.0, domain=Domain.NONNEGATIVE_INTEGER)
    countdown: float = state(0.0, domain=Domain.NONNEGATIVE)
    regrowth: float = parameter(30.0, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the regrowth rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.regrow)]

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


class Sheep(AgentType):
    """Sheep that wanders, eats grown grass, reproduces, and dies without energy.

    Args:
        label:
            The label of the agent type or group.

        location:
            The initial cell, or `None` for a random cell.

        energy:
            The initial energy.

        gain:
            The energy a sheep gains from grown grass.

        reproduction:
            The probability of reproducing at every step.

    Examples:
        >>> from skcomplex.models.ecology import Sheep
        >>> Sheep.group('sheep', 100).size
        100
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'radius': 0.3})
    energy: float = state(8.0)
    gain: float = parameter(4.0, domain=Domain.NONNEGATIVE)
    reproduction: float = parameter(0.04, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the life rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.graze)]

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


class Wolf(AgentType):
    """Wolf that wanders, eats a sheep of its cell, reproduces, and dies without energy.

    Args:
        label:
            The label of the agent type or group.

        location:
            The initial cell, or `None` for a random cell.

        energy:
            The initial energy.

        gain:
            The energy a wolf gains from a sheep.

        reproduction:
            The probability of reproducing at every step.

    Examples:
        >>> from skcomplex.models.ecology import Wolf
        >>> Wolf.group('wolves', 25).size
        25
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'radius': 0.35})
    energy: float = state(20.0)
    gain: float = parameter(20.0, domain=Domain.NONNEGATIVE)
    reproduction: float = parameter(0.05, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the life rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.hunt)]

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


def wolf_sheep(
    width: int = 20,
    height: int = 20,
    *,
    sheep: int = 100,
    wolves: int = 25,
    regrowth: float = 30.0,
    random_state: int | np.random.Generator | None = None,
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
        >>> from skcomplex.models.ecology import wolf_sheep
        >>> [group.label for group in wolf_sheep(10, 10).elements]
        ['grass', 'sheep', 'wolves']
    """
    rng = np.random.default_rng(random_state)
    cells = [(column, row) for column in range(width) for row in range(height)]
    grown = (rng.random(len(cells)) < GROWN_SHARE).astype(float)
    countdown = np.where(grown > 0, regrowth, rng.uniform(0, regrowth, len(cells)))
    grass = Grass.group('grass', len(cells), location=cells, grown=grown, countdown=countdown, regrowth=regrowth)
    return System(
        elements=[grass, Sheep.group('sheep', sheep), Wolf.group('wolves', wolves)],
        space=GridSpace(width, height, torus=True),
    )
