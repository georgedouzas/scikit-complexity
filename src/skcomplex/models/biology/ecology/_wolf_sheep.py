"""Ecology models: population growth, predation, and invasion fronts."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement
from ....core.interactions import LifeCycleInteraction, MovementInteraction, TransferInteraction
from ....core.laws import BaseLaw, RuleLaw, Sink, Source
from ....core.spaces import LatticeSpace
from ....core.system import System
from ....core.types import AgentViewProtocol, ModelViewProtocol
from ....engine.solvers import ModelView
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_model
from ._grass_view import GrassView
from ._sheep_view import SheepView
from ._wolf_view import WolfView

GROWN_SHARE = 0.5


def _get_gain(food: AgentViewProtocol, animal: AgentViewProtocol, model: ModelViewProtocol) -> float:
    """Return the energy `gain` an animal draws from its food."""
    return float(animal.gain) if isinstance(animal, SheepView | WolfView) else 0.0


def _is_grown(patch: AgentViewProtocol, sheep: AgentViewProtocol, model: ModelViewProtocol) -> bool:
    """Return whether the grass of a patch is grown, so a sheep eats it."""
    return isinstance(patch, GrassView) and bool(patch.grown)


def _is_starved(animal: AgentViewProtocol, model: ModelViewProtocol) -> bool:
    """Return whether an animal has run out of energy, so it dies."""
    return isinstance(animal, SheepView | WolfView) and animal.energy < 0


def _get_reproduction(animal: AgentViewProtocol, model: ModelViewProtocol) -> float:
    """Return the probability `reproduction` that an animal gives birth in a step."""
    return float(animal.reproduction) if isinstance(animal, SheepView | WolfView) else 0.0


def _get_animal(
    group: str,
    food: str,
    removed: bool,
) -> list[MovementInteraction | TransferInteraction | LifeCycleInteraction]:
    """Return the member interactions of an animal: wander at a unit cost of energy, eat in its cell, and live."""
    cost = TransferInteraction(
        f'{group}_cost',
        source=group,
        target=group,
        state='energy',
        into=Sink(),
        amount=1.0,
        form='agent',
    )
    eating = TransferInteraction(
        f'{group}_eating',
        source=food,
        target=group,
        state=Source(),
        into='energy',
        amount=_get_gain,
        legs=() if removed else (('grown', Sink(), 1.0),),
        guard=None if removed else _is_grown,
        pairing='cell',
        form='agent',
        schedule='member',
        actor='target',
        partners=1,
        remove='source' if removed else None,
    )
    return [
        MovementInteraction(f'{group}_walk', target=group, cost=cost, schedule='member'),
        eating,
        LifeCycleInteraction(
            f'{group}_life',
            target=group,
            death=_is_starved,
            birth=_get_reproduction,
            split=('energy',),
            schedule='member',
        ),
    ]


class GrassElement(Element):
    """Grass patch that regrows a fixed number of steps after it is eaten.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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

    @classmethod
    def _get_view(cls) -> type[GrassView]:
        """Return the view of a patch of grass, `GrassView`."""
        return GrassView

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

    def regrow(self, agent: GrassView, model: ModelView) -> None:
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

    In its own activation, a sheep wanders to a neighbouring cell at a cost of one unit of energy, eats the grown grass
    of its cell, and dies below zero energy or gives birth with probability `reproduction`, splitting its energy with
    the lamb, by member interactions.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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

    @classmethod
    def _get_view(cls) -> type[SheepView]:
        """Return the view of a sheep, `SheepView`."""
        return SheepView


class WolfElement(Element):
    """Wolf that wanders, eats a sheep of its cell, reproduces, and dies without energy.

    In its own activation, a wolf wanders to a neighbouring cell at a cost of one unit of energy, eats and removes one
    random sheep of its cell, and dies below zero energy or gives birth with probability `reproduction`, by member
    interactions.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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

    @classmethod
    def _get_view(cls) -> type[WolfView]:
        """Return the view of a wolf, `WolfView`."""
        return WolfView


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
        interactions=[*_get_animal('sheep', 'grass', removed=False), *_get_animal('wolves', 'sheep', removed=True)],
        space=LatticeSpace((width, height), boundary='periodic'),
    )
