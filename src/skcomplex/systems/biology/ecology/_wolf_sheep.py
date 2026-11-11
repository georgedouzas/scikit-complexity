"""Wolves and sheep that hunt, graze, and breed on a grid of regrowing grass."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import TransferInteraction
from ....core.components.laws import BaseLaw, RuleLaw, Sink, Source
from ....core.components.spaces import LatticeSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ....core.engine.solvers import AgentView, SystemView
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ...general.agents import BirthDeathInteraction, MovementInteraction

GROWN_SHARE = 0.5


class GrassView(AgentView):
    """View of a patch of grass, with its states and parameters as attributes.

    `GrassElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    grown: float
    countdown: float
    regrowth: float


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
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.biology.ecology import GrassElement
        >>> GroupElement('grass', element=GrassElement, size=4, values={'grown': 1.0}).size
        4
    """

    law_classes = (RuleLaw,)

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

    def regrow(self, agent: GrassView, system: SystemView) -> None:
        """Count down eaten grass and grow it back when the countdown ends.

        Args:
            agent:
                The patch.

            system:
                The system.
        """
        del system
        if agent.grown:
            return
        agent.countdown -= 1
        if agent.countdown <= 0:
            agent.grown = 1.0
            agent.countdown = agent.regrowth


class SheepView(AgentView):
    """View of a sheep, with its states and parameters as attributes.

    `SheepElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    energy: float
    gain: float
    reproduction: float


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
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.biology.ecology import SheepElement
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


class WolfView(AgentView):
    """View of a wolf, with its states and parameters as attributes.

    `WolfElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    energy: float
    gain: float
    reproduction: float


class WolfElement(Element):
    """Wolf that wanders, eats a sheep, reproduces, and dies without energy.

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
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.biology.ecology import WolfElement
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


class WolfSheepSystem(BaseSystem):
    """The wolf and sheep predation system with one grass patch per cell.

    The system has the groups `grass`, `sheep`, and `wolves`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.ecology import WolfSheepSystem
        >>> [group.label for group in WolfSheepSystem(10, 10).prepare().system_.elements]
        ['grass', 'sheep', 'wolves']
    """

    field = 'biology/ecology'
    topics = ('predator-prey', 'agents', 'grid')
    references = ('Wilensky 1997',)
    until = 100

    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        *,
        sheep: int = 100,
        wolves: int = 25,
        regrowth: float = 30.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.width = width
        self.height = height
        self.sheep = sheep
        self.wolves = wolves
        self.regrowth = regrowth
        self.random_state = random_state

    @staticmethod
    def _get_gain(food: AgentViewProtocol, animal: AgentViewProtocol, system: SystemViewProtocol) -> float:
        """Return the energy `gain` an animal draws from its food."""
        return float(animal.gain) if isinstance(animal, SheepView | WolfView) else 0.0

    @staticmethod
    def _is_grown(patch: AgentViewProtocol, sheep: AgentViewProtocol, system: SystemViewProtocol) -> bool:
        """Return whether the grass of a patch is grown.

        A sheep eats grown grass.
        """
        return isinstance(patch, GrassView) and bool(patch.grown)

    @staticmethod
    def _is_starved(animal: AgentViewProtocol, system: SystemViewProtocol) -> bool:
        """Return whether an animal has run out of energy.

        It then dies.
        """
        return isinstance(animal, SheepView | WolfView) and animal.energy < 0

    @staticmethod
    def _get_reproduction(animal: AgentViewProtocol, system: SystemViewProtocol) -> float:
        """Return the probability `reproduction` that an animal gives birth in a step."""
        return float(animal.reproduction) if isinstance(animal, SheepView | WolfView) else 0.0

    def _get_animal(
        self,
        group: str,
        food: str,
        removed: bool,
    ) -> list[MovementInteraction | TransferInteraction | BirthDeathInteraction]:
        """Return the member interactions of an animal.

        An animal wanders at a unit cost of energy, eats in its cell, and lives.
        """
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
            amount=self._get_gain,
            legs=() if removed else (('grown', Sink(), 1.0),),
            guard=None if removed else self._is_grown,
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
            BirthDeathInteraction(
                f'{group}_life',
                target=group,
                death=self._is_starved,
                birth=self._get_reproduction,
                split=('energy',),
                schedule='member',
            ),
        ]

    def _build_system(self) -> CompositeSystem:
        """Return the components of the wolf and sheep predation system.

        It has one grass patch per cell.
        """
        rng = np.random.default_rng(self.random_state)
        cells = [(column, row) for column in range(self.width) for row in range(self.height)]
        grown = (rng.random(len(cells)) < GROWN_SHARE).astype(float)
        countdown = np.where(grown > 0, self.regrowth, rng.uniform(0, self.regrowth, len(cells)))
        grass = GroupElement(
            'grass',
            element=GrassElement,
            size=len(cells),
            values={'grown': grown, 'countdown': countdown, 'regrowth': self.regrowth},
            variable=True,
            location=cells,
        )
        return CompositeSystem(
            self.label,
            elements=[
                grass,
                GroupElement('sheep', element=SheepElement, size=self.sheep, variable=True),
                GroupElement('wolves', element=WolfElement, size=self.wolves, variable=True),
            ],
            interactions=[
                *self._get_animal('sheep', 'grass', removed=False),
                *self._get_animal('wolves', 'sheep', removed=True),
            ],
            space=LatticeSpace((self.width, self.height), boundary='periodic'),
        )
