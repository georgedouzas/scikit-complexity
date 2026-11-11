"""Ecology systems: cows that graze a meadow of regrowing grass."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from types import SimpleNamespace
from typing import Annotated, cast

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import TransferInteraction
from ....core.components.laws import BaseLaw, RuleLaw, Sink
from ....core.components.spaces import LatticeSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ....core.engine.solvers import AgentView, SystemView
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ...general.agents import MovementInteraction


class CowView(AgentView):
    """View of a cow, with its parameters as attributes.

    `CowElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    bite: float


class CowElement(Element):
    """Cow of the meadow.

    The grazing and herding interactions feed and move it every week.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        bite:
            The grass a cow eats in a week.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.biology.ecology import CowElement
        >>> GroupElement('cows', element=CowElement, size=10).size
        10
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'radius': 0.4}),
        *,
        bite: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.75,
    ) -> None:
        super().__init__(label, visual=visual)
        self.bite = bite

    @classmethod
    def _get_view(cls) -> type[CowView]:
        """Return the view of a cow, `CowView`."""
        return CowView


class PastureView(AgentView):
    """View of a patch of pasture, with its states and parameters as attributes.

    `PastureElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    grass: float
    growth: float


class PastureElement(Element):
    """Patch of grass that regrows logistically every week, `G' = G + g G (1 - G)`.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        grass:
            The amount of grass, between 0 and 1.

        growth:
            The weekly growth rate `g` of the grass.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.biology.ecology import PastureElement
        >>> GroupElement('pasture', element=PastureElement, size=4, values={'growth': 0.1}).size
        4
    """

    law_classes = (RuleLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'grass', 'color_range': (0.0, 1.0)}),
        *,
        grass: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        growth: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
    ) -> None:
        super().__init__(label, visual=visual)
        self.grass = grass
        self.growth = growth

    @classmethod
    def _get_view(cls) -> type[PastureView]:
        """Return the view of a patch of pasture, `PastureView`."""
        return PastureView

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

    def regrow(self, agent: PastureView, system: SystemView) -> None:
        """Grow the grass by one week of logistic growth.

        Args:
            agent:
                The patch.

            system:
                The system.
        """
        del system
        agent.grass += agent.growth * agent.grass * (1 - agent.grass)


class GrazingInteraction(TransferInteraction):
    """Weekly meal of every cow, a bite of the grass of the patch in its cell.

    It is a transfer of the grass of every patch, the bite of the cow in its cell or what grass is left, to the
    environment.

    Args:
        label:
            The label of the interaction.

        cows:
            The path of the group of cows.

        pasture:
            The path of the group of patches.

    Examples:
        >>> from skcomplex.systems.biology.ecology import GrazingInteraction
        >>> GrazingInteraction('grazing', cows='cows', pasture='pasture').pasture
        'pasture'
    """

    def __init__(
        self,
        label: str,
        *,
        cows: Annotated[str, RoleSpec()],
        pasture: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(
            label,
            source=pasture,
            target=cows,
            state='grass',
            into=Sink(),
            amount=self._get_bite,
            pairing='cell',
            form='agent',
        )
        self.cows = cows
        self.pasture = pasture

    def _get_role(self, role: str) -> str:
        """Return the patches as the source of the grass and the cows as its target."""
        return {'source': 'pasture', 'target': 'cows'}.get(role, role)

    @staticmethod
    def _get_bite(patch: AgentViewProtocol, cow: AgentViewProtocol, system: SystemViewProtocol) -> float:
        """Return the grass a cow eats from the patch of its cell.

        It is its bite or what grass is left.
        """
        del system
        return min(cast(PastureView, patch).grass, cast(CowView, cow).bite)


class HerdingInteraction(MovementInteraction):
    """Weekly move of every cow to the nearest free patch with enough grass, or death.

    The cows move in random order.

    It is a movement of the cows whose step leads every cow to the patch nearest to it that no other cow occupies and
    whose grass reaches `threshold`, and removes a cow that finds none. Every cow moves every week, the probability
    of movement fixed to 1 without a parameter.

    Args:
        label:
            The label of the interaction.

        cows:
            The path of the group of cows.

        pasture:
            The path of the group of patches.

        threshold:
            The least grass a patch needs for a cow to move there.

    Examples:
        >>> from skcomplex.systems.biology.ecology import HerdingInteraction
        >>> HerdingInteraction('herding', cows='cows', pasture='pasture').threshold
        0.9
    """

    def __init__(
        self,
        label: str,
        *,
        cows: Annotated[str, RoleSpec()],
        pasture: Annotated[str, RoleSpec()],
        threshold: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.9,
    ) -> None:
        super().__init__(label, target=cows, step=self._herd)
        self.cows = cows
        self.pasture = pasture
        self.threshold = threshold

    def _get_role(self, role: str) -> str:
        """Return the cows as the target of the movement."""
        return 'cows' if role == 'target' else role

    @staticmethod
    def _get_cell(agent: AgentViewProtocol) -> tuple[int, ...]:
        """Return the cell of an agent as a tuple of integers."""
        location = agent.location
        if not isinstance(location, tuple):
            error_msg = (
                f'The agent is located at a {type(location).__name__}, not at the coordinates of a lattice cell.'
            )
            raise TypeError(error_msg)
        return tuple(int(item) for item in location)

    def _herd(self, agent: AgentViewProtocol, system: SystemViewProtocol) -> tuple[int, ...] | None:
        """Return the cell of the nearest free patch with enough grass.

        It removes the cow that finds none.
        """
        patches = [cast(PastureView, patch) for patch in system.get_agents('pasture')]
        cells = [self._get_cell(patch) for patch in patches]
        index = {cell: number for number, cell in enumerate(cells)}
        occupied = np.zeros(len(cells), dtype=bool)
        occupied[
            [index[self._get_cell(cow)] for cow in system.get_agents('cows') if cow.alive and cow.id != agent.id]
        ] = True
        free = (np.asarray([patch.grass for patch in patches]) >= system.get_parameter('threshold')) & ~occupied
        if not free.any():
            agent.remove()
            return None
        distances = np.hypot(*(np.asarray(cells, dtype=float) - np.asarray(self._get_cell(agent), dtype=float)).T)
        return cells[int(np.argmin(np.where(free, distances, np.inf)))]


class MeadowCapacitySystem(BaseSystem):
    """Cows that graze a meadow for a summer of 25 weeks.

    It shows how many cows the meadow can carry.

    Every week the grass regrows logistically, each cow eats `bite` of the grass of its patch, and each cow moves to the
    nearest patch that no other cow occupies and whose grass reaches `threshold`. A cow that finds none dies. Without
    regrowth a cow grazes its first patch and 25 more over the summer, so the meadow feeds about `width * height / 26`
    cows, and regrowth raises its carrying capacity. The cows graze the patches down at the same pace, so a herd above
    the capacity runs out of grown patches all at once and dies out within a week, instead of shrinking to the capacity.
    The system has the groups `pasture` and `cows`.

    Args:
        width:
            The number of columns of the meadow.

        height:
            The number of rows of the meadow.

        cows:
            The initial number of cows.

        growth:
            The weekly growth rate of the grass.

        bite:
            The grass a cow eats in a week.

        threshold:
            The least grass a patch needs for a cow to move there.

        random_state:
            The seed or generator of the initial cells of the cows.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.ecology import MeadowCapacitySystem
        >>> [group.size for group in MeadowCapacitySystem(10, 10, cows=5).prepare().system_.elements]
        [100, 5]
    """

    field = 'biology/ecology'
    topics = ('carrying capacity', 'agents', 'grid', 'grazing', 'commons')
    references = ('Hamill and Gilbert 2016',)
    until = 25

    def __init__(
        self,
        width: int = 40,
        height: int = 40,
        *,
        cows: int = 80,
        growth: float = 0.2,
        bite: float = 0.75,
        threshold: float = 0.9,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.width = width
        self.height = height
        self.cows = cows
        self.growth = growth
        self.bite = bite
        self.threshold = threshold
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return the components of cows that graze a summer of 25 weeks on a meadow."""
        rng = np.random.default_rng(self.random_state)
        cells = [(column, row) for column in range(self.width) for row in range(self.height)]
        picks = rng.choice(len(cells), size=self.cows, replace=False)
        return CompositeSystem(
            self.label,
            elements=[
                GroupElement(
                    'pasture',
                    element=PastureElement,
                    size=len(cells),
                    values={'growth': self.growth},
                    variable=True,
                    location=cells,
                ),
                GroupElement(
                    'cows',
                    element=CowElement,
                    size=self.cows,
                    values={'bite': self.bite},
                    variable=True,
                    location=[cells[int(pick)] for pick in picks],
                ),
            ],
            interactions=[
                GrazingInteraction('grazing', cows='cows', pasture='pasture'),
                HerdingInteraction('herding', cows='cows', pasture='pasture', threshold=self.threshold),
            ],
            space=LatticeSpace((self.width, self.height), boundary='fixed'),
        )
