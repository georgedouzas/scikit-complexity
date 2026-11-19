"""Rumour systems: news that spreads by contact until spreaders lose interest."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import JumpLaw
from ....core.components.spaces import BaseSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Role


class ListenersElement(Element):
    """Number of people in one state of a rumour: ignorant, spreading, or stifling.

    Args:
        label:
            The label of the element.

        n:
            The initial number of people.

    Examples:
        >>> from skcomplex.systems.social.spreading import ListenersElement
        >>> ListenersElement('spreaders', n=1).n
        1
    """

    def __init__(
        self,
        label: str,
        *,
        n: Annotated[float, NN] = 0,
    ) -> None:
        super().__init__(label)
        self.n = n


class RumourContactsInteraction(Interaction):
    """Contacts of the Maki-Thompson rumour system in a well-mixed population.

    The population has size `N`. A spreader who meets an ignorant tells the rumour at rate `spread * X * Y / N`, and a
    spreader who meets another spreader or a stifler stops spreading at rate `stifle * Y * (Y - 1 + Z) / N`.

    Args:
        label:
            The label of the interaction.

        ignorants:
            The path of the ignorants `X`.

        spreaders:
            The path of the spreaders `Y`.

        stiflers:
            The path of the stiflers `Z`.

        spread:
            The contact rate that spreads the rumour.

        stifle:
            The contact rate that stifles a spreader.

    Examples:
        >>> from skcomplex.systems.social.spreading import RumourContactsInteraction
        >>> RumourContactsInteraction('contacts', ignorants='X', spreaders='Y', stiflers='Z').spread
        1.0
    """

    law_classes = (JumpLaw,)

    def __init__(
        self,
        label: str,
        *,
        ignorants: Role,
        spreaders: Role,
        stiflers: Role,
        spread: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        stifle: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
    ) -> None:
        super().__init__(label)
        self.ignorants = ignorants
        self.spreaders = spreaders
        self.stiflers = stiflers
        self.spread = spread
        self.stifle = stifle

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the telling and stifling jumps as the laws."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            ignorants, spreaders, stiflers = context.ignorants.n, context.spreaders.n, context.stiflers.n
            total = ignorants + spreaders + stiflers
            self.laws_: list[BaseComponent] = [
                JumpLaw(context.spread * ignorants * spreaders / total, {ignorants: -1, spreaders: 1}),
                JumpLaw(
                    context.stifle * spreaders * (spreaders - 1 + stiflers) / total,
                    {spreaders: -1, stiflers: 1},
                ),
            ]


class MakiThompsonSystem(BaseSystem):
    """The Maki-Thompson rumour system started by one spreader.

    It leaves about 20% of people ignorant. The system has the counts `ignorants.n`, `spreaders.n`, and `stiflers.n`.

    Args:
        n_people:
            The size of the population.

        spread:
            The contact rate that spreads the rumour.

        stifle:
            The contact rate that stifles a spreader.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.social.spreading import MakiThompsonSystem
        >>> [element.n for element in MakiThompsonSystem(100).prepare().system_.elements]
        [99, 1, 0]
    """

    field = 'social/spreading'
    topics = ('rumours', 'jump process', 'stochastic')
    references = ('Maki and Thompson 1973', 'Pastor-Satorras et al. 2015')
    until = 20.0
    dt = 0.5

    def __init__(
        self,
        n_people: int = 1000,
        *,
        spread: float = 1.0,
        stifle: float = 1.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_people = n_people
        self.spread = spread
        self.stifle = stifle

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components built from the model parameters."""
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[
                ListenersElement('ignorants', n=self.n_people - 1),
                ListenersElement('spreaders', n=1),
                ListenersElement('stiflers', n=0),
            ],
            interactions=[
                RumourContactsInteraction(
                    'contacts',
                    ignorants='ignorants',
                    spreaders='spreaders',
                    stiflers='stiflers',
                    spread=self.spread,
                    stifle=self.stifle,
                ),
            ],
        )
