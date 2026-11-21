"""Innovation systems: diversity that grows by combining existing entities.

The combinations make new entities.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import UpdateLaw
from ....core.components.laws.utils import where
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Role


class AdjacentPossibleElement(Element):
    """Combinatorial system of entities.

    It holds the share that exists and the share its new pairs can produce.

    Args:
        label:
            The label of the system.

        x:
            The share of existing entities, the diversity.

        d:
            The share of new entities, the adjacent possible.

        r:
            The rule density, the number of productive pairs per entity.

    Examples:
        >>> from skcomplex.systems.economics.innovation import AdjacentPossibleElement
        >>> AdjacentPossibleElement('system', r=2.0).r
        2.0
    """

    def __init__(
        self,
        label: str,
        *,
        x: Annotated[float, RealSet.closed_open(0, oo)] = 0.1,
        d: Annotated[float, RealSet.closed_open(0, oo)] = 0.0,
        r: float = 5.0,
    ) -> None:
        super().__init__(label)
        self.x = x
        self.d = d
        self.r = r


class CombinationInteraction(Interaction):
    """Growth of the diversity of a combinatorial system by the new entities its pairs produce.

    The share of existing entities grows by the adjacent possible, `x' = x + d`, and the new pairs of existing entities
    produce the next adjacent possible, `d' = r (1 - x') (x'^2 - x^2)`, at most the share `1 - x'` that does not exist
    yet. The rule density `r` is a value of the system, the number of productive pairs per entity.

    Args:
        label:
            The label of the interaction.

        target:
            The combinatorial system, with the diversity `x`, the adjacent possible `d`, and the rule density `r`.

    Examples:
        >>> from skcomplex.systems.economics.innovation import CombinationInteraction
        >>> CombinationInteraction('combination').target
        'system'
    """

    law_classes = (UpdateLaw,)

    def __init__(self, label: str, *, target: Role = 'system') -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the growth of the diversity and the next adjacent possible as the laws.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        x, d, r = context.target.x, context.target.d, context.target.r
        current = x + d
        produced = r * (1 - current) * (current**2 - x**2)
        self.laws_: list[BaseComponent] = [
            UpdateLaw(x, x + d, increment=True),
            UpdateLaw(d, where(produced < 1 - current, produced, 1 - current)),
        ]


class CreativePhaseTransitionSystem(BaseSystem):
    """The diversity of a combinatorial system.

    It jumps from sparse to full above a critical rule density. The share of existing entities `x` grows by the share
    `d` of new entities that pairs of existing entities can produce, `x_{t+1} = x_t + d_t` and
    `d_{t+1} = r (1 - x_{t+1}) (x_{t+1}^2 - x_t^2)`, with the rule density `r`. The first step produces
    `d_0 = r x_0^2 (1 - x_0)`, and no step produces more than the entities that do not exist yet. The final diversity
    changes discontinuously with `r` and `x_0`. The system has the diversity `system.x` and the adjacent possible
    `system.d`.

    Args:
        x:
            The initial share of existing entities.

        r:
            The rule density, the number of productive pairs per entity.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.economics.innovation import CreativePhaseTransitionSystem
        >>> CreativePhaseTransitionSystem(r=2.0).prepare().system_.elements[0].r
        2.0
    """

    field = 'economics/innovation'
    topics = ('innovation', 'combinatorial evolution', 'phase transition', 'discrete map')
    references = ('Thurner et al. 2018', 'Hanel et al. 2005')
    until = 50
    dt = 1

    def __init__(
        self,
        x: float = 0.1,
        *,
        r: float = 5.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x
        self.r = r

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the combinatorial system and its combinations."""
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[AdjacentPossibleElement('system', x=self.x, d=self.r * self.x**2 * (1 - self.x), r=self.r)],
            interactions=[CombinationInteraction('combination', target='system')],
        )
