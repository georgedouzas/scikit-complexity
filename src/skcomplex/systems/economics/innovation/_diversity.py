"""Innovation systems: diversity that grows by combining existing entities into new ones."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import MapInteraction
from ....core.components.laws.utils import where
from ....core.components.system import BaseSystem, System
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class AdjacentPossibleElement(Element):
    """Combinatorial system of entities, with the share that exists and the share its new pairs can produce.

    Args:
        label:
            The label of the system.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        d: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        r: Annotated[object, ParameterSpec()] = 5.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.d = d
        self.r = r


class CreativePhaseTransitionSystem(BaseSystem):
    """The diversity of a combinatorial system, which jumps from sparse to full above a critical rule density.

    The share of existing entities `x` grows by the share `d` of new entities that pairs of existing entities can
    produce, `x_{t+1} = x_t + d_t` and `d_{t+1} = r (1 - x_{t+1}) (x_{t+1}^2 - x_t^2)`, with the rule density `r`. The
    first step produces `d_0 = r x_0^2 (1 - x_0)`, and no step produces more than the entities that do not exist yet.
    The final diversity changes discontinuously with `r` and `x_0`. The system has the diversity `system.x` and the
    adjacent possible `system.d`.

    Args:
        x:
            The initial share of existing entities.

        r:
            The rule density, the number of productive pairs per entity.

        label:
            The label of the system, which names it when it is nested in another one.

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
    ) -> None:
        super().__init__(label)
        self.x = x
        self.r = r

    @staticmethod
    def _new_entities(s: SimpleNamespace) -> Expression:
        """Return the share of new entities that the new pairs produce, at most the share that does not exist yet."""
        previous, current = s.target.x, s.target.x + s.target.d
        produced = s.target.r * (1 - current) * (current**2 - previous**2)
        return where(produced < 1 - current, produced, 1 - current)

    def _build_system(self) -> System:
        """Return the parts of the system built from its model parameters."""
        return System(
            self.label,
            elements=[AdjacentPossibleElement('system', x=self.x, d=self.r * self.x**2 * (1 - self.x), r=self.r)],
            interactions=[
                MapInteraction(
                    'combination',
                    target='system',
                    state='x',
                    function=lambda s: s.target.d,
                    mode='increment',
                ),
                MapInteraction('adjacent_possible', target='system', state='d', function=self._new_entities),
            ],
        )
