"""Innovation models: diversity that grows by combining existing entities into new ones."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, UpdateLaw
from ....core.laws.utils import where
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ....core.system import System
from ..._catalog import register_model


def _new_entities(previous: Expression, current: Expression, r: Expression) -> Expression:
    """Return the share of new entities that the new pairs produce, at most the share that does not exist yet."""
    produced = r * (1 - current) * (current**2 - previous**2)
    return where(produced < 1 - current, produced, 1 - current)


class AdjacentPossibleElement(Element):
    """Adjacent possible element."""

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

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the adjacent possible element."""
        return [UpdateLaw(s.x, s.x + s.d), UpdateLaw(s.d, _new_entities(s.x, s.x + s.d, s.r))]


@register_model(
    field='economics/innovation',
    tags=('innovation', 'combinatorial evolution', 'phase transition', 'discrete map'),
    references=('Thurner et al. 2018', 'Hanel et al. 2005'),
    defaults={'until': 50, 'dt': 1},
)
def creative_phase_transition(x: float = 0.1, *, r: float = 5.0) -> System:
    """Return the diversity of a combinatorial system, which jumps from sparse to full above a critical rule density.

    The share of existing entities `x` grows by the share `d` of new entities that pairs of existing entities can
    produce, `x_{t+1} = x_t + d_t` and `d_{t+1} = r (1 - x_{t+1}) (x_{t+1}^2 - x_t^2)`, with the rule density `r`.
    The first step produces `d_0 = r x_0^2 (1 - x_0)`, and no step produces more than the entities that do not exist
    yet. The final diversity changes discontinuously with `r` and `x_0`.

    Args:
        x:
            The initial share of existing entities.

        r:
            The rule density, the number of productive pairs per entity.

    Returns:
        system:
            The model, with the diversity `system.x` and the adjacent possible `system.d`.

    Examples:
        >>> from skcomplex.models.economics.innovation import creative_phase_transition
        >>> creative_phase_transition(r=2.0).elements[0].r
        2.0
    """
    return System(elements=[AdjacentPossibleElement('system', x=x, d=r * x**2 * (1 - x), r=r)])
