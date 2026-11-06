"""Small zone of a two-phase apparatus that exchanges particles with the rest of it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, JumpLaw
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class ZoneElement(Element):
    """Small zone of a two-phase apparatus that exchanges particles with the rest of it.

    Every one of the `Na` particles enters the zone at the rate `k p` and leaves it at the rate `k (1 - p)`, where
    `p = dV / V` is the volume fraction of the zone, so the number in the zone is binomial with mean `Na p`.

    Args:
        label:
            The label of the zone or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        N:
            The initial number of particles in the zone.

        Na:
            The number of particles in the apparatus and the zone.

        p:
            The volume fraction of the zone.

        k:
            The rate of exchange.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import ZoneElement
        >>> ZoneElement('zone').Na
        100.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        N: Annotated[float, StateSpec(domain=NN)] = 0.0,
        Na: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 100.0,
        p: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        k: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.N = N
        self.Na = Na
        self.p = p
        self.k = k

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the entries into and exits from the zone.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [
            JumpLaw(s.k * s.p * (s.Na - s.N), {s.N: 1}),
            JumpLaw(s.k * (1 - s.p) * s.N, {s.N: -1}),
        ]
