"""Continuously fed reactor of cubic autocatalysis, the concentrations of the reactant `a` and the autocatalyst `b`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class CubicAutocatalysisElement(Element):
    """Continuously fed reactor of cubic autocatalysis, the concentrations of the reactant `a` and the autocatalyst `b`.

    The reactions and the flows of `cubic_autocatalysis` change the concentrations and read the feed rate `F` and the
    decay rate `k` from the reactor.

    Args:
        label:
            The label of the reactor.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        a:
            The initial concentration of the reactant.

        b:
            The initial concentration of the autocatalyst.

        F:
            The feed rate, the inverse residence time.

        k:
            The decay rate of the autocatalyst.

    Examples:
        >>> from skcomplex.systems.chemistry.kinetics import CubicAutocatalysisElement
        >>> CubicAutocatalysisElement('reactor').F
        0.04
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        a: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        b: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        F: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.04,
        k: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
    ) -> None:
        super().__init__(label, visual=visual)
        self.a = a
        self.b = b
        self.F = F
        self.k = k
