"""Ecology models: population growth, predation, and invasion fronts."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, FieldElement
from ....core.laws import BaseLaw, RateLaw
from ....core.laws.utils import laplacian
from ....core.spaces import LatticeSpace
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ....core.system import System
from ..._catalog import register_model


class FisherKPPElement(Element):
    """Fisher k p p element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        u: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        D: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        r: Annotated[object, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.u = u
        self.D = D
        self.r = r

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the fisher k p p element."""
        return [RateLaw(s.u, s.D * laplacian(s.u) + s.r * s.u * (1 - s.u))]


@register_model(
    field='biology/ecology',
    tags=('field', 'fronts', 'reaction-diffusion', 'pde'),
    references=('Fisher 1937', 'Kolmogorov, Petrovsky, and Piskunov 1937'),
    defaults={'until': 40.0, 'dt': 1.0},
)
def fisher_kpp(
    length: int = 800,
    *,
    D: float = 1.0,
    r: float = 1.0,
    spacing: float = 0.25,
    front: float = 0.05,
) -> System:
    """Return the Fisher-KPP equation, whose step invades the empty line as a front of speed `2 sqrt(r D)`.

    Args:
        length:
            The number of sites of the line.

        D:
            The diffusion coefficient.

        r:
            The growth rate.

        spacing:
            The distance between sites.

        front:
            The fraction of the line that starts occupied, on its left.

    Returns:
        system:
            The model, with the density `line.u`.

    Examples:
        >>> from skcomplex.models.biology.ecology import fisher_kpp
        >>> int(fisher_kpp(100, front=0.1).elements[0].values['u'].sum())
        10
    """
    u = (np.arange(length) < front * length).astype(float)
    space = LatticeSpace(length, spacing=spacing)
    return System(
        elements=[
            FieldElement(
                'line',
                element=FisherKPPElement,
                space=space,
                boundary={'u': 'zero-flux'},
                values={'u': u, 'D': D, 'r': r},
            ),
        ],
    )
