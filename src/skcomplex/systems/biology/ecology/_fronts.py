"""Invasion fronts of a population that grows logistically and diffuses."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, FieldElement
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import laplacian
from ....core.components.spaces import LatticeSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class FisherKPPElement(Element):
    """Site of a line whose density `u` diffuses and grows logistically.

    The density follows `du/dt = D laplacian(u) + r u (1 - u)`.

    Args:
        label:
            The label of the site.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        u:
            The initial density, as a fraction of the carrying capacity.

        D:
            The diffusion coefficient.

        r:
            The growth rate.

    Examples:
        >>> from skcomplex.systems.biology.ecology import FisherKPPElement
        >>> FisherKPPElement('site', u=0.5).u
        0.5
    """

    law_classes = (RateLaw,)

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
        """Return the diffusion and the logistic growth of the density.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate law of the density.
        """
        return [RateLaw(s.u, s.D * laplacian(s.u) + s.r * s.u * (1 - s.u))]


class FisherKPPSystem(BaseSystem):
    """The Fisher-KPP equation, whose step invades the empty line as a front.

    The front has the speed `2 sqrt(r D)`.

    The system has the density `line.u`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.ecology import FisherKPPSystem
        >>> int(FisherKPPSystem(100, front=0.1).prepare().system_.elements[0].values['u'].sum())
        10
    """

    field = 'biology/ecology'
    topics = ('field', 'fronts', 'reaction-diffusion', 'pde')
    references = ('Fisher 1937', 'Kolmogorov, Petrovsky, and Piskunov 1937')
    until = 40.0
    dt = 1.0

    def __init__(
        self,
        length: int = 800,
        *,
        D: float = 1.0,
        r: float = 1.0,
        spacing: float = 0.25,
        front: float = 0.05,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.length = length
        self.D = D
        self.r = r
        self.spacing = spacing
        self.front = front

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the Fisher-KPP equation."""
        u = (np.arange(self.length) < self.front * self.length).astype(float)
        space = LatticeSpace(self.length, spacing=self.spacing)
        return CompositeSystem(
            self.label,
            elements=[
                FieldElement(
                    'line',
                    element=FisherKPPElement,
                    space=space,
                    boundary={'u': 'zero-flux'},
                    values={'u': u, 'D': self.D, 'r': self.r},
                ),
            ],
        )
