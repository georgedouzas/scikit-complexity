"""Site of a tissue where cells and a chemical move with finite speeds, the hyperbolic limit of velocity jumps."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import divergence, gradient
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class ChemotacticTissueElement(Element):
    """Site of a tissue where cells and a chemical move with finite speeds, the hyperbolic limit of velocity jumps.

    The cell density `n` and its flux `q`, and the chemical density `c` and its flux `j`, follow Eq. (6.50) of Bellomo
    et al. (2017) in two dimensions with the cells as the only source of the chemical, `dn/dt = -div q`, `dq/dt = -(s^2
    / 2) grad n - mu1 q + mu2 alpha n grad c`, `tau dc/dt = -div j + n`, and `tau dj/dt = -(s^2 / 2) grad c - sigma
    j`. Cells and chemical move at the speed `s` and turn at the rates `mu1` and `sigma`, and the cells turn towards
    the gradient of the chemical with the sensitivity `alpha`. In the limit of fast turning it is the Keller-Segel model
    with the diffusions `s^2 / (2 mu1)` and `s^2 / (2 sigma)`.

    Args:
        label:
            The label of the tissue.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        n:
            The initial cell density.

        qx:
            The initial flux of cells along the first axis.

        qy:
            The initial flux of cells along the second axis.

        c:
            The initial chemical density.

        jx:
            The initial flux of the chemical along the first axis.

        jy:
            The initial flux of the chemical along the second axis.

        s:
            The speed of the cells and of the chemical.

        mu1:
            The turning rate of the cells.

        mu2:
            The rate of turning of the cells towards the chemical.

        sigma:
            The turning rate of the chemical.

        tau:
            The time scale of the chemical.

        alpha:
            The chemotactic sensitivity of the cells.

    Examples:
        >>> from skcomplex.systems.biology.physiology import ChemotacticTissueElement
        >>> ChemotacticTissueElement('tissue').alpha
        0.33
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        n: Annotated[float, StateSpec()] = 0.0,
        qx: Annotated[float, StateSpec()] = 0.0,
        qy: Annotated[float, StateSpec()] = 0.0,
        c: Annotated[float, StateSpec()] = 0.0,
        jx: Annotated[float, StateSpec()] = 0.0,
        jy: Annotated[float, StateSpec()] = 0.0,
        s: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 100.0,
        mu1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5000.0,
        mu2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5000.0,
        sigma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5.0e6,
        tau: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        alpha: Annotated[float, ParameterSpec()] = 0.33,
    ) -> None:
        super().__init__(label, visual=visual)
        self.n = n
        self.qx = qx
        self.qy = qy
        self.c = c
        self.jx = jx
        self.jy = jy
        self.s = s
        self.mu1 = mu1
        self.mu2 = mu2
        self.sigma = sigma
        self.tau = tau
        self.alpha = alpha

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the balance laws of the cells and the chemical with their turning relaxations.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        pressure = s.s**2 / 2
        cells, chemical = gradient(s.n), gradient(s.c)
        return [
            RateLaw(s.n, -divergence([s.qx, s.qy])),
            RateLaw(s.qx, -pressure * cells[0] - s.mu1 * s.qx + s.mu2 * s.alpha * s.n * chemical[0]),
            RateLaw(s.qy, -pressure * cells[1] - s.mu1 * s.qy + s.mu2 * s.alpha * s.n * chemical[1]),
            RateLaw(s.c, (-divergence([s.jx, s.jy]) + s.n) / s.tau),
            RateLaw(s.jx, (-pressure * chemical[0] - s.sigma * s.jx) / s.tau),
            RateLaw(s.jy, (-pressure * chemical[1] - s.sigma * s.jy) / s.tau),
        ]
