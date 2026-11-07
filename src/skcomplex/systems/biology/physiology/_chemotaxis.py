"""Physiology systems: chemotaxis of cells that move with finite speed."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, FieldElement
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import divergence, gradient
from ....core.components.spaces import LatticeSpace
from ....core.components.system import BaseSystem, System
from ....core.engine.solvers import ScipyODESolver
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

    law_classes = (RateLaw,)

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


class HyperbolicChemotaxisSystem(BaseSystem):
    """Cells and a chemical moving with finite speed, whose two bumps of cells spread and attract each other.

    The model is the hyperbolic limit of a kinetic model of cells and a chemical that change velocity at random, with
    the speed `s` and turning rates set by the diffusions `D_n = s^2 / (2 mu1)` and `D_c = s^2 / (2 sigma)` of its
    parabolic limit, Keller-Segel, and `mu2 = mu1`. Following the test of Bellomo et al. (2017, section 6.6.2), the
    cells start as two Gaussian bumps of mass `n0` and width `width` at `(3 width, 3 width)` and `(-3 width, -3 width)`
    in the square `[-0.4, 0.4]^2`, here with periodic edges, the cells secrete the chemical, and all fluxes start at
    rest. The fluxes relax fast, so the bumps spread out at a finite speed while the cell mass `2 n0` is conserved and
    the chemical accumulates where the cells are. The system has the cell density `tissue.n`, the chemical `tissue.c`,
    and their fluxes.

    Args:
        sites:
            The number of sites along each side of the square.

        s:
            The speed of the cells and of the chemical.

        D_n:
            The diffusion of the cells in the parabolic limit.

        D_c:
            The diffusion of the chemical in the parabolic limit.

        alpha:
            The chemotactic sensitivity of the cells.

        tau:
            The time scale of the chemical.

        n0:
            The mass of each bump of cells.

        width:
            The width of each bump of cells.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.physiology import HyperbolicChemotaxisSystem
        >>> round(HyperbolicChemotaxisSystem(20).prepare().system_.elements[0].values['mu1'])
        5000
    """

    field = 'biology/physiology'
    topics = ('chemotaxis', 'hyperbolic', 'kinetic theory', 'field', 'pde')
    references = ('Bellomo et al. 2017',)
    until = 0.015
    dt = 0.0005
    solver = ScipyODESolver(method='BDF')

    def __init__(
        self,
        sites: int = 40,
        *,
        s: float = 100.0,
        D_n: float = 1.0,
        D_c: float = 0.001,
        alpha: float = 0.33,
        tau: float = 1.0,
        n0: float = 0.25,
        width: float = 0.03,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.sites = sites
        self.s = s
        self.D_n = D_n
        self.D_c = D_c
        self.alpha = alpha
        self.tau = tau
        self.n0 = n0
        self.width = width

    def _build_system(self) -> System:
        """Return the parts of the system."""
        spacing = 0.8 / self.sites
        axis = -0.4 + spacing * (np.arange(self.sites) + 0.5)
        x, y = np.meshgrid(axis, axis, indexing='ij')
        offset = 3 * self.width
        bumps = np.exp(-((x - offset) ** 2 + (y - offset) ** 2) / (2 * self.width**2))
        bumps += np.exp(-((x + offset) ** 2 + (y + offset) ** 2) / (2 * self.width**2))
        turning = self.s**2 / (2 * self.D_n)
        values = {
            'n': self.n0 / (2 * np.pi * self.width**2) * bumps,
            's': self.s,
            'mu1': turning,
            'mu2': turning,
            'sigma': self.s**2 / (2 * self.D_c),
            'tau': self.tau,
            'alpha': self.alpha,
        }
        space = LatticeSpace((self.sites, self.sites), boundary='periodic', spacing=spacing)
        return System(
            self.label,
            elements=[FieldElement('tissue', element=ChemotacticTissueElement, space=space, values=values)],
        )
