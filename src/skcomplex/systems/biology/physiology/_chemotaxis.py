"""Physiology systems: chemotaxis of cells that move with finite speed."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np

from ....core.components.elements import FieldElement
from ....core.components.spaces import LatticeSpace
from ....core.components.system import System
from ....core.engine.solvers import ScipyODESolver
from ..._catalog import register_system
from ._chemotactic_tissue import ChemotacticTissueElement


@register_system(
    field='biology/physiology',
    tags=('chemotaxis', 'hyperbolic', 'kinetic theory', 'field', 'pde'),
    references=('Bellomo et al. 2017',),
    defaults={'until': 0.015, 'dt': 0.0005, 'solver': ScipyODESolver(method='BDF')},
)
def hyperbolic_chemotaxis(
    sites: int = 40,
    *,
    s: float = 100.0,
    D_n: float = 1.0,
    D_c: float = 0.001,
    alpha: float = 0.33,
    tau: float = 1.0,
    n0: float = 0.25,
    width: float = 0.03,
) -> System:
    """Return cells and a chemical moving with finite speed, whose two bumps of cells spread and attract each other.

    The model is the hyperbolic limit of a kinetic model of cells and a chemical that change velocity at random, with
    the speed `s` and turning rates set by the diffusions `D_n = s^2 / (2 mu1)` and `D_c = s^2 / (2 sigma)` of its
    parabolic limit, Keller-Segel, and `mu2 = mu1`. Following the test of Bellomo et al. (2017, section 6.6.2), the
    cells start as two Gaussian bumps of mass `n0` and width `width` at `(3 width, 3 width)` and `(-3 width, -3 width)`
    in the square `[-0.4, 0.4]^2`, here with periodic edges, the cells secrete the chemical, and all fluxes start at
    rest. The fluxes relax fast, so the bumps spread out at a finite speed while the cell mass `2 n0` is conserved and
    the chemical accumulates where the cells are.

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

    Returns:
        system:
            The system, with the cell density `tissue.n`, the chemical `tissue.c`, and their fluxes.

    Examples:
        >>> from skcomplex.systems.biology.physiology import hyperbolic_chemotaxis
        >>> round(hyperbolic_chemotaxis(20).elements[0].values['mu1'])
        5000
    """
    spacing = 0.8 / sites
    axis = -0.4 + spacing * (np.arange(sites) + 0.5)
    x, y = np.meshgrid(axis, axis, indexing='ij')
    offset = 3 * width
    bumps = np.exp(-((x - offset) ** 2 + (y - offset) ** 2) / (2 * width**2))
    bumps += np.exp(-((x + offset) ** 2 + (y + offset) ** 2) / (2 * width**2))
    turning = s**2 / (2 * D_n)
    values = {
        'n': n0 / (2 * np.pi * width**2) * bumps,
        's': s,
        'mu1': turning,
        'mu2': turning,
        'sigma': s**2 / (2 * D_c),
        'tau': tau,
        'alpha': alpha,
    }
    space = LatticeSpace((sites, sites), boundary='periodic', spacing=spacing)
    return System(elements=[FieldElement('tissue', element=ChemotacticTissueElement, space=space, values=values)])
