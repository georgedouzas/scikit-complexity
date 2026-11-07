"""Size-structured forest: a density of trees over position and height with shading, competition, and dispersal."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

import numpy as np
from sage.all__sagemath_symbolics import Expression, exp

from ....core.components.elements import DensityElement, FieldElement
from ....core.components.interactions import AdvectionInteraction, RateInteraction
from ....core.components.laws.utils import (
    convolve,
    coordinate,
    cumulative_integral,
    displacement,
    reduce_axes,
    where,
)
from ....core.components.spaces import LatticeSpace, PhaseSpace
from ....core.components.system import System
from ....core.engine.solvers import RK4Solver
from ....core.specs import AxisSpec
from ..._catalog import register_system

HEIGHT_AXIS = 1


@register_system(
    field='biology/ecology',
    tags=('field', 'size structure', 'forest', 'integral operator', 'dispersal', 'pde'),
    references=('Grilletta et al. 2026',),
    defaults={'until': 200.0, 'dt': 5.0, 'solver': RK4Solver(dt=0.1)},
)
def forest_size_structure(
    positions: int = 40,
    heights: int = 100,
    *,
    width: float = 10.0,
    dispersal: float = 1.0,
    hurst: float = 1.0,
    sapling: float = 0.1,
    top: float = 50.0,
    shading: float = 1.0,
    fecundity: float = 0.01,
    resources: float = 125.0,
) -> System:
    """Return the McKendrick-von Foerster density of trees over position across a strip and height.

    The density `f(x, h)` of trees of height `h` at position `x` across a forest strip grows in height at the rate
    `I (1 - h / top)`, where the light `I = exp(-shading int_h^top f(x, h') h'^(2H) dh')` falls through the canopy of
    taller trees by the Beer-Lambert law. Trees die from competition for canopy space at the rate `I h^(2H) f` and for
    resources at the rate `I (1 - h / top) a / h`, where `a = R / resources` and `R = int h^(1+2H) f dh` is the
    resource consumption of the trees at `x`. Saplings enter at the height `sapling` from the seeds `int K(x - x') R(x')
    dx'` of the trees around, dispersed by the exponential kernel `K` of length `dispersal` and integral `fecundity`,
    and no seed comes from beyond the edges of the strip, where the forest was removed. Time and density are rescaled
    so that the growth rate and the canopy mortality are one (Grilletta et al. 2026, arXiv 2608.11918 Eq. 9, 10, 16,
    and 25). The kernel is the exponential of Eq. 35 along the axis across the strip, normalized in one dimension.

    Args:
        positions:
            The number of positions across the strip.

        heights:
            The number of cells of the height axis.

        width:
            The width `L` of the strip.

        dispersal:
            The dispersal length `xi`.

        hurst:
            The Hurst exponent `H` of the crown radius, which scales as `h^H`.

        sapling:
            The height `h0` at which saplings enter.

        top:
            The asymptotic height `h_u`.

        shading:
            The light attenuation `gamma`.

        fecundity:
            The integral `kappa` of the dispersal kernel.

        resources:
            The resource flux `R_a`.

    Returns:
        system:
            The system, with the density `forest.f` over position and height.

    Examples:
        >>> from skcomplex.systems.biology.ecology import forest_size_structure
        >>> forest_size_structure(8, 12).elements[0].space.prepare().shape_
        (8, 12)
    """
    step = (top - sapling) / heights
    space = PhaseSpace(
        LatticeSpace((positions,), spacing=width / positions, boundary='fixed'),
        [AxisSpec('h', heights, sapling, top)],
    )
    centres = space.prepare().coordinates_['h']
    start = np.tile(0.1 * (centres / sapling) ** -2.0, (positions, 1))
    forest = FieldElement(
        'forest',
        element=DensityElement,
        space=space,
        values={'f': start},
        discretization='finite-volume',
    )
    h = coordinate('h')
    growth = 1 - h / top
    kernel = fecundity / (2 * dispersal) * exp(-abs(displacement(0)) / dispersal)

    def light(s: SimpleNamespace) -> Expression:
        return exp(-shading * cumulative_integral(s.target.f * h ** (2 * hurst), HEIGHT_AXIS, reverse=True))

    def demography(s: SimpleNamespace) -> Expression:
        consumption = reduce_axes(s.target.f * h ** (1 + 2 * hurst), ('h',))
        recruits = where(h < sapling + step, light(s) * growth * convolve(consumption, kernel) / step, 0)
        mortality = light(s) * (h ** (2 * hurst) * s.target.f + growth * consumption / (resources * h))
        return recruits - mortality * s.target.f

    grow = AdvectionInteraction(
        'growth',
        target='forest',
        state='f',
        drift=lambda s: [0, light(s) * growth],
        limiter=None,
    )
    demographics = RateInteraction('demography', target='forest', state='f', rate=demography)
    return System(elements=[forest], interactions=[grow, demographics])
