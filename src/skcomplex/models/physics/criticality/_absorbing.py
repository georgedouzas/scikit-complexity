"""Absorbing-state transitions: the Langevin field theory of directed percolation coupled to an energy field."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import numpy as np
from sage.all__sagemath_symbolics import sqrt

from skcomplex.core.declarations import parameter, state

from ....core.components import Element
from ....core.elements import FieldElement
from ....core.laws import BoundaryLaw, NoiseLaw, RateLaw
from ....core.laws.utils import laplacian
from ....core.spaces import LatticeSpace
from ....core.system import System
from ....engine.runtime import register_model
from ....engine.solvers import EulerMaruyamaSolver
from ...core.base import BaseLaw
from ...core.laws import Symbols


class ActiveFieldElement(Element):
    """Active field element."""

    rho = state(0.5)
    nu = state(1.0)
    a = parameter(-1.0)
    b = parameter(1.0)
    c = parameter(1.0)
    d = parameter(1.0)
    e = parameter(0.2)
    f = parameter(0.0)
    g = parameter(0.0)

    def laws(self, s: Symbols) -> list[BaseLaw]:
        """Return the laws of the active field element."""
        return [
            RateLaw(s.rho, (s.a + s.b * s.nu) * s.rho - s.c * s.rho**2 + s.d * laplacian(s.rho)),
            NoiseLaw(s.rho, s.e * sqrt(s.rho)),
            RateLaw(s.nu, laplacian(s.nu) + s.f - s.g * s.rho),
            BoundaryLaw(s.rho, (s.rho + abs(s.rho)) / 2),
            BoundaryLaw(s.nu, (s.nu + abs(s.nu)) / 2),
        ]


@register_model(
    field='physics/criticality',
    tags=('field', 'absorbing state', 'directed percolation', 'multiplicative noise', 'spde'),
    references=('Tian et al. 2022', 'Henkel, Hinrichsen, and Lübeck 2008', 'Bonachela and Muñoz 2009'),
    defaults={'until': 100.0, 'dt': 1.0, 'solver': EulerMaruyamaSolver(dt=0.01)},
)
def directed_percolation_langevin(
    shape: Sequence[int] = (64, 64),
    *,
    a: float = -1.0,
    b: float = 1.0,
    c: float = 1.0,
    d: float = 1.0,
    e: float = 0.2,
    f: float = 0.0,
    g: float = 0.0,
    energy: float = 1.2,
    density: float = 0.5,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the Langevin field theory of an absorbing-state transition, an active density fed by an energy field.

    The active density `rho` grows at the local rate `a + b nu`, saturates through `-c rho^2`, spreads with the
    diffusivity `d`, and fluctuates with the multiplicative noise `e sqrt(rho)`, which vanishes in the absorbing state
    `rho = 0`. The energy field `nu` diffuses, gains the drive `f`, and loses the dissipation `g rho`, so
    `rho_t = (a + b nu) rho - c rho^2 + d laplacian(rho) + e sqrt(rho) xi` and `nu_t = laplacian(nu) + f - g rho`
    (Tian et al., Eqs. 3-4). Both fields stay nonnegative. Without drive and dissipation the energy is a fixed control
    parameter: below the threshold `nu_c = -a / b` activity dies out, and above it the density settles near
    `(a + b nu) / c`.

    Args:
        shape:
            The number of sites along every axis of the periodic lattice.

        a:
            The bare growth rate of the activity.

        b:
            The gain of the growth rate per unit of energy.

        c:
            The saturation of the activity.

        d:
            The diffusivity of the activity.

        e:
            The amplitude of the multiplicative noise.

        f:
            The drive of the energy.

        g:
            The dissipation of the energy per unit of activity.

        energy:
            The initial energy at every site.

        density:
            The mean initial activity, drawn uniformly between 0 and twice this value.

        random_state:
            The seed or generator of the initial activity.

    Returns:
        system:
            The model, with the activity `neurons.rho` and the energy `neurons.nu`.

    Examples:
        >>> from skcomplex.models.physics.criticality import directed_percolation_langevin
        >>> float(directed_percolation_langevin((8, 8), energy=0.5).elements[0].values['nu'].mean())
        0.5
    """
    shape = tuple(shape)
    rng = np.random.default_rng(random_state)
    values = {
        'rho': rng.uniform(0.0, 2 * density, shape),
        'nu': np.full(shape, energy),
        'a': a,
        'b': b,
        'c': c,
        'd': d,
        'e': e,
        'f': f,
        'g': g,
    }
    return System(elements=[FieldElement('neurons', ActiveFieldElement, LatticeSpace(shape), values=values)])
