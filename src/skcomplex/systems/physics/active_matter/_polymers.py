"""Active polymers: chains of self-propelled beads bonded by springs."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np

from ....core.components.elements import FieldElement
from ....core.components.spaces import LatticeSpace
from ....core.components.system import System
from ....core.engine.solvers import EulerMaruyamaSolver
from ..._catalog import register_system
from ._active_bead import ActiveBeadElement


@register_system(
    field='physics/active_matter',
    tags=('active matter', 'sde', 'polymer', 'self-propulsion', 'rouse chain'),
    references=('Winkler and Gompper 2020', 'Osmanović and Rabin 2017', 'Eisenstecken et al. 2016'),
    defaults={'until': 50.0, 'dt': 0.5, 'solver': EulerMaruyamaSolver(dt=0.005)},
)
def active_polymer(
    n_beads: int = 20,
    *,
    v0: float = 1.0,
    k: float = 10.0,
    D_t: float = 0.1,
    D_r: float = 1.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return an active polymer, a Rouse chain of active Brownian beads that each swim along their own orientation.

    Bead `i` follows `r_i' = v0 n_i + k (r_{i+1} + r_{i-1} - 2 r_i) + sqrt(2 D_t) xi_i`, with its orientation
    `n_i = (cos theta_i, sin theta_i)` diffusing with `D_r`, and the end beads bonded to one neighbour. The bonds cancel
    in the motion of the centre of mass, whose mean squared displacement is
    `(4 D_t t + 2 v0² (D_r t - 1 + exp(-D_r t)) / D_r²) / N`, that of an active Brownian particle divided by `N`. The
    activity swells the chain: a passive chain has the mean squared end-to-end distance `2 D_t (N - 1) / k` in the
    plane, and self-propulsion of persistence time `1 / D_r` adds to the bond fluctuations and stretches it further. The
    beads start on a straight line with unit spacing and random orientations.

    Args:
        n_beads:
            The number of beads.

        v0:
            The self-propulsion speed of every bead.

        k:
            The stiffness of the bonds.

        D_t:
            The translational diffusion coefficient.

        D_r:
            The rotational diffusion coefficient.

        random_state:
            The seed or generator of the initial orientations.

    Returns:
        system:
            The system, with the coordinates `chain.x` and `chain.y` and the orientations `chain.theta` of the beads.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import active_polymer
        >>> active_polymer(5).elements[0].prepare().size_
        5
    """
    angles = np.random.default_rng(random_state).uniform(0.0, 2 * np.pi, n_beads)
    chain = FieldElement(
        'chain',
        element=ActiveBeadElement,
        space=LatticeSpace(n_beads, boundary='reflecting'),
        boundary={'x': 'zero-flux', 'y': 'zero-flux'},
        values={
            'x': np.arange(n_beads, dtype=float),
            'y': np.zeros(n_beads),
            'theta': angles,
            'v0': v0,
            'k': k,
            'D_t': D_t,
            'D_r': D_r,
        },
    )
    return System(elements=[chain])
