"""Physiology models: delayed feedback and excitable tissue."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import numpy as np

from ....core.elements import FieldElement
from ....core.interactions import DiffusionInteraction
from ....core.spaces import LatticeSpace
from ....core.system import System
from ....engine.solvers import RK4Solver
from ..._catalog import register_model
from ..common.excitable import FitzHughNagumoInteraction
from ._fitz_hugh_nagumo import FitzHughNagumoElement
from ._mackey_glass import MackeyGlassElement


@register_model(
    field='biology/physiology',
    tags=('field', 'excitable media', 'spirals', 'pde'),
    references=('FitzHugh 1961', 'Nagumo, Arimoto, and Yoshizawa 1962', 'Gros 2015'),
    defaults={'until': 200.0, 'dt': 2.0, 'solver': RK4Solver(dt=0.05)},
)
def fitzhugh_nagumo(
    shape: Sequence[int] = (100, 100),
    *,
    D: float = 1.0,
    a: float = 0.7,
    b: float = 0.8,
    epsilon: float = 0.02,
) -> System:
    """Return the FitzHugh-Nagumo excitable medium, started from a broken wave that curls into a spiral.

    The upper half starts excited and the left half refractory, so the free end of the wave rotates.

    Args:
        shape:
            The number of rows and columns of the lattice with zero-flux edges.

        D:
            The diffusion of the potential `u`.

        a:
            The offset of the recovery.

        b:
            The damping of the recovery.

        epsilon:
            The time scale of the recovery.

    Returns:
        system:
            The model, with the potential `medium.u` and the recovery `medium.v`.

    Examples:
        >>> from skcomplex.models.biology.physiology import fitzhugh_nagumo
        >>> fitzhugh_nagumo((20, 20)).interactions[0].epsilon
        0.02
    """
    rows, columns = shape
    u, v = np.full(shape, -1.2), np.full(shape, -0.62)
    u[: rows // 2, :] = 1.5
    v[:, : columns // 2] = 1.0
    space = LatticeSpace((rows, columns), boundary='reflecting')
    medium = FieldElement('medium', element=FitzHughNagumoElement, space=space, values={'u': u, 'v': v})
    kinetics = FitzHughNagumoInteraction('kinetics', target='medium', fast='u', slow='v', a=a, b=b, epsilon=epsilon)
    diffusion = DiffusionInteraction('diffusion', target='medium', state='u', coefficient=D)
    return System(elements=[medium], interactions=[kinetics, diffusion])


@register_model(
    field='biology/physiology',
    tags=('chaos', 'delay', 'physiology'),
    references=('Mackey and Glass 1977', 'Erneux 2009'),
    defaults={'until': 500.0, 'dt': 0.5},
)
def mackey_glass(
    x: float = 0.5,
    *,
    beta: float = 0.2,
    gamma: float = 0.1,
    n: float = 10.0,
    tau: float = 17.0,
) -> System:
    """Return the Mackey-Glass equation `x' = beta x(t - tau) / (1 + x(t - tau)^n) - gamma x` of blood cell production.

    It is periodic for `tau = 6` and chaotic for `tau = 17`. The history before the start is the initial value.

    Args:
        x:
            The initial concentration, also its history.

        beta:
            The production rate.

        gamma:
            The decay rate.

        n:
            The steepness of the production feedback.

        tau:
            The delay of production.

    Returns:
        system:
            The model, with the concentration `cells.x`.

    Examples:
        >>> from skcomplex.models.biology.physiology import mackey_glass
        >>> mackey_glass(tau=6.0).elements[0].tau
        6.0
    """
    return System(elements=[MackeyGlassElement('cells', x=x, beta=beta, gamma=gamma, n=n, tau=tau)])
