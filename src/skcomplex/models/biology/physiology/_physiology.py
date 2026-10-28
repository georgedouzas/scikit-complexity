"""Physiology models: delayed feedback and excitable tissue."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, FieldElement
from ....core.laws import BaseLaw, RateLaw
from ....core.laws.utils import delayed, laplacian
from ....core.spaces import LatticeSpace
from ....core.system import System
from ....engine.runtime import register_model
from ....engine.solvers import RK4Solver
from ....specs import ParameterSpec, StateSpec, VisualSpec


class FitzHughNagumoElement(Element):
    """Fitz hugh nagumo element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        u: Annotated[object, StateSpec()] = 0.0,
        v: Annotated[object, StateSpec()] = 0.0,
        D: Annotated[object, ParameterSpec()] = 1.0,
        a: Annotated[object, ParameterSpec()] = 0.7,
        b: Annotated[object, ParameterSpec()] = 0.8,
        epsilon: Annotated[object, ParameterSpec()] = 0.02,
    ) -> None:
        super().__init__(label, visual=visual)
        self.u = u
        self.v = v
        self.D = D
        self.a = a
        self.b = b
        self.epsilon = epsilon

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the fitz hugh nagumo element."""
        return [
            RateLaw(s.u, s.D * laplacian(s.u) + s.u - s.u**3 / 3 - s.v),
            RateLaw(s.v, s.epsilon * (s.u + s.a - s.b * s.v)),
        ]


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
        >>> fitzhugh_nagumo((20, 20)).elements[0].values['epsilon']
        0.02
    """
    rows, columns = shape
    u, v = np.full(shape, -1.2), np.full(shape, -0.62)
    u[: rows // 2, :] = 1.5
    v[:, : columns // 2] = 1.0
    values = {'u': u, 'v': v, 'D': D, 'a': a, 'b': b, 'epsilon': epsilon}
    space = LatticeSpace((rows, columns), boundary='reflecting')
    return System(elements=[FieldElement('medium', element=FitzHughNagumoElement, space=space, values=values)])


class MackeyGlassElement(Element):
    """Mackey glass element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        beta: Annotated[object, ParameterSpec()] = 0.2,
        gamma: Annotated[object, ParameterSpec()] = 0.1,
        n: Annotated[object, ParameterSpec()] = 10.0,
        tau: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 17.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.beta = beta
        self.gamma = gamma
        self.n = n
        self.tau = tau

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the mackey glass element."""
        return [RateLaw(s.x, s.beta * delayed(s.x, s.tau) / (1 + delayed(s.x, s.tau) ** s.n) - s.gamma * s.x)]


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
