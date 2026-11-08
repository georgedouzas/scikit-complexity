"""Define the accelerated spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import dataclass

from ..types import Array
from ._base_spec import BaseSpec


@dataclass(frozen=True)
class AcceleratedSpec(BaseSpec):
    """Compiled stepper loops of a problem.

    They run every internal step of a frame in compiled code. Every loop takes `(t, x, p, until, ...)` and returns the
    state at `until`. A loop is `None` when the problem's form does not use it.

    Args:
        rk4:
            The fourth-order Runge-Kutta loop `rk4(t, x, p, until, h)`.

        velocity_verlet:
            The velocity Verlet loop `velocity_verlet(t, x, p, until, h, positions, velocities)`.

        yoshida4:
            The fourth-order Yoshida loop `yoshida4(t, x, p, until, h, first, second, weights)`, which drifts the first
            block and kicks the second in every velocity Verlet stage of a weight.

        euler_maruyama:
            The Euler-Maruyama loop `euler_maruyama(t, x, p, until, h, rng)`.

        gillespie:
            The Gillespie loop `gillespie(t, x, p, until, rng)`.

        iterate:
            The discrete map loop `iterate(t, x, p, until)`, or `iterate(t, x, p, until, rng)` for a stochastic map.
    """

    rk4: Callable[..., Array] | None = None
    velocity_verlet: Callable[..., Array] | None = None
    yoshida4: Callable[..., Array] | None = None
    euler_maruyama: Callable[..., Array] | None = None
    gillespie: Callable[..., Array] | None = None
    iterate: Callable[..., Array] | None = None
