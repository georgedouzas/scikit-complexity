"""Swarmalators: oscillators that swarm, whose phases and positions couple to each other."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import cos, sin, sqrt

from ....core.base import Domain, VisualSpec, parameter, role, state
from ....core.components import Element
from ....core.interactions import Interaction
from ....core.laws import Law, Rate, Symbols
from ....core.runtime import register_model
from ....core.solvers import RK4
from ....core.space import ContinuousSpace
from ....core.system import System


class Swarmalator(Element):
    """Oscillator in the plane with a position `x` and a phase `theta` that turns at the natural frequency `omega`.

    Args:
        label:
            The label of the swarmalator or the group.

        x:
            The initial position.

        theta:
            The initial phase.

        omega:
            The natural frequency.

    Examples:
        >>> from skcomplex.models.physics.synchronization import Swarmalator
        >>> Swarmalator('unit').omega
        0.0
    """

    visual: ClassVar[VisualSpec] = VisualSpec('points', {'position': 'x', 'color_by': 'theta', 'radius': 0.03})
    x: npt.ArrayLike = state(default=0.0, vector=True, position=True)
    theta: float = state(default=0.0, domain=Domain.ANGLE)
    omega: float = parameter(default=0.0)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the turning of the phase at the natural frequency.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rate(s.theta, s.omega)]


class SwarmalatorCoupling(Interaction):
    """Coupling of two swarmalators through their separation `r` and their phase difference `dtheta`.

    Each swarmalator moves toward the other at the speed `(A + J cos(dtheta)) / N`, away from it at `B / (N |r|)`, and
    turns its phase toward the other's at the rate `K sin(dtheta) / (N |r|)`, as in O'Keeffe, Ceron, and Petersen.

    Args:
        label:
            The label of the interaction.

        first:
            The path of the group of the first members.

        second:
            The path of the group of the second members.

        cutoff:
            The distance beyond which pairs do not interact, large for the all-to-all model.

        period:
            The side of a periodic box, for minimum image distances, or `None` for an open space.

        A:
            The strength of the attraction.

        B:
            The strength of the repulsion.

        J:
            The influence of the phase similarity on the attraction.

        K:
            The phase coupling, synchronizing when positive.

        n_units:
            The number of swarmalators `N` that normalizes the coupling.

    Examples:
        >>> from skcomplex.models.physics.synchronization import SwarmalatorCoupling
        >>> SwarmalatorCoupling('coupling', first='units', second='units', J=0.5).J
        0.5
    """

    over: ClassVar[str] = 'pairs'
    position: ClassVar[str] = 'x'
    first: str = role()
    second: str = role()
    cutoff: float = parameter(default=1e6, domain=Domain.POSITIVE)
    period: float | None = None

    A: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    B: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    J: float = parameter(default=1.0)
    K: float = parameter(default=0.0)
    n_units: int = 1

    def laws(self, s: Symbols) -> list[Law]:
        """Return the opposite moves and the opposite phase turns of the two members of a pair.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        separation = s.second.x - s.first.x
        r = sqrt(separation.dot_product(separation))
        difference = s.second.theta - s.first.theta
        move = ((s.A + s.J * cos(difference)) / r - s.B / r**2) * separation / self.n_units
        turn = s.K * sin(difference) / (self.n_units * r)
        return [Rate(s.first.x, move), Rate(s.second.x, -move), Rate(s.first.theta, turn), Rate(s.second.theta, -turn)]


@register_model(
    field='physics/synchronization',
    tags=('swarmalators', 'synchronization', 'swarming', 'pair interaction', 'ode'),
    references=("O'Keeffe, Ceron, and Petersen 2017", 'te Vrugt and Wittkowski 2024'),
    defaults={'until': 100.0, 'dt': 1.0, 'solver': RK4(dt=0.1)},
)
def swarmalators(
    n_units: int = 100,
    *,
    J: float = 1.0,
    K: float = 0.0,
    A: float = 1.0,
    B: float = 1.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return swarmalators, identical oscillators whose phases set their attraction and whose positions their coupling.

    Every swarmalator `i` moves by `(1 / N) sum_j [(x_j - x_i) / |x_j - x_i| (A + J cos(theta_j - theta_i)) -
    B (x_j - x_i) / |x_j - x_i|^2]` and turns by `(K / N) sum_j sin(theta_j - theta_i) / |x_j - x_i|`. With `J > 0`
    like phases attract. With `K = 0` and `J = 1` the swarm settles into a static phase wave, a disk whose phase winds
    once around its center, so one of the order parameters `S± = |mean(exp(i (phi ± theta)))|` of the spatial angle
    `phi` and the phase approaches 1. With `K > 0` the phases synchronize into a uniform disk, and with `K < 0` and
    small `J` they stay asynchronous.

    Args:
        n_units:
            The number of swarmalators `N`.

        J:
            The influence of the phase similarity on the attraction.

        K:
            The phase coupling.

        A:
            The strength of the attraction.

        B:
            The strength of the repulsion.

        random_state:
            The seed or generator of the initial positions in the square of side 2 and of the initial phases.

    Returns:
        system:
            The model, with the positions `units.x` and the phases `units.theta`.

    Examples:
        >>> from skcomplex.models.physics.synchronization import swarmalators
        >>> swarmalators(10, K=-0.1).interactions[0].K
        -0.1
    """
    rng = np.random.default_rng(random_state)
    units = Swarmalator.group(
        'units',
        n_units,
        x=rng.uniform(-1.0, 1.0, (n_units, 2)),
        theta=rng.uniform(-np.pi, np.pi, n_units),
    )
    coupling = SwarmalatorCoupling('coupling', first='units', second='units', A=A, B=B, J=J, K=K, n_units=n_units)
    return System(components=[units], interactions=[coupling], space=ContinuousSpace(n_dim=2))
