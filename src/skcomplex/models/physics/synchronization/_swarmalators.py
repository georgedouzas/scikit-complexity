"""Swarmalators: oscillators that swarm, whose phases and positions couple to each other."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import cos, sin, sqrt

from ....core.base import BaseInteraction, BaseLaw, NumericDomain, VisualSpec, parameter, role, state
from ....core.components import Element
from ....core.laws import RateLaw, Symbols
from ....core.space import ContinuousSpace
from ....core.system import System
from ....engine.runtime import register_model
from ....engine.solvers import RK4Solver


class SwarmalatorElement(Element):
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
        >>> from skcomplex.models.physics.synchronization import SwarmalatorElement
        >>> SwarmalatorElement('unit').omega
        0.0
    """

    visual: ClassVar[VisualSpec] = VisualSpec('points', {'position': 'x', 'color_by': 'theta', 'radius': 0.03})
    x: npt.ArrayLike = state(default=0.0, vector=True, position=True)
    theta: float = state(default=0.0, domain=NumericDomain.ANGLE)
    omega: float = parameter(default=0.0)

    def laws(self, s: Symbols) -> list[BaseLaw]:
        """Return the turning of the phase at the natural frequency.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RateLaw(s.theta, s.omega)]


class SwarmalatorCouplingInteraction(BaseInteraction):
    """Coupling of two swarmalators through their separation `r` and their phase difference `dtheta`.

    Each swarmalator moves toward the other at the speed `(A + J cos(dtheta)) / N`, away from it at `B / (N |r|)`, and
    turns its phase toward the other's at the rate `K sin(dtheta) / (N |r|)`, as in O'Keeffe, Ceron, and Petersen.

    Args:
        label:
            The label of the interaction.

        period:
            The side of a periodic box, for minimum image distances, or `None` for an open space.

        n_units:
            The number of swarmalators `N` that normalizes the coupling.

        **values:
            The values of the declared variables, by name:

            first:
                The path of the group of the first members.

            second:
                The path of the group of the second members.

            cutoff:
                The distance beyond which pairs do not interact, large for the all-to-all model.

            A:
                The strength of the attraction.

            B:
                The strength of the repulsion.

            J:
                The influence of the phase similarity on the attraction.

            K:
                The phase coupling, synchronizing when positive.

    Examples:
        >>> from skcomplex.models.physics.synchronization import SwarmalatorCouplingInteraction
        >>> SwarmalatorCouplingInteraction('coupling', first='units', second='units', J=0.5).J
        0.5
    """

    over: ClassVar[str] = 'pairs'
    position: ClassVar[str] = 'x'
    first: str = role()
    second: str = role()
    cutoff: float = parameter(default=1e6, domain=NumericDomain.POSITIVE)

    A: float = parameter(default=1.0, domain=NumericDomain.NONNEGATIVE)
    B: float = parameter(default=1.0, domain=NumericDomain.NONNEGATIVE)
    J: float = parameter(default=1.0)
    K: float = parameter(default=0.0)

    def __init__(
        self,
        label: str,
        *,
        period: float | None = None,
        n_units: int = 1,
        **values: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(label, **values)
        self.period = period
        self.n_units = n_units

    def laws(self, s: Symbols) -> list[BaseLaw]:
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
        return [
            RateLaw(s.first.x, move),
            RateLaw(s.second.x, -move),
            RateLaw(s.first.theta, turn),
            RateLaw(s.second.theta, -turn),
        ]


@register_model(
    field='physics/synchronization',
    tags=('swarmalators', 'synchronization', 'swarming', 'pair interaction', 'ode'),
    references=("O'Keeffe, Ceron, and Petersen 2017", 'te Vrugt and Wittkowski 2024'),
    defaults={'until': 100.0, 'dt': 1.0, 'solver': RK4Solver(dt=0.1)},
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
    units = SwarmalatorElement.group(
        'units',
        n_units,
        x=rng.uniform(-1.0, 1.0, (n_units, 2)),
        theta=rng.uniform(-np.pi, np.pi, n_units),
    )
    coupling = SwarmalatorCouplingInteraction(
        'coupling',
        first='units',
        second='units',
        A=A,
        B=B,
        J=J,
        K=K,
        n_units=n_units,
    )
    return System(components=[units], interactions=[coupling], space=ContinuousSpace(n_dim=2))
