"""Swarmalators: oscillators that swarm, whose phases and positions couple to each other."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import cos, oo, sin, sqrt
from sage.sets.real_set import RealSet

from ....core.elements import GroupElement
from ....core.interactions import PairCouplingInteraction
from ....core.spaces import ContinuousSpace
from ....core.system import System
from ....core.types import Target
from ....engine.solvers import RK4Solver
from ....specs import ParameterSpec, RoleSpec, VisualSpec
from ..._catalog import register_model
from ..common.oscillators import SitedOscillatorElement


class SwarmalatorCouplingInteraction(PairCouplingInteraction):
    """Coupling of two swarmalators through their separation `r` and their phase difference `dtheta`.

    Each swarmalator moves toward the other at the speed `(A + J cos(dtheta)) / N`, away from it at `B / (N |r|)`, and
    turns its phase toward the other's at the rate `K sin(dtheta) / (N |r|)`, as in O'Keeffe, Ceron, and Petersen. It
    is a pair coupling of the positions and the phases jointly.

    Args:
        label:
            The label of the interaction.

        first:
            The path of the group of the first members.

        second:
            The path of the group of the second members.

        cutoff:
            The distance beyond which pairs do not interact, large for the all-to-all model.

        position:
            The name of the vector state that holds the positions of the members.

        A:
            The strength of the attraction.

        B:
            The strength of the repulsion.

        J:
            The influence of the phase similarity on the attraction.

        K:
            The phase coupling, synchronizing when positive.

        period:
            The side of a periodic box, for minimum image distances, or `None` for an open space.

        n_units:
            The number of swarmalators `N` that normalizes the coupling.

    Examples:
        >>> from skcomplex.models.physics.synchronization import SwarmalatorCouplingInteraction
        >>> SwarmalatorCouplingInteraction('coupling', first='units', second='units', J=0.5).J
        0.5
    """

    def __init__(
        self,
        label: str,
        first: Annotated[str, RoleSpec()],
        second: Annotated[str, RoleSpec()],
        cutoff: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1000000.0,
        position: str = 'x',
        *,
        A: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        B: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        J: Annotated[float, ParameterSpec()] = 1.0,
        K: Annotated[float, ParameterSpec()] = 0.0,
        period: float | None = None,
        n_units: int = 1,
    ) -> None:
        super().__init__(
            label,
            first,
            second,
            cutoff=cutoff,
            position=position,
            write=(position, 'theta'),
            function=self._get_terms,
            period=period,
        )
        self.A = A
        self.B = B
        self.J = J
        self.K = K
        self.n_units = n_units

    def _get_terms(self, s: SimpleNamespace) -> tuple[Target, Target]:
        """Return the move and the phase turn of the first member of a pair toward the second."""
        separation = getattr(s.second, self.position) - getattr(s.first, self.position)
        r = sqrt(separation.dot_product(separation))
        difference = s.second.theta - s.first.theta
        move = ((s.A + s.J * cos(difference)) / r - s.B / r**2) * separation / self.n_units
        return move, s.K * sin(difference) / (self.n_units * r)


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
    units = GroupElement(
        'units',
        element=SitedOscillatorElement,
        size=n_units,
        values={'x': rng.uniform(-1.0, 1.0, (n_units, 2)), 'theta': rng.uniform(-np.pi, np.pi, n_units), 'omega': 0.0},
        visual=VisualSpec('points', {'position': 'x', 'color_by': 'theta', 'radius': 0.03}),
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
    return System(elements=[units], interactions=[coupling], space=ContinuousSpace(n_dim=2))
