"""Trap models: walkers that escape from traps of random depth after activated waiting times."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import exp

from ....core.base import Domain, VisualSpec, parameter, state
from ....core.components import Element
from ....core.laws import Jump, Law, Symbols, draw, uniform
from ....core.runtime import register_model
from ....core.space import ContinuousSpace
from ....core.system import System


class TrapWalker(Element):
    """Walker in a trap of depth `E`, which escapes after an exponential time of mean `tau0 exp(E / T)`.

    On escape the walker falls into a new trap of the fully connected landscape, at a uniformly random place in the
    unit square, whose depth is drawn from the density `exp(-E)`.

    Args:
        label:
            The label of the walker or the group.

        x:
            The place of the trap.

        E:
            The depth of the trap.

        T:
            The temperature, in units of the mean depth.

        tau0:
            The escape time from a trap of zero depth.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import TrapWalker
        >>> TrapWalker('walker', T=0.5).T
        0.5
    """

    visual: ClassVar[VisualSpec] = VisualSpec('points', {'position': 'x', 'color_by': 'E', 'color_range': (0.0, 5.0)})
    x: npt.ArrayLike = state(default=0.5, vector=True)
    E: float = state(default=1.0, domain=Domain.NONNEGATIVE)
    T: float = parameter(default=0.5, domain=Domain.POSITIVE)
    tau0: float = parameter(default=1.0, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the escapes after activated waiting times.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        reset = {item: uniform(f'place{index}') for index, item in enumerate(s.x)}
        waiting = draw('escape', 'expon', scale=s.tau0 * exp(s.E / s.T))
        return [Jump(waiting=waiting, reset={**reset, s.E: draw('depth', 'expon')}, label='escapes')]


@register_model(
    field='physics/statistical_mechanics',
    tags=('trap model', 'renewal process', 'aging', 'glass'),
    references=('Bouchaud 1992', 'Monthus and Bouchaud 1996', 'Bertin 2021'),
    defaults={'until': 10000.0, 'dt': 50.0},
)
def trap_model(
    n_walkers: int = 500,
    *,
    T: float = 0.5,
    tau0: float = 1.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the Bouchaud trap model, walkers that escape from traps of exponentially distributed depths.

    The escape times `tau0 exp(E / T)` of depths with the density `exp(-E)` have the tail `tau^(-1 - T)`. Below the
    glass temperature `T = 1` their mean diverges, so the walkers fall into ever deeper traps, the number of escapes
    grows as `t^T` instead of linearly, and the system ages without reaching equilibrium.

    Args:
        n_walkers:
            The number of independent walkers.

        T:
            The temperature, in units of the mean depth.

        tau0:
            The escape time from a trap of zero depth.

        random_state:
            The seed or generator of the first traps.

    Returns:
        system:
            The model, with the places `walkers.x`, the depths `walkers.E`, and the escape count `escapes.renewals`.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import trap_model
        >>> trap_model(10, T=0.8).elements[0].values['T']
        0.8
    """
    rng = np.random.default_rng(random_state)
    walkers = TrapWalker.group(
        'walkers',
        n_walkers,
        x=rng.random((n_walkers, 2)),
        E=rng.exponential(1.0, n_walkers),
        T=T,
        tau0=tau0,
    )
    return System(components=[walkers], space=ContinuousSpace(n_dim=2))
