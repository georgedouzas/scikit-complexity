"""Trap systems: walkers that escape from traps of random depth after activated waiting times."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import exp, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.laws import BaseLaw, JumpLaw
from ....core.components.laws.utils import draw, uniform
from ....core.components.spaces import ContinuousSpace
from ....core.components.system import BaseSystem, System
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class TrapWalkerElement(Element):
    """Walker in a trap of depth `E`, which escapes after an exponential time of mean `tau0 exp(E / T)`.

    On escape the walker falls into a new trap of the fully connected landscape, at a uniformly random place in the
    unit square, whose depth is drawn from the density `exp(-E)`.

    Args:
        label:
            The label of the walker or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The place of the trap.

        E:
            The depth of the trap.

        T:
            The temperature, in units of the mean depth.

        tau0:
            The escape time from a trap of zero depth.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import TrapWalkerElement
        >>> TrapWalkerElement('walker', T=0.5).T
        0.5
    """

    law_classes = (JumpLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('points', {'position': 'x', 'color_by': 'E', 'color_range': (0.0, 5.0)}),
        *,
        x: Annotated[npt.ArrayLike, StateSpec(vector=True, role='position')] = 0.5,
        E: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        T: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
        tau0: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.E = E
        self.T = T
        self.tau0 = tau0

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
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
        return [JumpLaw(waiting=waiting, reset={**reset, s.E: draw('depth', 'expon')}, label='escapes')]


class TrapSystemSystem(BaseSystem):
    """The Bouchaud trap system, walkers that escape from traps of exponentially distributed depths.

    The escape times `tau0 exp(E / T)` of depths with the density `exp(-E)` have the tail `tau^(-1 - T)`. Below the
    glass temperature `T = 1` their mean diverges, so the walkers fall into ever deeper traps, the number of escapes
    grows as `t^T` instead of linearly, and the system ages without reaching equilibrium. The system has the places
    `walkers.x`, the depths `walkers.E`, and the escape count `escapes.renewals`.

    Args:
        n_walkers:
            The number of independent walkers.

        T:
            The temperature, in units of the mean depth.

        tau0:
            The escape time from a trap of zero depth.

        random_state:
            The seed or generator of the first traps.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import TrapSystemSystem
        >>> TrapSystemSystem(10, T=0.8).prepare().system_.elements[0].values['T']
        0.8
    """

    field = 'physics/statistical_mechanics'
    topics = ('trap system', 'renewal process', 'aging', 'glass')
    references = ('Bouchaud 1992', 'Monthus and Bouchaud 1996', 'Bertin 2021')
    until = 10000.0
    dt = 50.0

    def __init__(
        self,
        n_walkers: int = 500,
        *,
        T: float = 0.5,
        tau0: float = 1.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_walkers = n_walkers
        self.T = T
        self.tau0 = tau0
        self.random_state = random_state

    def _build_system(self) -> System:
        """Return the Bouchaud trap system, walkers that escape from traps of exponentially distributed depths."""
        rng = np.random.default_rng(self.random_state)
        walkers = GroupElement(
            'walkers',
            element=TrapWalkerElement,
            size=self.n_walkers,
            values={
                'x': rng.random((self.n_walkers, 2)),
                'E': rng.exponential(1.0, self.n_walkers),
                'T': self.T,
                'tau0': self.tau0,
            },
        )
        return System(self.label, elements=[walkers], space=ContinuousSpace(n_dim=2))
