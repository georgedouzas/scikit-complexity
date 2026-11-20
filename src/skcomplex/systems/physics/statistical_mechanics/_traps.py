"""Trap systems.

Walkers escape from traps of random depth after activated waiting times.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import exp, oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element, VisualSpec
from ....core.components.interactions import Interaction
from ....core.components.laws import JumpLaw
from ....core.components.laws.utils import draw, uniform
from ....core.components.spaces import BaseSpace, ContinuousSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Role


class TrapWalkerElement(Element):
    """Walker in a trap of depth `E` at the place `x`.

    A `TrapEscapeInteraction` makes the walker escape after an exponential time of mean `tau0 exp(E / T)`.

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

        size:
            The number of walkers, or `None` for one.

        space:
            The square of the traps, or `None` for none.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import TrapWalkerElement
        >>> TrapWalkerElement('walker', T=0.5).T
        0.5
    """

    visual = VisualSpec('points', {'position': 'x', 'color_by': 'E', 'color_range': (0.0, 5.0)})

    def __init__(
        self,
        label: str,
        *,
        x: npt.ArrayLike = 0.5,
        E: Annotated[object, RealSet.closed_open(0, oo)] = 1.0,
        T: Annotated[float, RealSet.open(0, oo)] = 0.5,
        tau0: Annotated[float, RealSet.open(0, oo)] = 1.0,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, size=size, space=space)
        self.x = x
        self.E = E
        self.T = T
        self.tau0 = tau0


class TrapEscapeInteraction(Interaction):
    """Escape of walkers from their traps after activated waiting times.

    A walker in a trap of depth `E` escapes after an exponential time of mean `tau0 exp(E / T)` and falls into a new
    trap of the fully connected landscape, at a uniformly random place in the unit square, whose depth is drawn from
    the density `exp(-E)`. It is a carrier hopping between the defects of a glass, or a protein between its folds.

    Args:
        label:
            The label of the interaction, which names the count of escapes.

        target:
            The path of the walkers.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import TrapEscapeInteraction
        >>> TrapEscapeInteraction('escapes', target='walkers').target
        'walkers'
    """

    law_classes = (JumpLaw,)

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the escapes of the walkers.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context.target
            reset = {item: uniform(f'place{index}') for index, item in enumerate(s.x)}
            waiting = draw('escape', 'expon', scale=s.tau0 * exp(s.E / s.T))
            self.laws_ = [JumpLaw(waiting=waiting, reset={**reset, s.E: draw('depth', 'expon')}, label=self.label)]


class TrapSystem(BaseSystem):
    """The Bouchaud trap system.

    Walkers escape from traps of exponentially distributed depths.

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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import TrapSystem
        >>> TrapSystem(10, T=0.8).prepare().system_.elements[0].T
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_walkers = n_walkers
        self.T = T
        self.tau0 = tau0
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the walkers in traps of random depths and places.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        rng = np.random.default_rng(self.random_state)
        square = ContinuousSpace(n_dim=2)
        walkers = TrapWalkerElement(
            'walkers',
            x=rng.random((self.n_walkers, 2)),
            E=rng.exponential(1.0, self.n_walkers),
            T=self.T,
            tau0=self.tau0,
            size=self.n_walkers,
            space=square,
        )
        escapes = TrapEscapeInteraction('escapes', target='walkers')
        self.system_ = System(self.label, elements=[walkers], interactions=[escapes], space=square)
