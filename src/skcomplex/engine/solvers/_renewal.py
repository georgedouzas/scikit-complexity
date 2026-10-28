"""Exact stepper of renewal processes, which jumps from renewal to renewal without a time grid."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

from ...core.laws import BoundaryLaw, EquationLaw, FluxLaw, JumpLaw, NoiseLaw, RateLaw
from ...core.spaces import CayleySpace, ContinuousSpace, ContinuousTime, LatticeSpace, MeshSpace, NetworkSpace
from ...core.types import Vector
from ..problems import Form, NoiseKind, OrderKind, SupportSpec
from ._solver import Solver


class RenewalSolver(Solver):
    """Exact stepper of renewal processes, which jumps from renewal to renewal without a time grid.

    Every member renews on its own clock, and a frame holds the states the last renewal at or before its time set.

    Examples:
        >>> from typing import Annotated
        >>> import numpy as np
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.core.specs import StateSpec
        >>> from skcomplex.engine.compile import compile_system
        >>> from skcomplex.core.elements import Element
        >>> from skcomplex.core.laws import JumpLaw
        >>> from skcomplex.core.laws.utils import draw
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.solvers import RenewalSolver
        >>> class CounterElement(Element):
        ...     def __init__(self, label, n: Annotated[float, StateSpec()] = 0.0):
        ...         super().__init__(label)
        ...         self.n = n
        ...
        ...     def _get_laws(self, s):
        ...         return [JumpLaw(waiting=draw('wait', 'expon'), reset={s.n: s.n + 1})]
        >>> problem = compile_system(System(elements=[GroupElement('counters', element=CounterElement, size=3)]))
        >>> stepper = RenewalSolver().prepare(problem, np.random.default_rng(0))
        >>> state = stepper.run(0.0, problem.initial_state, problem.initial_parameters, 100.0)
        >>> bool(np.all(state[:3] > 50))
        True
    """

    supports: ClassVar[SupportSpec] = SupportSpec(
        time={ContinuousTime},
        space={None, LatticeSpace, NetworkSpace, ContinuousSpace, MeshSpace, CayleySpace},
        laws={JumpLaw, RateLaw, FluxLaw, NoiseLaw, EquationLaw, BoundaryLaw},
        noise={NoiseKind.DIAGONAL},
        orders={OrderKind.INTEGER},
    )

    form: ClassVar[Form] = Form.RENEWAL
    renews: ClassVar[bool] = True

    def _advance(self, t: float, x: Vector, p: Vector, until: float) -> Vector:
        """Fire every renewal from time `t` up to time `until`, each member in time order.

        Args:
            t:
                The current time.

            x:
                The current state vector.

            p:
                The parameter vector.

            until:
                The time to advance to.

        Returns:
            state:
                The state vector at time `until`.
        """
        clock = self.renewals_
        if until <= t or clock is None:
            return x
        clock.begin(t, x, p)
        return self._bounded(until, clock.fire_until(until, x, p), p)
