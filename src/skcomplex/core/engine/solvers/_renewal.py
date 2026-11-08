"""Exact stepper of renewal processes, which jumps from renewal to renewal without a time grid."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ...components.laws import BoundaryLaw, EquationLaw, FluxLaw, JumpLaw, NoiseLaw, RateLaw
from ...components.spaces import CayleySpace, ContinuousSpace, ContinuousTime, LatticeSpace, MeshSpace, NetworkSpace
from ...specs import SupportSpec
from ...types import Array, FormKind, NoiseKind, OrderKind
from ._solver import Solver


class RenewalSolver(Solver):
    """Exact stepper of renewal processes, which jumps from renewal to renewal without a time grid.

    Every member renews on its own clock, and a frame holds the states the last renewal at or before its time set.

    Examples:
        >>> from typing import Annotated
        >>> import numpy as np
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.core.specs import StateSpec
        >>> from skcomplex.core.engine.runtime import simulate
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.components.laws import JumpLaw
        >>> from skcomplex.core.components.laws.utils import draw
        >>> from skcomplex.core.components.system import System
        >>> from skcomplex.core.engine.solvers import RenewalSolver
        >>> from skcomplex.core.specs import StepSpec
        >>> class CounterElement(Element):
        ...     def __init__(self, label, n: Annotated[float, StateSpec()] = 0.0):
        ...         super().__init__(label)
        ...         self.n = n
        ...
        ...     def _get_laws(self, s):
        ...         return [JumpLaw(waiting=draw('wait', 'expon'), reset={s.n: s.n + 1})]
        >>> system = System(elements=[GroupElement('counters', element=CounterElement, size=3)])
        >>> problem = simulate(system, until=0).problem
        >>> stepper = RenewalSolver(random_state=0).prepare(problem)
        >>> state = stepper.run(StepSpec(0.0, problem.initial_state, problem.initial_parameters, 100.0)).state_
        >>> bool(np.all(state[:3] > 50))
        True
    """

    @classmethod
    def _get_support(cls) -> SupportSpec:
        """Return the times, spaces, laws, noise, and orders the solver solves."""
        return SupportSpec(
            time={ContinuousTime},
            space={None, LatticeSpace, NetworkSpace, ContinuousSpace, MeshSpace, CayleySpace},
            laws={JumpLaw, RateLaw, FluxLaw, NoiseLaw, EquationLaw, BoundaryLaw},
            noise=frozenset({NoiseKind.DIAGONAL}),
            orders=frozenset({OrderKind.INTEGER}),
        )

    @classmethod
    def _get_form(cls) -> FormKind:
        """Return the form the solver solves."""
        return FormKind.RENEWAL

    @classmethod
    def _is_renewing(cls) -> bool:
        """Return whether the solver fires renewal laws."""
        return True

    def _advance(self, t: float, x: Array, p: Array, until: float) -> Array:
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
