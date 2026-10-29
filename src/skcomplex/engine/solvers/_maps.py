"""Iterate discrete maps, one application per unit of time."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

from ...core.laws import BoundaryLaw, EmissionLaw, TransferLaw, TransitionLaw, UpdateLaw
from ...core.spaces import CayleySpace, ContinuousSpace, DiscreteTime, LatticeSpace, MeshSpace, NetworkSpace
from ...specs import SupportSpec
from ...types import Array, FormKind, OrderKind
from ..problems import ProblemSpec
from ._base import BaseSolver
from ._solver import Solver


class IterationSolver(Solver):
    """Iteration of a discrete map, one application per unit of time.

    A stochastic map takes its random draws from the stepper's random number generator at every step.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.engine.compile import compile_system
        >>> from skcomplex.core.system import System
        >>> from skcomplex.models.economics.markets import CobwebInteraction, MarketElement
        >>> from skcomplex.engine.solvers import IterationSolver
        >>> from skcomplex.specs import StepSpec
        >>> problem = compile_system(
        ...     System(
        ...         elements=[MarketElement('m', price=1)],
        ...         interactions=[CobwebInteraction('c', target='m', a=10, b=2, c=1, d=1)],
        ...     ),
        ... )
        >>> stepper = IterationSolver(random_state=0).check().prepare(problem)
        >>> stepper.run(StepSpec(0.0, problem.initial_state, problem.initial_parameters, 1.0)).tolist()
        [4.0]
    """

    @classmethod
    def _get_support(cls) -> SupportSpec:
        """Return the times, spaces, laws, noise, and orders the solver solves."""
        return SupportSpec(
            time={DiscreteTime},
            space={None, LatticeSpace, NetworkSpace, ContinuousSpace, MeshSpace, CayleySpace},
            laws={UpdateLaw, TransitionLaw, EmissionLaw, BoundaryLaw, TransferLaw},
            orders={OrderKind.INTEGER},
        )

    @classmethod
    def _get_form(cls) -> FormKind:
        """Return the form the solver solves."""
        return FormKind.DISCRETE

    def _prepare(self, problem: ProblemSpec) -> None:
        """Prepare the stepper for a problem.

        Args:
            problem:
                The compiled problem.

        Raises:
            SolverError:
                The problem has another form, delays, or events.
        """
        super()._prepare(problem)
        BaseSolver._reject_hybrid(problem, type(self).__name__)

    def _advance(self, t: float, x: Array, p: Array, until: float) -> Array:
        """Apply the map once per integer time step from `t` up to `until`.

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
                The state vector after the last whole step.
        """
        update = self.problem_.update
        if update is None:
            return x
        drawn = (self.rng_,) if self.problem_.stochastic else ()
        accelerated = self.problem_.accelerated
        if accelerated is not None and accelerated.iterate is not None:
            return accelerated.iterate(t, x, p, until, *drawn)
        state = x.copy()
        topology = self.problem_.topology
        for step in range(round(t), math.floor(until + 1e-9)):
            following = update(float(step), state, p, *drawn)
            if topology is not None:
                fired = topology._decide(float(step), state, p, self.rng_)
                following = topology._apply(fired, float(step), following, p, self.rng_)
            state = self._bounded(float(step + 1), following, p)
        return state
