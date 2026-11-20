"""Iterate discrete maps, one application per unit of time."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence

import numpy as np

from ...components.laws import UpdateLaw
from ...components.laws._boundary import _BoundaryLaw
from ...components.laws._linear_program import _LinearProgramLaw
from ...components.laws._transfer import _TransferLaw
from ...components.laws._transition import _TransitionLaw
from ...components.spaces import BaseSpace
from ...components.spaces._discrete_time import _DiscreteTime
from ...enums import FormKind, OrderKind
from ...types import Array
from ..backends import SupportSpec
from ..compile import PhaseSpec
from ..problems import ProblemSpec
from ._base import BaseSolver
from ._solver import Solver


class IterationSolver(Solver):
    """A solver that iterates a discrete map, one application per unit of time.

    A stochastic map takes its random draws from the stepper's random number generator at every step, through the draw
    recorder of a run that records or replays its draws. A step may run its update laws in named phases, each reading
    the values the phases before it set, such as a count in the phase `count` and its copy in the phase `copy`.

    Args:
        phases:
            The named phases of a step after the default one, such as `[PhaseSpec('count'), PhaseSpec('copy')]`, in
            the order they run, which the `phase` of every update law names. A phase of period `k` runs every `k`
            steps. Without phases, the phases of the update laws run in the order of their laws.

        random_state:
            The seed or generator of stochastic maps.

        until:
            The end time of the run the solver prepares for, infinite for a run that goes on until it is stopped.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.economics.markets import CobwebInteraction, MarketElement
        >>> from skcomplex.core.engine.solvers import IterationSolver
        >>> from skcomplex.core.engine.solvers import StepSpec
        >>> system = System(
        ...     elements=[MarketElement('m', price=1)],
        ...     interactions=[CobwebInteraction('c', target='m', a=10, b=2, c=1, d=1)],
        ... )
        >>> problem = system.prepare().simulate(until=0).problem
        >>> stepper = IterationSolver(random_state=0).prepare(problem)
        >>> stepper.run(StepSpec(0.0, problem.initial_state, problem.initial_parameters, 1.0)).state_.tolist()
        [4.0]
    """

    def __init__(
        self,
        *,
        phases: Sequence[PhaseSpec] = (),
        random_state: int | np.random.Generator | None = None,
        until: float = math.inf,
    ) -> None:
        super().__init__(random_state=random_state, until=until)
        self.phases = phases

    @classmethod
    def _get_support(cls) -> SupportSpec:
        """Return the times, spaces, laws, noise, and orders the solver solves."""
        return SupportSpec(
            time={_DiscreteTime},
            space={None, BaseSpace},
            laws={UpdateLaw, _TransitionLaw, _LinearProgramLaw, _BoundaryLaw, _TransferLaw},
            orders=frozenset({OrderKind.INTEGER}),
        )

    @classmethod
    def _get_form(cls) -> FormKind:
        """Return the form the solver solves."""
        return FormKind.DISCRETE

    def _prepare(self, context: ProblemSpec | None) -> None:
        """Prepare the stepper for a problem.

        Args:
            context:
                The compiled problem.

        Raises:
            SolverError:
                The problem has another form, delays, or events.
        """
        super()._prepare(context)
        problem = self.problem_
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
        drawn: tuple[object, ...] = (self.rng_,) if self.problem_.stochastic else ()
        if drawn and self.draw_recorder_ is not None:
            drawn = (self.rng_, self.draw_recorder_)
        accelerated = self.problem_.accelerated
        if accelerated is not None and accelerated.iterate is not None and self.draw_recorder_ is None:
            return accelerated.iterate(t, x, p, until, *drawn)
        state = x.copy()
        topology = self.problem_.topology
        for step in range(round(t), math.floor(until + 1e-9)):
            following = update(float(step), state, p, *drawn)
            if topology is not None:
                fired = topology._decide(float(step), state, p, self.rng_)
                following = topology._apply(fired, float(step), following, p, self.rng_)
            state = self._get_bounded(float(step + 1), following, p)
        return state
