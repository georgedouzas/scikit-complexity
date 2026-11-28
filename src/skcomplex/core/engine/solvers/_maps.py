"""Iterate discrete maps, one application per unit of time."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence

import numpy as np

from ...components.laws import UpdateLaw
from ...components.laws._boundary import _BoundaryLaw
from ...components.laws._linear_program import _LinearProgramLaw
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
    """A solver that applies a discrete map once per unit of time.

    Every step applies the update laws once, such as `x ← r x (1 - x)` of the logistic map, which takes 0.2 to 0.592 and
    then 0.894 at `r = 3.7`. A map that draws random numbers takes them from the solver's generator at every step, and a
    step may run its laws in named phases, each reading what the phases before it set, such as a count in the phase
    `count` and its copy in the phase `copy`. An update law with an activity records its runs of activity as events,
    read at the start of the run and at the end of every step. It runs the logistic map of a population, the cobweb of a market, and the
    layers of a neural network.

    Read more in the user guide, under Solvers.

    Args:
        phases:
            The named phases of a step after the default one, such as `[PhaseSpec('count'), PhaseSpec('copy')]`, in
            the order they run, which the `phase` of every update law names. A phase of period `k` runs every `k`
            steps. Without phases, the phases of the update laws run in the order of their laws.

        random_state:
            The seed or generator of stochastic maps.

        until:
            The end time of the run the solver prepares for, infinite for a run that goes on until it is stopped.

    See Also:
        LatticeSolver: Applies the laws of a lattice once per step.
        MesaSolver: Runs agents one step per unit of time.

    Examples:
        >>> from skcomplex.core.engine.solvers import IterationSolver
        >>> from skcomplex.systems.mathematics.dynamical_systems import LogisticMapSystem
        >>> system = LogisticMapSystem(solver=IterationSolver()).prepare()
        >>> system.simulate(until=2).get_values('map.x').round(3).tolist()
        [0.2, 0.592, 0.894]
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
            laws={UpdateLaw, _LinearProgramLaw, _BoundaryLaw},
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
        compiled = accelerated is not None and accelerated.iterate is not None
        if accelerated is not None and compiled and self.draw_recorder_ is None and not self.problem_.activities:
            return accelerated.iterate(t, x, p, until, *drawn)
        state = x.copy()
        topology = self.problem_.topology
        self._record_activity(t, state, p, first=True)
        for step in range(round(t), math.floor(until + 1e-9)):
            following = update(float(step), state, p, *drawn)
            if topology is not None:
                fired = topology._decide(float(step), state, p, self.rng_)
                following = topology._apply(fired, float(step), following, p, self.rng_)
            state = self._get_bounded(float(step + 1), following, p)
            self._record_activity(float(step + 1), state, p)
        return state
