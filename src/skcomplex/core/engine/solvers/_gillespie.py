"""Exact stochastic simulation of jump processes by the Gillespie direct method."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np

from ...components.laws import BoundaryLaw, JumpLaw, TransferLaw
from ...components.spaces import CayleySpace, ContinuousSpace, ContinuousTime, LatticeSpace, MeshSpace, NetworkSpace
from ...errors import SolverError
from ...specs import SupportSpec
from ...types import Array, FormKind, OrderKind
from ..problems import ProblemSpec
from ._base import BaseSolver
from ._solver import Solver


class GillespieSolver(Solver):
    """Exact stochastic simulation of jump processes by the Gillespie direct method.

    An event that would fire after `until` is discarded, which is exact because waiting times are memoryless.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.engine.runtime import simulate
        >>> from skcomplex.core.components.system import System
        >>> from skcomplex.core.components.elements import CompartmentElement
        >>> from skcomplex.systems.biology.epidemiology import RecoveryInteraction
        >>> from skcomplex.core.engine.solvers import GillespieSolver
        >>> from skcomplex.core.specs import StepSpec
        >>> system = System(
        ...     elements=[CompartmentElement('I', n=10), CompartmentElement('R', n=0)],
        ...     interactions=[RecoveryInteraction('rec', infected='I', recovered='R', gamma=1.0, stochastic=True)],
        ... )
        >>> problem = simulate(system, until=0).problem
        >>> stepper = GillespieSolver(random_state=0).prepare(problem)
        >>> stepper.run(StepSpec(0.0, problem.initial_state, problem.initial_parameters, 100.0)).state_.tolist()
        [0.0, 10.0]
    """

    @classmethod
    def _get_support(cls) -> SupportSpec:
        """Return the times, spaces, laws, noise, and orders the solver solves."""
        return SupportSpec(
            time={ContinuousTime},
            space={None, LatticeSpace, NetworkSpace, ContinuousSpace, MeshSpace, CayleySpace},
            laws={JumpLaw, BoundaryLaw, TransferLaw},
            orders=frozenset({OrderKind.INTEGER}),
        )

    @classmethod
    def _get_form(cls) -> FormKind:
        """Return the form the solver solves."""
        return FormKind.JUMP

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
        """Fire random events from time `t` until the next event falls after `until`.

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

        Raises:
            SolverError:
                A propensity is negative or not finite.
        """
        propensities, jumps = self.problem_.propensities, self.problem_.jumps
        if propensities is None:
            return x
        accelerated = self.problem_.accelerated
        if accelerated is not None and accelerated.gillespie is not None:
            try:
                return accelerated.gillespie(t, x, p, until, self.rng_)
            except ValueError as error:
                error_msg = f'Propensities at t = {t} must be finite and non-negative: {error}'
                raise SolverError(error_msg) from error
        offsets = np.array([channels.offset for channels in jumps])
        state = x.copy()
        while True:
            rates = propensities(t, state, p)
            if np.any(rates < 0) or not np.all(np.isfinite(rates)):
                error_msg = f'Propensities {rates.tolist()} at t = {t} must be finite and non-negative.'
                raise SolverError(error_msg)
            total = float(rates.sum())
            if total == 0:
                return state
            t += self.rng_.exponential(1 / total)
            if t > until:
                return state
            event = min(
                int(np.searchsorted(np.cumsum(rates), self.rng_.uniform(0, total), side='right')),
                len(rates) - 1,
            )
            channel = int(np.searchsorted(offsets, event, side='right')) - 1
            channels = jumps[channel]
            topology = self.problem_.topology
            if topology is not None and channel in (topology.channels or {}):
                state = topology._fire(channel, event - channels.offset, t, state, p, rng=self.rng_)
                state = self._bounded(t, state, p)
                continue
            targets = channels.targets[:, event - channels.offset]
            kept = targets >= 0
            increments = channels.increments_at(t, state, p, event - channels.offset)
            np.add.at(state, targets[kept], increments[kept])
            state = self._bounded(t, state, p)
