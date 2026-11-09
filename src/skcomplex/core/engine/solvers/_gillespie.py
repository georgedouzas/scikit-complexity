"""Exact stochastic simulation of jump processes by the Gillespie direct method."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np

from ...components.laws import BoundaryLaw, JumpLaw, TransferLaw
from ...components.spaces import CayleySpace, ContinuousSpace, ContinuousTime, LatticeSpace, MeshSpace, NetworkSpace
from ...errors import SolverError, ValidationError
from ...specs import SupportSpec
from ...types import Array, FormKind, OrderKind
from ..problems import ProblemSpec
from ..problems._draw_recorder import DrawRecorder
from ._base import BaseSolver
from ._solver import Solver


class GillespieSolver(Solver):
    """Exact stochastic simulation of jump processes by the Gillespie direct method.

    An event that would fire after `until` is discarded, which is exact because waiting times are memoryless. A run
    that records its draws keeps the time and the law of every firing, and a run that replays them fires the recorded
    laws at the recorded times and sums the log-likelihood of the path under its own propensities.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.engine.runtime import simulate
        >>> from skcomplex.core.components.systems import System
        >>> from skcomplex.core.components.elements import NonnegativeElement
        >>> from skcomplex.core.components.interactions import ReactionInteraction
        >>> from skcomplex.core.engine.solvers import GillespieSolver
        >>> from skcomplex.core.specs import StepSpec
        >>> system = System(
        ...     elements=[NonnegativeElement('I', x=10), NonnegativeElement('R', x=0)],
        ...     interactions=[
        ...         ReactionInteraction('rec', reactants={('I', 'x'): 1}, products={('R', 'x'): 1}, k=1.0, mode='jump')
        ...     ],
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

    def _rates(self, t: float, state: Array, p: Array) -> Array:
        """Return the propensities at a state, after checking them.

        Args:
            t:
                The time.

            state:
                The state vector.

            p:
                The parameter vector.

        Returns:
            rates:
                The propensity of every jump law.

        Raises:
            SolverError:
                A propensity is negative or not finite.
        """
        propensities = self.problem_.propensities
        rates = np.zeros(0) if propensities is None else propensities(t, state, p)
        if np.any(rates < 0) or not np.all(np.isfinite(rates)):
            error_msg = f'Propensities {rates.tolist()} at t = {t} must be finite and non-negative.'
            raise SolverError(error_msg)
        return rates

    def _fire(self, t: float, state: Array, p: Array, event: int) -> Array:
        """Return the state after one firing of a jump law."""
        jumps = self.problem_.jumps
        offsets = np.array([channels.offset for channels in jumps])
        channel = int(np.searchsorted(offsets, event, side='right')) - 1
        channels = jumps[channel]
        topology = self.problem_.topology
        if topology is not None and channel in (topology.channels or {}):
            state = topology._fire(channel, event - channels.offset, t, state, p, rng=self.rng_)
            return self._bounded(t, state, p)
        targets = channels.targets[:, event - channels.offset]
        kept = targets >= 0
        increments = channels.increments_at(t, state, p, event - channels.offset)
        np.add.at(state, targets[kept], increments[kept])
        return self._bounded(t, state, p)

    def _replay(self, t: float, x: Array, p: Array, until: float, recorder: DrawRecorder) -> Array:
        """Fire the recorded laws at their recorded times from time `t` until `until`.

        The log-probability of the path is the sum of the log propensities of the firings minus the integral of the
        total propensity, which is exact while the propensities stay constant between firings.
        """
        state = x.copy()
        log_probability = 0.0
        with np.errstate(divide='ignore'):
            for time, event in recorder.firings(t, until):
                rates = self._rates(t, state, p)
                if event >= rates.size:
                    error_msg = f'The replayed trajectory fires the jump law {event}, but the system has {rates.size}.'
                    raise ValidationError(error_msg)
                log_probability += float(np.log(rates[event]) - rates.sum() * (time - t))
                t = time
                state = self._fire(t, state, p, event)
            log_probability -= float(self._rates(t, state, p).sum() * (until - t))
        recorder.add_log_probability(log_probability)
        return state

    def _advance(self, t: float, x: Array, p: Array, until: float) -> Array:
        """Fire random events from time `t` until the next event falls after `until`.

        A run that records keeps the time and the law of every firing with its log-probability, and a run that
        replays fires the recorded laws at the recorded times.

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
        if self.problem_.propensities is None:
            return x
        recorder = self.draw_recorder_
        if recorder is not None and recorder.replaying:
            return self._replay(t, x, p, until, recorder)
        accelerated = self.problem_.accelerated
        if recorder is None and accelerated is not None and accelerated.gillespie is not None:
            try:
                return accelerated.gillespie(t, x, p, until, self.rng_)
            except ValueError as error:
                error_msg = f'Propensities at t = {t} must be finite and non-negative: {error}'
                raise SolverError(error_msg) from error
        state = x.copy()
        while True:
            rates = self._rates(t, state, p)
            total = float(rates.sum())
            if total == 0:
                return state
            waiting = self.rng_.exponential(1 / total)
            if t + waiting > until:
                if recorder is not None:
                    recorder.add_log_probability(-total * (until - t))
                return state
            t += waiting
            event = min(
                int(np.searchsorted(np.cumsum(rates), self.rng_.uniform(0, total), side='right')),
                len(rates) - 1,
            )
            if recorder is not None:
                recorder.record_firing(t, event, float(np.log(rates[event]) - total * waiting))
            state = self._fire(t, state, p, event)
