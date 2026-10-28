"""Exact stochastic simulation of jump processes by the Gillespie direct method."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from typing import ClassVar

import numpy as np

from ...core.errors import SolverError
from ...core.laws import BoundaryLaw, JumpLaw, TransferLaw
from ...core.spaces import CayleySpace, ContinuousSpace, ContinuousTime, LatticeSpace, MeshSpace, NetworkSpace
from ...core.types import Vector
from ..problems import Form, OrderKind, Problem, SupportSpec
from ._base import BaseSolver
from ._solver import Solver


class GillespieSolver(Solver):
    """Exact stochastic simulation of jump processes by the Gillespie direct method.

    An event that would fire after `until` is discarded, which is exact because waiting times are memoryless.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.engine.compile import compile_system
        >>> from skcomplex.core.system import System
        >>> from skcomplex.models.biology.epidemiology import CompartmentElement, RecoveryInteraction
        >>> from skcomplex.engine.solvers import GillespieSolver
        >>> problem = compile_system(
        ...     System(
        ...         elements=[CompartmentElement('I', n=10), CompartmentElement('R', n=0)],
        ...         interactions=[RecoveryInteraction('rec', infected='I', recovered='R', gamma=1.0, stochastic=True)],
        ...     ),
        ... )
        >>> stepper = GillespieSolver().prepare(problem, np.random.default_rng(0))
        >>> stepper.run(0.0, problem.initial_state, problem.initial_parameters, 100.0).tolist()
        [0.0, 10.0]
    """

    supports: ClassVar[SupportSpec] = SupportSpec(
        time={ContinuousTime},
        space={None, LatticeSpace, NetworkSpace, ContinuousSpace, MeshSpace, CayleySpace},
        laws={JumpLaw, BoundaryLaw, TransferLaw},
        orders={OrderKind.INTEGER},
    )

    form: ClassVar[Form] = Form.JUMP

    def _prepare(self, problem: Problem, rng: np.random.Generator, until: float = math.inf) -> None:
        """Prepare the stepper for a problem.

        Args:
            problem:
                The compiled problem.

            rng:
                The random number generator.

            until:
                The end time of the run.

        Raises:
            SolverError:
                The problem has another form, delays, or events.
        """
        super()._prepare(problem, rng, until)
        BaseSolver._reject_hybrid(problem, type(self).__name__)

    def _advance(self, t: float, x: Vector, p: Vector, until: float) -> Vector:
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
            if topology is not None and channel in topology.channels:
                state = topology.fire(channel, event - channels.offset, t, state, p, rng=self.rng_)
                state = self._bounded(t, state, p)
                continue
            targets = channels.targets[:, event - channels.offset]
            kept = targets >= 0
            np.add.at(state, targets[kept], channels.increments[kept])
            state = self._bounded(t, state, p)
