"""Iterate discrete maps, one application per unit of time."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from typing import ClassVar

import numpy as np

from ..base import ComponentKind, DomainKind, Form, LawKind, OrderKind, SpaceKind, Supports, TimeKind
from ..compile import Problem, Vector
from ._hybrid import reject_hybrid
from ._stepper import Stepper, register_solver


class Iteration(Stepper):
    """Iteration of a discrete map, one application per unit of time.

    A stochastic map takes its random draws from the stepper's random number generator at every step.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.compile import compile_system
        >>> from skcomplex.core.system import System
        >>> from skcomplex.models.economics.markets import Cobweb, Market
        >>> from skcomplex.core.solvers import Iteration
        >>> problem = compile_system(
        ...     System(components=[Market('m', price=1)], interactions=[Cobweb('c', target='m', a=10, b=2, c=1, d=1)]),
        ... )
        >>> stepper = Iteration()
        >>> stepper.initialize(problem, np.random.default_rng(0))
        >>> stepper.advance(0.0, problem.initial_state, problem.initial_parameters, 1.0).tolist()
        [4.0]
    """

    supports: ClassVar[Supports] = Supports(
        time={TimeKind.DISCRETE},
        space=set(SpaceKind.registered()),
        laws={LawKind.UPDATE, LawKind.TRANSITION, LawKind.BOUNDARY, LawKind.TRANSFER},
        domains=set(DomainKind) - {DomainKind.HILBERT},
        components={
            ComponentKind.ELEMENT,
            ComponentKind.GROUP,
            ComponentKind.SITE_GROUP,
            ComponentKind.EDGE_GROUP,
            ComponentKind.VARIABLE_GROUP,
            ComponentKind.SYSTEM,
        },
        orders={OrderKind.INTEGER},
    )

    form: ClassVar[Form] = Form.DISCRETE

    def initialize(self, problem: Problem, rng: np.random.Generator) -> None:
        """Prepare the stepper for a problem.

        Args:
            problem:
                The compiled problem.

            rng:
                The random number generator.

        Raises:
            SolverError:
                The problem has another form, delays, or events.
        """
        super().initialize(problem, rng)
        reject_hybrid(problem, type(self).__name__)

    def advance(self, t: float, x: Vector, p: Vector, until: float) -> Vector:
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
                fired = topology.decide(float(step), state, p, self.rng_)
                following = topology.apply(fired, float(step), following, p, self.rng_)
            state = self._bounded(float(step + 1), following, p)
        return state


register_solver('iteration', Iteration, default=True)
