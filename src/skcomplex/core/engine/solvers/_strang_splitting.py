"""Alternate transport and collision steps of a field.

The scheme has second-order accuracy in the time step.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from dataclasses import replace

import numpy as np
from sklearn.base import clone

from ...components.laws import BilinearLaw, FluxLaw, RateLaw
from ...components.spaces import ContinuousTime, LatticeSpace, PhaseSpace
from ...errors import SolverError
from ...specs import RunStateSpec, SupportSpec
from ...types import Array, FormKind, OrderKind
from ..problems import ProblemSpec
from ..problems._steps import step_count
from ._rk4 import RK4Solver
from ._solver import Solver


class StrangSplittingSolver(Solver):
    """Strang splitting of half transport steps around a full collision step.

    A half transport step, a full collision step, and a half transport step follow each other. The transport solver
    advances the flux differences of the finite-volume fields and the collision solver the source rates, which the
    problem keeps apart.

    Args:
        transport:
            The solver of the flux differences, or `None` for `RK4Solver`.

        collision:
            The solver of the source rates, or `None` for `RK4Solver`.

        dt:
            The largest internal time step, the length of one split step.

        random_state:
            The seed or generator of stochastic solvers.

        until:
            The end time of the run the solver prepares for, infinite for a run that goes on until it is stopped.

    Attributes:
        transport_:
            The prepared solver of the flux differences.

        collision_:
            The prepared solver of the source rates.

    Examples:
        >>> import numpy as np
        >>> from typing import Annotated
        >>> from skcomplex.core.components.elements import Element, FieldElement
        >>> from skcomplex.core.specs import StateSpec
        >>> class AmountElement(Element):
        ...     def __init__(self, label, visual=None, *, x: Annotated[float, StateSpec()] = 0.0):
        ...         super().__init__(label, visual=visual)
        ...         self.x = x
        >>> from skcomplex.core.components.interactions import ConservationInteraction, RateInteraction
        >>> from skcomplex.core.components.laws.utils import coordinate, maximum_entropy
        >>> from skcomplex.core.components.spaces import LatticeSpace, PhaseSpace
        >>> from skcomplex.core.components.systems import CompositeSystem
        >>> from skcomplex.core.engine.solvers import RK4Solver, SemiLagrangianSolver, StrangSplittingSolver
        >>> from skcomplex.core.specs import AxisSpec
        >>> from skcomplex.core.engine.runtime import simulate
        >>> space = PhaseSpace(LatticeSpace((8,)), [AxisSpec('v', 16, -6.0, 6.0)]).prepare()
        >>> start = np.outer(2 + np.sin(2 * np.pi * np.arange(8) / 8), np.exp(-space.coordinates_['v'] ** 2))
        >>> gas = FieldElement(
        ...     'gas', element=AmountElement, space=space, values={'x': start}, discretization='finite-volume'
        ... )
        >>> stream = ConservationInteraction(
        ...     's', target='gas', state='x', flux=lambda s: [coordinate('v') * s.target.x, 0]
        ... )
        >>> def relaxation(s):
        ...     return (maximum_entropy(s.target.x, axes=('v',)) - s.target.x) / 0.5
        >>> relax = RateInteraction('r', target='gas', state='x', rate=relaxation)
        >>> system = CompositeSystem(elements=[gas], interactions=[stream, relax])
        >>> solver = StrangSplittingSolver(SemiLagrangianSolver(dt=0.1), RK4Solver(dt=0.1), dt=0.1)
        >>> final = simulate(system, until=0.2, solver=solver).values('gas.x')[-1]
        >>> abs(float(final.sum() / start.sum()) - 1.0) < 1e-12
        True
    """

    @classmethod
    def _get_support(cls) -> SupportSpec:
        """Return the times, spaces, laws, noise, and orders the solver solves."""
        return SupportSpec(
            time={ContinuousTime},
            space={PhaseSpace, LatticeSpace},
            laws={FluxLaw, RateLaw, BilinearLaw},
            orders=frozenset({OrderKind.INTEGER}),
        )

    @classmethod
    def _get_form(cls) -> FormKind:
        """Return the form the solver solves."""
        return FormKind.ODE

    @classmethod
    def _is_implicit(cls) -> bool:
        """Return whether explicit stability limits do not apply.

        The inner solvers answer for it.
        """
        return True

    def __init__(
        self,
        transport: Solver | None = None,
        collision: Solver | None = None,
        dt: float = 0.01,
        *,
        random_state: int | np.random.Generator | None = None,
        until: float = math.inf,
    ) -> None:
        self.transport = transport
        self.collision = collision
        self.dt = dt
        self.random_state = random_state
        self.until = until

    def _get_flux_differences(self, t: float, x: Array, p: Array) -> Array:
        """Return the rates of the flux differences of the finite-volume fields.

        They are zero for the other states.
        """
        rates = np.zeros(x.size)
        for volume in self.problem_.volumes:
            rates = rates + np.asarray(volume.run(RunStateSpec(t, x, p)).values_, dtype=float)
        return rates

    def _get_inner(self, solver: Solver | None, law: type, dt: float, problem: ProblemSpec) -> Solver:
        """Return an inner solver prepared on a part of the problem.

        An error is raised for a law it does not advance.
        """
        inner = RK4Solver(dt=dt) if solver is None else clone(solver)
        if inner._get_support().get_missing('laws', {law}):
            error_msg = f'StrangSplittingSolver advances {law.__name__} with {type(inner).__name__}, which lacks it.'
            raise SolverError(error_msg)
        return inner.prepare(problem)

    def _prepare(self, context: ProblemSpec | None) -> None:
        """Prepare the two inner solvers on the transport and the collision parts.

        The parts are those of the problem.

        Args:
            context:
                The compiled problem.

        Raises:
            SolverError:
                The problem has no flux or no source rates, or an inner solver does not advance its part.
        """
        super()._prepare(context)
        problem = self.problem_
        if not problem.volumes or problem.sources is None:
            error_msg = 'StrangSplittingSolver splits flux differences from source rates, but the problem has no flux.'
            raise SolverError(error_msg)
        plain = replace(problem, accelerated=None, jacobian=None, sources=None)
        transport = replace(plain, drift=self._get_flux_differences)
        collision = replace(plain, drift=problem.sources, volumes=(), linear=())
        self.transport_ = self._get_inner(self.transport, FluxLaw, self.dt / 2, transport)
        self.collision_ = self._get_inner(self.collision, RateLaw, self.dt, collision)

    def _advance(self, t: float, x: Array, p: Array, until: float) -> Array:
        """Advance from time `t` to time `until` in equal split steps no longer than `dt`.

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
        steps = step_count(t, until, self.dt)
        h = (until - t) / steps
        state = x
        for index in range(steps):
            time = t + index * h
            state = self.transport_._advance(time, state, p, time + h / 2)
            state = self.collision_._advance(time, state, p, time + h)
            state = self.transport_._advance(time + h / 2, state, p, time + h)
        return state
