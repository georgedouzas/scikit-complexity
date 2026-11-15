"""Minimise the objective of a static system once."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable

import numpy as np
from sage.all__sagemath_symbolics import SR
from scipy.optimize import minimize

from ...components.laws import ConstraintLaw, ObjectiveLaw
from ...components.spaces import ContinuousSpace, LatticeSpace, MeshSpace, NetworkSpace, StaticTime
from ...enums import FormKind, OrderKind
from ...errors import SolverError
from ...types import Array
from ..backends import SupportSpec
from ..problems import ProblemSpec
from ._objective import BaseObjectiveSolver


class MinimizeSolver(BaseObjectiveSolver):
    """A solver that minimises the objective of a static system once.

    It minimises the sum of the objective laws over their target states with `scipy.optimize.minimize`, using the exact
    gradient of the objective. The constraint laws of the system hold at the minimum as equality constraints, which
    the SLSQP method enforces, and the states they determine vary with the targets. It finds the rest state of a
    loaded structure, the allocation that maximises a utility under a budget, and the weights of a model at its least
    loss.

    Read more in the user guide, under Solvers.

    Attributes:
        initial_state_:
            The state vector at the minimum, the one frame of the run.

    See Also:
        RootSolver: Solves the constraint laws of a static system that has no objective.

    Examples:
        >>> from typing import Annotated
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.components.interactions import Interaction
        >>> from skcomplex.core.components.laws import ObjectiveLaw
        >>> from skcomplex.core.specs import RoleSpec, StateSpec
        >>> from skcomplex.core.systems import System
        >>> class PointElement(Element):
        ...     def __init__(self, label, *, x: Annotated[float, StateSpec()] = 0.0):
        ...         super().__init__(label)
        ...         self.x = x
        >>> class WellInteraction(Interaction):
        ...     def __init__(self, label, *, target: Annotated[str, RoleSpec()] = 'point'):
        ...         super().__init__(label)
        ...         self.target = target
        ...
        ...     def _get_laws(self, s):
        ...         return [ObjectiveLaw((s.target.x - 2) ** 2, s.target.x)]
        >>> system = System(elements=[PointElement('point')], interactions=[WellInteraction('well')]).prepare()
        >>> round(float(system.simulate(until=1.0).get_values('point.x')[0]), 6)
        2.0
    """

    def __init__(
        self,
        *,
        method: str | None = None,
        tolerance: float = 1e-9,
        random_state: int | np.random.Generator | None = None,
        until: float = np.inf,
    ) -> None:
        """Store the settings of the solver.

        Args:
            method:
                The method of `scipy.optimize.minimize`, or `None` for SLSQP with constraints and BFGS without.

            tolerance:
                The tolerance of the minimisation, and the largest residual of a constraint at the minimum.

            random_state:
                The seed or generator of the solver, which draws nothing.

            until:
                The end time of the run the solver prepares for, which a static system does not read.
        """
        super().__init__(random_state=random_state, until=until)
        self.method = method
        self.tolerance = tolerance

    @classmethod
    def _get_support(cls) -> SupportSpec:
        """Return the times, spaces, laws, noise, and orders the solver solves."""
        return SupportSpec(
            time={StaticTime},
            space={None, LatticeSpace, NetworkSpace, ContinuousSpace, MeshSpace},
            laws={ObjectiveLaw, ConstraintLaw},
            orders=frozenset({OrderKind.INTEGER}),
        )

    @classmethod
    def _get_form(cls) -> FormKind:
        """Return the form the solver solves."""
        return FormKind.STATIC

    def _get_constraints(
        self,
        problem: ProblemSpec,
        variables: Array,
        p: Array,
        x: Array,
    ) -> list[Callable[[Array], Array]]:
        """Return the residuals of the constraint laws that hold numerically."""
        if not problem.unknowns.size:
            return []
        drift = self._get_drift()

        def residuals(values: Array) -> Array:
            state = x.copy()
            state[variables] = values
            return np.asarray(drift(0.0, state, p), dtype=float)[problem.unknowns]

        return [residuals]

    @staticmethod
    def _get_determined(problem: ProblemSpec) -> Array:
        """Return the positions of the states that explicit constraints determine."""
        if problem.update is None:
            return np.zeros(0, dtype=np.intp)
        positions = [
            entry.start + column
            for entry in problem.states.entries
            for column, name in enumerate(entry.generic_names)
            if name in problem.equations
            and not (SR(problem.equations[name]) - SR(problem.equation_symbols[name])).is_trivial_zero()
        ]
        return np.setdiff1d(np.asarray(positions, dtype=np.intp), problem.unknowns)

    def _get_completed(self, state: Array, p: Array) -> Array:
        """Return a state vector with the explicitly constrained states set."""
        update = self.problem_.update
        return state if update is None else np.asarray(update(0.0, state, p), dtype=float)

    def _get_minimum(self, x: Array, p: Array) -> Array:
        """Return the state vector at the minimum of the objective."""
        problem = self.problem_
        if not isinstance(problem, ProblemSpec):
            error_msg = f'{type(self).__name__} solves a static problem, not a {type(problem).__name__}.'
            raise SolverError(error_msg)
        evaluate = self._get_objective_function(problem)
        targets = problem.objective_targets
        determined = self._get_determined(problem)
        variables = np.setdiff1d(np.unique(np.concatenate([targets, problem.unknowns])), determined)
        position = {int(index): order for order, index in enumerate(targets)}
        constraints = self._get_constraints(problem, variables, p, x)
        exact = problem.update is None

        def objective(values: Array) -> tuple[float, Array]:
            state = x.copy()
            state[variables] = values
            value, gradient = evaluate(self._get_completed(state, p), p)
            full = np.array([gradient[position[int(index)]] if int(index) in position else 0.0 for index in variables])
            return value, full

        method = self.method or ('SLSQP' if constraints or not exact else 'BFGS')
        result = minimize(
            objective if exact else (lambda values: objective(values)[0]),
            x[variables],
            jac=exact,
            method=method,
            constraints=[{'type': 'eq', 'fun': residuals} for residuals in constraints],
            tol=self.tolerance,
        )
        residual = max((float(np.max(np.abs(residuals(result.x)))) for residuals in constraints), default=0.0)
        if not result.success and not residual <= np.sqrt(self.tolerance):
            error_msg = (
                f'The minimisation of the objective did not converge: {result.message} Give an initial state closer to '
                'the minimum, or check that the objective is bounded below.'
            )
            raise SolverError(error_msg)
        solution = x.copy()
        solution[variables] = result.x
        return self._get_completed(solution, p)

    def _prepare(self, context: ProblemSpec | None) -> None:
        """Prepare the solver for a problem and minimise its objective.

        Args:
            context:
                The compiled problem.

        Raises:
            SolverError:
                The problem has another form or no objective, or its minimisation does not converge.
        """
        super()._prepare(context)
        self.initial_state_ = self._get_minimum(self.problem_.initial_state, self.problem_.initial_parameters)

    def _advance(self, t: float, x: Array, p: Array, until: float) -> Array:
        """Return the state unchanged.

        The state was minimised for the one frame of the run.

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
                The state vector.
        """
        return x
