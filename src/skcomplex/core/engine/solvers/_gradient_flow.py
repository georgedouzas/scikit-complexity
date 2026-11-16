"""Define the solver that follows the steepest descent of an objective."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

import numpy as np

from ...components.laws import ObjectiveLaw
from ...components.spaces import ContinuousSpace, LatticeSpace, NetworkSpace, StaticTime
from ...enums import FormKind, OrderKind
from ...errors import ValidationError
from ...types import Array
from ..backends import SupportSpec
from ..problems import ProblemSpec
from ._objective import BaseObjectiveSolver


class GradientFlowSolver(BaseObjectiveSolver):
    """A solver that follows the steepest descent of an objective over time.

    It integrates the gradient flow `x' = -grad U(x)` of the objective `U` over its target states with explicit steps
    of size `dt`, which is gradient descent with the learning rate `dt`. The trajectory shows the path to a minimum,
    such as a ball rolling into a well, a network of springs relaxing, or the weights of a model as its loss falls.

    Read more in the user guide, under Solvers.

    See Also:
        MinimizeSolver: Finds the minimum of the objective in one frame.

    Examples:
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.components.interactions import Interaction
        >>> from skcomplex.core.components.laws import ObjectiveLaw
        >>> from skcomplex.core.engine.solvers import GradientFlowSolver
        >>> from skcomplex.core.specs import Role
        >>> from skcomplex.core.systems import System
        >>> class Well(Interaction):
        ...     def __init__(self, label, *, target: Role = 'ball'):
        ...         super().__init__(label)
        ...         self.target = target
        ...
        ...     def _prepare(self, context):
        ...         super()._prepare(context)
        ...         x = context.target.x
        ...         self.laws_ = [ObjectiveLaw((x - 2) ** 2, x)]
        >>> ball = Element('ball', values={'x': 0.0})
        >>> system = System(elements=[ball], interactions=[Well('well')], solver=GradientFlowSolver(dt=0.1)).prepare()
        >>> round(float(system.simulate(until=1.0, dt=1.0).get_values('ball.x')[-1]), 3)
        1.785
    """

    def __init__(
        self,
        *,
        dt: float = 0.01,
        random_state: int | np.random.Generator | None = None,
        until: float = np.inf,
    ) -> None:
        """Store the settings of the solver.

        Args:
            dt:
                The size of a step of the flow, the learning rate of gradient descent.

            random_state:
                The seed or generator of the solver, which draws nothing.

            until:
                The end time of the run the solver prepares for.
        """
        super().__init__(random_state=random_state, until=until)
        self.dt = dt

    @classmethod
    def _get_support(cls) -> SupportSpec:
        """Return the times, spaces, laws, noise, and orders the solver solves."""
        return SupportSpec(
            time={StaticTime},
            space={None, LatticeSpace, NetworkSpace, ContinuousSpace},
            laws={ObjectiveLaw},
            orders=frozenset({OrderKind.INTEGER}),
        )

    @classmethod
    def _get_form(cls) -> FormKind:
        """Return the form the solver solves."""
        return FormKind.STATIC

    def _prepare(self, context: ProblemSpec | None) -> None:
        """Prepare the solver for a problem, reading the objective and its gradient.

        Args:
            context:
                The compiled problem.

        Raises:
            ValidationError:
                The step is not positive.
        """
        if not self.dt > 0:
            error_msg = f'The step dt of GradientFlowSolver must be positive, but it is {self.dt!r}.'
            raise ValidationError(error_msg)
        super()._prepare(context)
        self.evaluate_ = self._get_objective_function(self.problem_)

    def _advance(self, t: float, x: Array, p: Array, until: float) -> Array:
        """Return the state after descending the objective from `t` to `until`.

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
                The state vector at `until`.
        """
        state = np.asarray(x, dtype=float).copy()
        targets = self.problem_.objective_targets
        steps = max(1, math.ceil((until - t) / self.dt - 1e-9))
        size = (until - t) / steps
        for _ in range(steps):
            _, gradient = self.evaluate_(state, p)
            state[targets] -= size * gradient
        return state
