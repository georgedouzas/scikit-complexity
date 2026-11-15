"""Define the solver that samples states with the Boltzmann weight of an objective."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

import numpy as np

from ...components.laws import ObjectiveLaw
from ...components.spaces import ContinuousSpace, LatticeSpace, MeshSpace, NetworkSpace, StaticTime
from ...enums import FormKind, OrderKind
from ...errors import ValidationError
from ...types import Array
from ..backends import SupportSpec
from ..problems import ProblemSpec
from ._objective import BaseObjectiveSolver


class MetropolisSolver(BaseObjectiveSolver):
    """A solver that samples states with the Boltzmann weight of an objective.

    Every unit of time is one sweep: each target state in turn takes a proposal `x + step * N(0, 1)`, accepted with the
    probability `min(1, exp(-(U_new - U) / T))` at the temperature `T`. The states then visit low values of the
    objective often and high ones rarely, as the positions of a particle in a well at thermal equilibrium, the opinions
    of a population with a social cost, or the parameters of a model drawn from a posterior.

    Read more in the user guide, under Solvers.

    See Also:
        MinimizeSolver: Finds the minimum of the objective, the limit of zero temperature.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.components.interactions import Interaction
        >>> from skcomplex.core.components.laws import ObjectiveLaw
        >>> from skcomplex.core.engine.solvers import MetropolisSolver
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
        ...         self.laws_ = [ObjectiveLaw(x**2 / 2, x)]
        >>> solver = MetropolisSolver(temperature=1.0, step=1.0)
        >>> system = System(elements=[Element('ball', values={'x': 0.0})], interactions=[Well('well')], solver=solver)
        >>> positions = system.prepare().simulate(until=20000, dt=1.0, random_state=0).get_values('ball.x')
        >>> bool(abs(np.var(positions) - 1.0) < 0.1)
        True
    """

    def __init__(
        self,
        *,
        temperature: float = 1.0,
        step: float = 0.1,
        random_state: int | np.random.Generator | None = None,
        until: float = np.inf,
    ) -> None:
        """Store the settings of the solver.

        Args:
            temperature:
                The temperature `T` of the Boltzmann weight, in the units of the objective.

            step:
                The width of a proposal.

            random_state:
                The seed or generator of the proposals and acceptances.

            until:
                The end time of the run the solver prepares for.
        """
        super().__init__(random_state=random_state, until=until)
        self.temperature = temperature
        self.step = step

    @classmethod
    def _get_support(cls) -> SupportSpec:
        """Return the times, spaces, laws, noise, and orders the solver solves."""
        return SupportSpec(
            time={StaticTime},
            space={None, LatticeSpace, NetworkSpace, ContinuousSpace, MeshSpace},
            laws={ObjectiveLaw},
            orders=frozenset({OrderKind.INTEGER}),
        )

    @classmethod
    def _get_form(cls) -> FormKind:
        """Return the form the solver solves."""
        return FormKind.STATIC

    def _prepare(self, context: ProblemSpec | None) -> None:
        """Prepare the solver for a problem, reading the objective.

        Args:
            context:
                The compiled problem.

        Raises:
            ValidationError:
                The temperature or the step is not positive.
        """
        for name in ('temperature', 'step'):
            if not getattr(self, name) > 0:
                error_msg = f'The {name} of MetropolisSolver must be positive, but it is {getattr(self, name)!r}.'
                raise ValidationError(error_msg)
        super()._prepare(context)
        self.evaluate_ = self._get_objective_function(self.problem_)

    def _advance(self, t: float, x: Array, p: Array, until: float) -> Array:
        """Return the state after one sweep of proposals per unit of time until `until`.

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
        energy, _ = self.evaluate_(state, p)
        for _ in range(max(1, math.ceil(until - t - 1e-9))):
            for index in self.problem_.objective_targets:
                proposal = state.copy()
                proposal[index] += self.step * self.rng_.standard_normal()
                proposed, _ = self.evaluate_(proposal, p)
                if self.rng_.random() < math.exp(min(0.0, -(proposed - energy) / self.temperature)):
                    state, energy = proposal, proposed
        return state
