"""Define the solver that samples states with the Boltzmann weight of an objective."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from ...components.laws import ObjectiveLaw
from ...components.spaces import BaseNeighborhood, ContinuousSpace, LatticeSpace, NetworkSpace
from ...components.spaces._static_time import _StaticTime
from ...enums import FormKind, OrderKind
from ...errors import ValidationError
from ...types import Array
from ..backends import SupportSpec
from ..problems import ProblemSpec
from ._monte_carlo import MonteCarloSolver
from ._objective import BaseObjectiveSolver


class MetropolisSolver(BaseObjectiveSolver):
    """A solver that samples states by the Boltzmann weight of an objective.

    Every unit of time is one sweep in which each target state takes a proposal `x + step * N(0, 1)`, kept with the
    chance `min(1, exp(-(U_new - U) / T))` at the temperature `T`, so states with low objective come up often and high
    ones rarely. With `moves` it samples the members of a group on a lattice, a network, or the edges of a cell complex
    one by one, such as the spins of a magnet flipped by `moves={'spins.spin': 'flip'}`, where the objective is the
    local energy of a member over the temperature. Each move, `'flip'`, `'random-value'`, `'rotate'`, `'gaussian'`,
    `'cluster'`, or a function `move(values, random)`, is kept by `rule`, `'metropolis'`, `'glauber'`, or `'heat-bath'`,
    in the order of `schedule`. It samples a particle in a well, the spins of a magnet, and the parameters of a model
    from a posterior.

    Read more in the user guide, under Solvers.

    See Also:
        MinimizeSolver: Finds the minimum of the objective, the limit of zero temperature.

    Examples:
        >>> from skcomplex.core.engine.solvers import MetropolisSolver
        >>> from skcomplex.systems.physics.statistical_mechanics import IsingSystem
        >>> solver = MetropolisSolver(moves={'spins.spin': 'flip'})
        >>> system = IsingSystem(size=8, temperature=1.0, solver=solver).prepare()
        >>> abs(float(system.simulate(until=50).get_values('spins.spin')[-1].mean()))
        1.0
    """

    def __init__(
        self,
        *,
        temperature: float = 1.0,
        step: float = 0.1,
        moves: Mapping[str, str | Callable[..., object]] | None = None,
        rule: str = 'metropolis',
        schedule: str = 'auto',
        neighborhood: BaseNeighborhood | None = None,
        values: Mapping[str, Sequence[float]] | None = None,
        width: float = 1.0,
        random_state: int | np.random.Generator | None = None,
        until: float = np.inf,
    ) -> None:
        """Store the settings of the solver.

        Args:
            temperature:
                The temperature `T` of the Boltzmann weight, in the units of the objective.

            step:
                The width of a proposal.

            moves:
                The move of every state of the members of a group that the solver samples member by member, by its
                dotted name, such as `{'spins.s': 'flip'}`, or `None` to sample the states of single elements.

            rule:
                The acceptance of the moves of members, `'metropolis'`, `'glauber'`, or `'heat-bath'`.

            schedule:
                The order of the moves of members, `'checkerboard'`, `'colored'`, `'random-sequential'`, or `'auto'`
                for the checkerboard when the lattice allows it and the random sequential order otherwise.

            neighborhood:
                The neighbourhood the local objective of a member reduces over, or `None` for the von Neumann
                neighbourhood of a square lattice.

            values:
                The values a random-value move draws from, by the dotted name of the state.

            width:
                The largest change of a rotation, or the deviation of a Gaussian move.

            random_state:
                The seed or generator of the proposals and acceptances.

            until:
                The end time of the run the solver prepares for.
        """
        super().__init__(random_state=random_state, until=until)
        self.temperature = temperature
        self.step = step
        self.moves = moves
        self.rule = rule
        self.schedule = schedule
        self.neighborhood = neighborhood
        self.values = values
        self.width = width

    def _get_member_solver(self) -> MonteCarloSolver:
        """Return the solver of the sweeps of members, which carries the moves."""
        return _MemberMetropolisSolver(metropolis=self, random_state=self.random_state, until=self.until)

    @classmethod
    def _get_support(cls) -> SupportSpec:
        """Return the times, spaces, laws, noise, and orders the solver solves."""
        return SupportSpec(
            time={_StaticTime},
            space={None, LatticeSpace, NetworkSpace, ContinuousSpace},
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


class _MemberMetropolisSolver(MonteCarloSolver):
    """A solver of the sweeps of members by the moves of a Metropolis solver.

    The compiler reads the moves of `metropolis` to turn the objective of every member into a Monte Carlo sweep.

    Args:
        metropolis:
            The Metropolis solver whose moves the sweeps follow.

        random_state:
            The seed or generator of the proposals and acceptances.

        until:
            The end time of the run the solver prepares for.
    """

    def __init__(
        self,
        *,
        metropolis: MetropolisSolver,
        random_state: int | np.random.Generator | None = None,
        until: float = np.inf,
    ) -> None:
        super().__init__(random_state=random_state, until=until)
        self.metropolis = metropolis
