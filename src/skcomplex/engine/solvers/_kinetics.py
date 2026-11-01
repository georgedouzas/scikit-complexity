"""Sample encounter laws with finite particles by direct simulation Monte Carlo."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

import numpy as np
from sage.all__sagemath_symbolics import SR

from ...core.laws import DestructionLaw, EncounterLaw, ProliferationLaw
from ...core.spaces import ContinuousTime, LatticeSpace
from ...errors import SolverError
from ...specs import SupportSpec
from ...types import Array, FormKind, Indices, IntArray, OrderKind
from ..problems import KineticProblemSpec, ProblemSpec
from ..problems._encounter_kernel import EncounterKernel
from ._solver import Solver


class KineticSolver(Solver):
    """Direct simulation Monte Carlo of encounter laws with finite particles on activity grids.

    Every density is held as counts of particles of one common mass, chosen so that the largest initial density has `n`
    particles. Every step of length at most `dt` applies the encounter laws in turn. A candidate particle at activity
    `h` encounters with probability `1 - exp(-lambda_h dt)`, where `lambda_h = sum_k eta[h, k] g_k` over the field
    density `g`, and picks its field activity and outcome in proportion to `eta[h, k] A[h, k, i] g_k`. A proliferation
    adds a Poisson number of particles and a destruction removes a binomial number. The state holds the empirical
    densities, so the trajectories match those of the gain and loss equations, which the particles approach as `n`
    grows.

    Args:
        n:
            The number of particles of the largest initial density.

        dt:
            The largest time step.

        random_state:
            The seed or generator of stochastic solvers.

        until:
            The end time of the run the solver prepares for, infinite for a run that goes on until it is stopped.

    Examples:
        >>> from skcomplex.engine.solvers import KineticSolver
        >>> from skcomplex.core.laws import DestructionLaw, EncounterLaw, ProliferationLaw
        >>> KineticSolver(n=1000)._get_support().laws == {EncounterLaw, ProliferationLaw, DestructionLaw}
        True
    """

    @classmethod
    def _get_support(cls) -> SupportSpec:
        """Return the times, spaces, laws, noise, and orders the solver solves."""
        return SupportSpec(
            time={ContinuousTime},
            space={LatticeSpace},
            laws={EncounterLaw, ProliferationLaw, DestructionLaw},
            orders=frozenset({OrderKind.INTEGER}),
        )

    @classmethod
    def _get_form(cls) -> FormKind:
        """Return the form the solver solves."""
        return FormKind.ODE

    def __init__(
        self,
        n: int = 10_000,
        dt: float = 0.01,
        *,
        random_state: int | np.random.Generator | None = None,
        until: float = math.inf,
    ) -> None:
        self.n = n
        self.dt = dt
        self.random_state = random_state
        self.until = until

    def _check_problem(self, problem: ProblemSpec) -> KineticProblemSpec:
        """Return a problem of encounter laws, raising an error for laws other than encounters, or no encounter."""
        if not isinstance(problem, KineticProblemSpec) or not problem.encounters:
            error_msg = 'KineticSolver samples encounter laws, but the problem has none. Use an ODE solver.'
            raise SolverError(error_msg)
        rates = any(not SR(value).is_zero() for value in problem.equations.values())
        if rates or problem.events or problem.delays or problem.renewals:
            error_msg = (
                'KineticSolver samples encounter laws alone, but the problem also has rate laws, events, delays, '
                'or renewals. Use ScipyODESolver or RK4Solver, which run them together.'
            )
            raise SolverError(error_msg)
        if not (isinstance(self.n, int) and self.n > 0 and self.dt > 0):
            error_msg = (
                f'KineticSolver needs a positive integer n and a positive dt, not n = {self.n!r} and '
                f'dt = {self.dt!r}.'
            )
            raise SolverError(error_msg)
        return problem

    def _prepare(self, context: ProblemSpec | None) -> None:
        """Prepare the stepper for a problem and draw the particles of the initial state.

        Args:
            context:
                The compiled problem.

        Raises:
            SolverError:
                The problem has another form, laws other than encounters, or no encounter, `n` or `dt` is not
                positive, or a density of the initial state is negative.
        """
        super()._prepare(context)
        problem = self.problem_
        self._encounters = self._check_problem(problem).encounters
        self.densities_ = self._densities(self._encounters)
        self.mass_ = 0.0
        self._counts: dict[tuple[int, ...], IntArray] = {}
        self._sample(self.initial_state_)
        self.initial_state_ = self._state(self.initial_state_)

    def _sample(self, x: Array) -> None:
        """Draw the particle counts of every density from a state, with the mass that gives the largest one `n`."""
        totals = [float(np.sum(x[indices])) for indices in self.densities_]
        if min((float(np.min(x[indices])) for indices in self.densities_), default=0.0) < 0:
            error_msg = 'KineticSolver samples nonnegative densities, but a density of the state is negative.'
            raise SolverError(error_msg)
        self.mass_ = max(totals, default=0.0) / self.n or 1.0
        for indices, total in zip(self.densities_, totals, strict=True):
            count = round(total / self.mass_)
            weights = x[indices] / total if total > 0 else np.full(indices.size, 1 / indices.size)
            self._counts[tuple(indices.tolist())] = self.rng_.multinomial(count, weights).astype(np.int64)

    def _state(self, x: Array) -> Array:
        """Return a state with every density replaced by its empirical density."""
        state = x.copy()
        for indices in self.densities_:
            state[indices] = self._counts[tuple(indices.tolist())] * self.mass_
        return state

    def _move(self, encounter: EncounterKernel, p: Array, step: float) -> None:
        """Move the candidates that encounter in one step to their outcome activities."""
        counts = self._counts[tuple(encounter.candidate.tolist())]
        field = self._counts[tuple(encounter.field.tolist())] * self.mass_
        test = self._counts[tuple(encounter.test.tolist())]
        candidates, fields, outcomes = encounter.entries
        rates = encounter.weights(p) * field[fields]
        totals = np.bincount(candidates, rates, minlength=counts.size)
        movers = self.rng_.binomial(counts, -np.expm1(-totals * step))
        bounds = np.searchsorted(candidates, np.arange(counts.size + 1))
        for activity in np.flatnonzero(movers):
            selected = slice(bounds[activity], bounds[activity + 1])
            shares = self.rng_.multinomial(movers[activity], rates[selected] / totals[activity])
            counts[activity] -= movers[activity]
            np.add.at(test, outcomes[selected], shares)

    def _proliferate(self, encounter: EncounterKernel, p: Array, step: float) -> None:
        """Add the particles that encounters create in one step."""
        counts = self._counts[tuple(encounter.candidate.tolist())]
        field = self._counts[tuple(encounter.field.tolist())] * self.mass_
        test = self._counts[tuple(encounter.test.tolist())]
        candidates, fields, outcomes = encounter.entries
        births = self.rng_.poisson(counts[candidates] * encounter.weights(p) * field[fields] * step)
        np.add.at(test, outcomes, births)

    def _destroy(self, encounter: EncounterKernel, p: Array, step: float) -> None:
        """Remove the particles that encounters destroy in one step."""
        counts = self._counts[tuple(encounter.candidate.tolist())]
        field = self._counts[tuple(encounter.field.tolist())] * self.mass_
        totals = encounter.rate_matrix(p) @ field
        counts -= self.rng_.binomial(counts, -np.expm1(-totals * step))

    def _step(self, p: Array, step: float) -> None:
        """Apply every encounter law in turn for one step."""
        actions = {'transition': self._move, 'proliferation': self._proliferate, 'destruction': self._destroy}
        for encounter in self._encounters:
            actions[encounter.kind](encounter, p, step)

    def _advance(self, t: float, x: Array, p: Array, until: float) -> Array:
        """Sample the encounters of the particles from time `t` to time `until` in equal steps no longer than `dt`.

        A state that differs from the empirical densities of the particles, such as one an observer changed, is
        sampled into particles anew.

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
                A density of the state is negative.
        """
        if until <= t:
            return x
        if not self._counts or not np.allclose(self._state(x), x, rtol=1e-12, atol=0.0):
            self._sample(x)
        n_steps = max(1, math.ceil((until - t) / self.dt - 1e-12))
        step = (until - t) / n_steps
        for _ in range(n_steps):
            self._step(p, step)
        return self._state(x)

    @staticmethod
    def _densities(encounters: tuple[EncounterKernel, ...]) -> list[Indices]:
        """Return the state indices of every density the encounters read or change, each once."""
        found: dict[tuple[int, ...], Indices] = {}
        for encounter in encounters:
            for indices in (encounter.candidate, encounter.field, encounter.test):
                found.setdefault(tuple(indices.tolist()), indices)
        return list(found.values())
