"""Sample encounter laws with finite particles by direct simulation Monte Carlo."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from typing import ClassVar

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR

from ..base import ComponentKind, DomainKind, Form, LawKind, OrderKind, SolverError, SpaceKind, Supports, TimeKind
from ..compile import CompiledEncounter, Problem, Vector
from ._base import Stepper, register_solver

type Counts = npt.NDArray[np.int64]
type Indices = npt.NDArray[np.intp]


def _densities(encounters: tuple[CompiledEncounter, ...]) -> list[Indices]:
    """Return the state indices of every density the encounters read or change, each once."""
    found: dict[tuple[int, ...], Indices] = {}
    for encounter in encounters:
        for indices in (encounter.candidate, encounter.field, encounter.test):
            found.setdefault(tuple(indices.tolist()), indices)
    return list(found.values())


class ParticleKinetics(Stepper):
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

    Examples:
        >>> from skcomplex.core.solvers import ParticleKinetics
        >>> ParticleKinetics(n=1000).supports.laws == {'encounter'}
        True
    """

    supports: ClassVar[Supports] = Supports(
        time={TimeKind.CONTINUOUS},
        space={SpaceKind.LATTICE},
        laws={LawKind.ENCOUNTER},
        domains={DomainKind.REAL},
        components={ComponentKind.SITE_GROUP},
        orders={OrderKind.INTEGER},
    )

    form: ClassVar[Form] = Form.ODE

    def __init__(self, n: int = 10_000, dt: float = 0.01) -> None:
        self.n = n
        self.dt = dt

    def _check_problem(self, problem: Problem) -> None:
        """Raise an error when a problem has laws other than encounters, or no encounter."""
        if not problem.encounters:
            error_msg = 'ParticleKinetics samples encounter laws, but the problem has none. Use an ODE solver.'
            raise SolverError(error_msg)
        rates = any(not SR(value).is_zero() for value in problem.equations.values())
        if rates or problem.events or problem.delays or problem.renewals:
            error_msg = (
                'ParticleKinetics samples encounter laws alone, but the problem also has rate laws, events, delays, '
                'or renewals. Use ScipyODE or RK4, which run them together.'
            )
            raise SolverError(error_msg)
        if not (isinstance(self.n, int) and self.n > 0 and self.dt > 0):
            error_msg = (
                f'ParticleKinetics needs a positive integer n and a positive dt, not n = {self.n!r} and '
                f'dt = {self.dt!r}.'
            )
            raise SolverError(error_msg)

    def initialize(self, problem: Problem, rng: np.random.Generator) -> None:
        """Prepare the stepper for a problem.

        Args:
            problem:
                The compiled problem.

            rng:
                The random number generator that draws the particles and their encounters.

        Raises:
            SolverError:
                The problem has another form, laws other than encounters, or no encounter, or `n` or `dt` is not
                positive.
        """
        super().initialize(problem, rng)
        self._check_problem(problem)
        self.densities_ = _densities(problem.encounters)
        self.mass_ = 0.0
        self._counts: dict[tuple[int, ...], Counts] = {}

    def _sample(self, x: Vector) -> None:
        """Draw the particle counts of every density from a state, with the mass that gives the largest one `n`."""
        totals = [float(np.sum(x[indices])) for indices in self.densities_]
        if min((float(np.min(x[indices])) for indices in self.densities_), default=0.0) < 0:
            error_msg = 'ParticleKinetics samples nonnegative densities, but a density of the state is negative.'
            raise SolverError(error_msg)
        self.mass_ = max(totals, default=0.0) / self.n or 1.0
        for indices, total in zip(self.densities_, totals, strict=True):
            count = round(total / self.mass_)
            weights = x[indices] / total if total > 0 else np.full(indices.size, 1 / indices.size)
            self._counts[tuple(indices.tolist())] = self.rng_.multinomial(count, weights).astype(np.int64)

    def _state(self, x: Vector) -> Vector:
        """Return a state with every density replaced by its empirical density."""
        state = x.copy()
        for indices in self.densities_:
            state[indices] = self._counts[tuple(indices.tolist())] * self.mass_
        return state

    def initial_state(self, x: Vector) -> Vector:
        """Return the state vector of the first frame, the empirical densities of the drawn particles.

        Args:
            x:
                The initial state vector of the problem.

        Returns:
            state:
                The state vector of the first frame.
        """
        self._sample(x)
        return self._state(x)

    def _move(self, encounter: CompiledEncounter, p: Vector, step: float) -> None:
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

    def _proliferate(self, encounter: CompiledEncounter, p: Vector, step: float) -> None:
        """Add the particles that encounters create in one step."""
        counts = self._counts[tuple(encounter.candidate.tolist())]
        field = self._counts[tuple(encounter.field.tolist())] * self.mass_
        test = self._counts[tuple(encounter.test.tolist())]
        candidates, fields, outcomes = encounter.entries
        births = self.rng_.poisson(counts[candidates] * encounter.weights(p) * field[fields] * step)
        np.add.at(test, outcomes, births)

    def _destroy(self, encounter: CompiledEncounter, p: Vector, step: float) -> None:
        """Remove the particles that encounters destroy in one step."""
        counts = self._counts[tuple(encounter.candidate.tolist())]
        field = self._counts[tuple(encounter.field.tolist())] * self.mass_
        totals = encounter.rate_matrix(p) @ field
        counts -= self.rng_.binomial(counts, -np.expm1(-totals * step))

    def _step(self, p: Vector, step: float) -> None:
        """Apply every encounter law in turn for one step."""
        actions = {'transition': self._move, 'proliferation': self._proliferate, 'destruction': self._destroy}
        for encounter in self.problem_.encounters:
            actions[encounter.kind](encounter, p, step)

    def advance(self, t: float, x: Vector, p: Vector, until: float) -> Vector:
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


register_solver('particles', ParticleKinetics)
