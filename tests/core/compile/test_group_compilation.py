"""Test the compilation of groups, pair interactions, and boundary laws."""

import time

import numpy as np
import pytest

from skcomplex.core.base import Domain, ValidationError, parameter, role
from skcomplex.core.compile import compile_system
from skcomplex.core.model import (
    EuclideanSpace,
    Interaction,
    Jump,
    Noise,
    PairPotential,
    Particle,
    PeriodicBox,
    Rate,
    ReflectingBox,
    System,
    Update,
)
from skcomplex.models.biology.ecology import Population

CUTOFF = 2.5


class LennardJones(PairPotential):
    """Lennard-Jones pair potential."""

    epsilon: float = parameter(1.0, domain=Domain.POSITIVE)
    sigma: float = parameter(1.0, domain=Domain.POSITIVE)

    def pair_energy(self, s, r):
        """Return `4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)`."""
        return 4 * s.epsilon * ((s.sigma / r) ** 12 - (s.sigma / r) ** 6)


class Doubling(Interaction):
    """Interaction with a callable rate over a group."""

    target: str = role()

    def laws(self, s):
        """Return a callable rate equal to twice the member counts."""
        name = f'{self.target}.n'
        return [Rate(s.target.n, lambda t, values: 2 * values[name])]


class Birth(Interaction):
    """Stochastic birth of every member of a group."""

    target: str = role()
    rate: float = parameter(1.0, domain=Domain.POSITIVE)

    def laws(self, s):
        """Return one jump per member."""
        return [Jump(s.rate * s.target.n, {s.target.n: 1})]


class Volatility(Interaction):
    """Noise on every member of a group."""

    target: str = role()

    def laws(self, s):
        """Return the noise amplitude `n`."""
        return [Noise(s.target.n, s.target.n)]


class Halving(Interaction):
    """Discrete halving of every member of a group."""

    target: str = role()

    def laws(self, s):
        """Return the next value `n / 2`."""
        return [Update(s.target.n, s.target.n / 2)]


class SingleUpdateFromGroup(Interaction):
    """Update of a single element from a group, which has no meaning."""

    target: str = role()
    total: str = role()

    def laws(self, s):
        """Return an update of the single element."""
        return [Update(s.total.n, s.target.n)]


def _lennard_jones_gas(n, seed=0):
    """Return a gas of Lennard-Jones particles on a jittered lattice."""
    rng = np.random.default_rng(seed)
    side = int(np.ceil(np.sqrt(n)))
    grid = np.stack(np.meshgrid(np.arange(side), np.arange(side)), axis=-1).reshape(-1, 2)[:n] * 1.2
    return System(
        components=[Particle.group('gas', n, x=grid + rng.normal(0, 0.05, (n, 2)), v=rng.normal(0, 0.5, (n, 2)))],
        interactions=[LennardJones('lj', first='gas', second='gas', cutoff=CUTOFF)],
        space=EuclideanSpace(2),
    )


def test_compile_group_time_does_not_grow_with_size():
    """Test that compiling 100000 members takes no longer than 10 members within a factor of 2."""

    def compile_time(n):
        start = time.perf_counter()
        compile_system(
            System(
                components=[Particle.group('gas', n)],
                interactions=[ReflectingBox('box', target='gas')],
                space=EuclideanSpace(2),
            ),
        )
        return time.perf_counter() - start

    compile_time(10)
    assert compile_time(100_000) <= 2 * compile_time(10) + 0.05


def test_compile_group_names_states_by_group_path():
    """Test that a group state is named by its group path with the group shape."""
    problem = compile_system(System(components=[Particle.group('gas', 500)], space=EuclideanSpace(2)))
    assert problem.states.entry('gas.x').shape == (500, 2)
    assert problem.equations['gas.x[i, 0]'] == problem.symbols['gas.v'][0]


def test_compile_pair_forces_match_brute_force_sum():
    """Test that pair forces equal the sum over members closer than the cutoff."""
    problem = compile_system(_lennard_jones_gas(60))
    x, p = problem.initial_state, problem.initial_parameters
    positions = problem.states.read(x, 'gas.x')
    accelerations = problem.states.read(problem.drift(0.0, x, p), 'gas.v')
    expected = np.zeros_like(positions)
    for i in range(len(positions)):
        for j in range(len(positions)):
            separation = positions[i] - positions[j]
            r = np.linalg.norm(separation)
            if i != j and r < CUTOFF:
                expected[i] += 24 * (2 / r**13 - 1 / r**7) * separation / r
    assert np.allclose(accelerations, expected, rtol=1e-10, atol=1e-10)


def test_compile_pair_forces_between_two_groups():
    """Test that a pair interaction between two groups attracts both beyond the potential minimum."""
    system = System(
        components=[Particle.group('a', 1, x=[[0.0, 0.0]]), Particle.group('b', 1, x=[[1.5, 0.0]])],
        interactions=[LennardJones('lj', first='a', second='b', cutoff=CUTOFF)],
        space=EuclideanSpace(2),
    )
    problem = compile_system(system)
    drift = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    a, b = problem.states.read(drift, 'a.v'), problem.states.read(drift, 'b.v')
    assert a[0, 0] > 0
    assert np.allclose(a, -b)


def test_compile_pair_forces_use_minimum_image():
    """Test that periodic pair interactions use minimum image distances."""
    system = System(
        components=[Particle.group('gas', 2, x=[[0.1, 0.5], [9.6, 0.5]])],
        interactions=[LennardJones('lj', first='gas', second='gas', cutoff=CUTOFF, period=10.0)],
        space=EuclideanSpace(2),
    )
    problem = compile_system(system)
    accelerations = problem.states.read(problem.drift(0.0, problem.initial_state, problem.initial_parameters), 'gas.v')
    assert accelerations[0, 0] > 0
    assert accelerations[1, 0] < 0


def test_compile_reflecting_box_keeps_ode_form():
    """Test that a reflecting box adds boundary laws without changing the form."""
    problem = compile_system(
        System(
            components=[Particle.group('gas', 4)],
            interactions=[ReflectingBox('box', target='gas')],
            space=EuclideanSpace(2),
        ),
    )
    assert problem.form.name == 'ODE'
    assert problem.boundary is not None


def test_compile_periodic_box_wraps_positions():
    """Test that a periodic box wraps a position that left the box."""
    problem = compile_system(
        System(
            components=[Particle.group('gas', 1, x=[[1.25, 0.5]], v=[[1.0, 0.0]])],
            interactions=[PeriodicBox('box', target='gas', size=1.0)],
            space=EuclideanSpace(2),
        ),
    )
    state = problem.boundary(0.0, problem.initial_state, problem.initial_parameters)
    assert np.allclose(problem.states.read(state, 'gas.x'), [[0.25, 0.5]])
    assert np.allclose(problem.states.read(state, 'gas.v'), [[1.0, 0.0]])


def test_compile_group_callable_rate():
    """Test that a callable law over a group returns one value per member."""
    problem = compile_system(
        System(
            components=[Population.group('herds', 3, n=[1.0, 2.0, 3.0])],
            interactions=[Doubling('d', target='herds')],
        ),
    )
    assert problem.drift(0.0, problem.initial_state, problem.initial_parameters).tolist() == [2.0, 4.0, 6.0]
    assert problem.has_callables


def test_compile_group_jumps_have_one_channel_per_member():
    """Test that a jump law over a group has one channel per member."""
    problem = compile_system(
        System(
            components=[Population.group('herds', 3, n=[1.0, 2.0, 3.0])],
            interactions=[Birth('birth', target='herds')],
        ),
    )
    assert problem.propensities(0.0, problem.initial_state, problem.initial_parameters).tolist() == [1.0, 2.0, 3.0]
    assert problem.stoichiometry.tolist() == np.eye(3).tolist()


def test_compile_group_noise_has_one_process_per_member():
    """Test that a noise law over a group drives every member with its own process."""
    problem = compile_system(
        System(
            components=[Population.group('herds', 3, n=[1.0, 2.0, 3.0])],
            interactions=[Volatility('vol', target='herds')],
        ),
    )
    assert problem.noise_rows.tolist() == [0, 1, 2]
    assert problem.diffusion(0.0, problem.initial_state, problem.initial_parameters).tolist() == [1.0, 2.0, 3.0]


def test_compile_group_updates_every_member():
    """Test that a discrete update over a group updates every member."""
    problem = compile_system(
        System(components=[Population.group('herds', 2, n=[4.0, 8.0])], interactions=[Halving('half', target='herds')]),
    )
    assert problem.update(0.0, problem.initial_state, problem.initial_parameters).tolist() == [2.0, 4.0]
    assert problem.equations['herds.n[i]'] == problem.symbols['herds.n'] / 2


def test_compile_update_of_single_element_from_group_raises_error():
    """Test that an update cannot set a single element from a group."""
    system = System(
        components=[Population.group('herds', 2), Population('total')],
        interactions=[SingleUpdateFromGroup('sum', target='herds', total='total')],
    )
    with pytest.raises(ValidationError, match='single element from a group'):
        compile_system(system)


def test_compile_boundary_twice_raises_error():
    """Test that a state takes one boundary law."""
    system = System(
        components=[Particle.group('gas', 2)],
        interactions=[ReflectingBox('box', target='gas'), PeriodicBox('wrap', target='gas')],
        space=EuclideanSpace(1),
    )
    with pytest.raises(ValidationError, match='second time'):
        compile_system(system)
