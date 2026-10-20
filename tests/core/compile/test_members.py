"""Test the lowering of member offset reads along the order of a group."""

import time

import numpy as np
import pytest
from sage.all__sagemath_symbolics import sin

from skcomplex.core.base import ValidationError, role
from skcomplex.core.compile import compile_system
from skcomplex.core.model import Element, Interaction, Jump, LatticeSpace, Rate, System, Update, shift_member
from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import RK4
from skcomplex.models.physics.statistical_mechanics import (
    ParticleSite,
    ZeroRangeHop,
    ZeroRangeHops,
    zero_range_process,
)

Car = Element.define('Car', states={'x': 0.0, 'ahead': 0.0})


class LookAhead(Interaction):
    """Update of every car to the position of the car a number of places ahead."""

    cars: str = role()
    offset: int = 1
    boundary: str = 'periodic'
    fill: float = 0.0

    def laws(self, s):
        """Return the update of the read position."""
        value = shift_member(s.cars.x, self.offset, boundary=self.boundary, fill=self.fill)
        return [Update(s.cars.ahead, value)]


class Ring(Interaction):
    """Coupling of every member of a ring to the members on either side."""

    cars: str = role()

    def laws(self, s):
        """Return the rate of a member from its neighbours on the ring."""
        x = s.cars.x
        return [Rate(x, sin(shift_member(x, 1) - x) + 0.5 * sin(shift_member(x, -2) - x))]


def _ahead(backend='numpy', **settings):
    """Return the read positions of a group of four cars after one update."""
    system = System(
        elements=[Car.group('cars', 4, x=[10.0, 11.0, 12.0, 13.0])],
        interactions=[LookAhead('look', cars='cars', **settings)],
    )
    problem = compile_system(system, backend=backend)
    state = problem.update(0.0, problem.initial_state, problem.initial_parameters)
    return problem.states.read(state, 'cars.ahead').tolist()


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
@pytest.mark.parametrize(
    ('settings', 'expected'),
    [
        ({}, [11.0, 12.0, 13.0, 10.0]),
        ({'boundary': 'fixed', 'fill': -1.0}, [11.0, 12.0, 13.0, -1.0]),
        ({'boundary': 'reflecting'}, [11.0, 12.0, 13.0, 13.0]),
        ({'offset': -1}, [13.0, 10.0, 11.0, 12.0]),
    ],
)
def test_shift_member_reads_the_member_along_the_group_order(backend, settings, expected):
    """Test that member `i` reads member `i + offset`, wrapped, mirrored, or filled past the ends."""
    assert _ahead(backend, **settings) == expected


def test_shift_member_runs_the_same_on_both_backends():
    """Test that a ring of 1000 members with offset reads has the same trajectory on NumPy and Numba."""
    x = np.random.default_rng(0).uniform(0, 2 * np.pi, 1000)
    trajectories = []
    for backend in ('numpy', 'numba'):
        system = System(elements=[Car.group('cars', 1000, x=x)], interactions=[Ring('ring', cars='cars')])
        simulation = Simulation(system, solver=RK4(dt=0.01), dt=0.5, backend=backend).run(until=2.0)
        assert simulation.problem_.backend == backend
        trajectories.append(simulation.trajectory_['cars.x'])
    np.testing.assert_allclose(trajectories[1], trajectories[0], rtol=1e-10)


def test_shift_member_of_a_single_element_raises_error():
    """Test that an offset read of a single element's state names the component and the state."""
    system = System(elements=[Car('car', x=1.0)], interactions=[LookAhead('look', cars='car')])
    with pytest.raises(ValidationError, match=r'look reads car\.x at a member offset'):
        compile_system(system)


def test_shift_member_rejects_unknown_boundary():
    """Test that the boundary of an offset read is periodic, reflecting, or fixed."""
    with pytest.raises(ValidationError, match="'open'"):
        shift_member(Car.group('cars', 2).size, 1, boundary='open')


class Hop(Interaction):
    """Hop of one particle from every site to the next one along the group order, at a rate equal to the occupation."""

    sites: str = role()
    boundary: str = 'periodic'

    def laws(self, s):
        """Return the hop to the next site."""
        n = s.sites.n
        return [Jump(n, {n: -1, shift_member(n, 1, boundary=self.boundary): 1})]


Site = Element.define('Site', states={'n': 0.0})


def _hopping(occupations, boundary='periodic'):
    """Return a group of sites whose particles hop to the next site."""
    sites = Site.group('sites', len(occupations), n=np.asarray(occupations, dtype=float))
    return System(elements=[sites], interactions=[Hop('hop', sites='sites', boundary=boundary)])


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_jump_at_member_offset_conserves_particles_on_a_ring(backend):
    """Test that a hop to the next member of a ring keeps every particle and moves one per event."""
    problem = compile_system(_hopping(np.full(50, 3.0)), backend=backend)
    assert problem.jumps[0].size == 50
    np.testing.assert_array_equal(problem.stoichiometry, np.roll(np.eye(50), 1, axis=1) - np.eye(50))
    for seed in range(3):
        trajectory = Simulation(problem, dt=1.0, random_state=seed).run(until=20).trajectory_
        totals = np.asarray(trajectory['sites.n']).sum(axis=1)
        assert (totals == 150).all()


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_jump_at_member_offset_leaves_past_a_fixed_end(backend):
    """Test that the hop of the last member past a fixed end removes the particle and changes no other member."""
    problem = compile_system(_hopping([0.0, 0.0, 0.0, 0.0, 1.0], boundary='fixed'), backend=backend)
    assert problem.stoichiometry[4].tolist() == [0.0, 0.0, 0.0, 0.0, -1.0]
    trajectory = Simulation(problem, dt=10.0, random_state=0).run(until=100).trajectory_
    assert np.asarray(trajectory['sites.n'])[-1].tolist() == [0.0] * 5


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_jump_at_member_offset_matches_per_link_interactions(backend):
    """Test that a zero-range process of offset jumps and one of per-link interactions have one mean profile."""
    occupations = np.zeros(20)
    occupations[0] = 40.0
    sites = ParticleSite.lattice('sites', LatticeSpace(20), n=occupations)
    links = [
        ZeroRangeHop(f'hop_{index}', link=[f'sites[{index}]', f'sites[{(index + 1) % 20}]'], q=0.5)
        for index in range(20)
    ]
    systems = [
        System(elements=[sites], interactions=[ZeroRangeHops('hops', sites='sites', q=0.5)]),
        System(elements=[sites], interactions=links),
    ]
    profiles = []
    for system in systems:
        problem = compile_system(system, backend=backend)
        finals = [
            np.asarray(Simulation(problem, dt=2.0, random_state=seed).run(until=2).trajectory_['sites.n'])[-1]
            for seed in range(200)
        ]
        profiles.append((np.mean(finals, axis=0), np.std(finals, axis=0) / np.sqrt(200)))
    (first, first_error), (second, second_error) = profiles
    assert (np.abs(first - second) <= 4 * np.hypot(first_error, second_error) + 1e-12).all()


def test_jump_at_member_offset_compiles_many_sites_fast():
    """Test that a zero-range process of 1000 sites compiles one group law and runs until 5 within 10 seconds."""
    start = time.perf_counter()
    simulation = Simulation(zero_range_process(1000), backend='numba', random_state=0).run(until=5)
    assert time.perf_counter() - start < 10
    assert len(simulation.problem_.interactions) == 1
    assert [channels.size for channels in simulation.problem_.jumps] == [1000, 1000]
