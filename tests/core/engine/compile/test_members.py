"""Test the lowering of member offset reads along the order of a group."""

import time
from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import sin

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import JumpLaw, RateLaw, UpdateLaw
from skcomplex.core.components.laws.utils import shift_member
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import RK4Solver
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from skcomplex.systems.physics.statistical_mechanics import (
    ParticleSiteElement,
    ZeroRangeHopInteraction,
    ZeroRangeHopsInteraction,
    ZeroRangeProcessSystem,
)


class CarElement(Element):
    """Car element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.0,
        ahead: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.ahead = ahead


class LookAheadInteraction(Interaction):
    """Update of every car to the position of the car a number of places ahead."""

    def __init__(
        self,
        label: str,
        *,
        cars: Annotated[str, RoleSpec()],
        offset: int = 1,
        boundary: str = 'periodic',
        fill: float = 0.0,
    ) -> None:
        super().__init__(label)
        self.cars = cars
        self.offset = offset
        self.boundary = boundary
        self.fill = fill

    def _get_laws(self, s):
        """Return the update of the read position."""
        value = shift_member(s.cars.x, self.offset, boundary=self.boundary, fill=self.fill)
        return [UpdateLaw(s.cars.ahead, value)]


class RingInteraction(Interaction):
    """Coupling of every member of a ring to the members on either side."""

    def __init__(
        self,
        label: str,
        *,
        cars: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.cars = cars

    def _get_laws(self, s):
        """Return the rate of a member from its neighbours on the ring."""
        x = s.cars.x
        return [RateLaw(x, sin(shift_member(x, 1) - x) + 0.5 * sin(shift_member(x, -2) - x))]


def _ahead(backend, **settings):
    """Return the read positions of a group of four cars after one update."""
    system = CompositeSystem(
        elements=[GroupElement('cars', element=CarElement, size=4, values={'x': [10.0, 11.0, 12.0, 13.0]})],
        interactions=[LookAheadInteraction('look', cars='cars', **settings)],
    )
    problem = PassCompiler(backend=backend).prepare(system).run().problem_
    state = problem.update(0.0, problem.initial_state, problem.initial_parameters)
    return problem.states.read(state, 'cars.ahead').tolist()


@pytest.mark.parametrize('backend', [NumpyBackend, NumbaBackend])
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
    assert _ahead(backend(), **settings) == expected


def test_shift_member_runs_the_same_on_both_backends():
    """Test that a ring of 1000 members with offset reads has the same trajectory on NumPy and Numba."""
    x = np.random.default_rng(0).uniform(0, 2 * np.pi, 1000)
    trajectories = []
    for backend in (NumpyBackend(), NumbaBackend()):
        system = CompositeSystem(
            elements=[GroupElement('cars', element=CarElement, size=1000, values={'x': x})],
            interactions=[RingInteraction('ring', cars='cars')],
        )
        trajectory = simulate(system, until=2.0, solver=RK4Solver(dt=0.01), dt=0.5, backend=backend)
        assert trajectory.problem.backend == backend._get_name()
        trajectories.append(trajectory.values('cars.x'))
    np.testing.assert_allclose(trajectories[1], trajectories[0], rtol=1e-10)


def test_shift_member_of_a_single_element_raises_error():
    """Test that an offset read of a single element's state names the component and the state."""
    system = CompositeSystem(
        elements=[CarElement('car', x=1.0)],
        interactions=[LookAheadInteraction('look', cars='car')],
    )
    with pytest.raises(ValidationError, match=r'look reads car\.x at a member offset'):
        PassCompiler().prepare(system).run()


class ShiftedGapInteraction(Interaction):
    """Update of every car to the position ahead plus a gap, a parameter of the interaction."""

    def __init__(
        self,
        label: str,
        *,
        cars: Annotated[str, RoleSpec()],
        gap: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label)
        self.cars = cars
        self.gap = gap

    def _get_laws(self, s):
        """Return the update of the read position."""
        return [UpdateLaw(s.cars.ahead, shift_member(s.cars.x + s.gap, 1))]


def test_shift_member_reads_a_parameter_of_an_interaction_as_it_is():
    """Test that a parameter of an interaction read at a member offset is the same at every member."""
    system = CompositeSystem(
        elements=[GroupElement('cars', element=CarElement, size=3, values={'x': [1.0, 2.0, 3.0]})],
        interactions=[ShiftedGapInteraction('gap', cars='cars', gap=0.5)],
    )
    problem = PassCompiler().prepare(system).run().problem_
    state = problem.update(0.0, problem.initial_state, problem.initial_parameters)
    assert problem.states.read(state, 'cars.ahead').tolist() == [2.5, 3.5, 1.5]


def test_shift_member_rejects_unknown_boundary():
    """Test that the boundary of an offset read is periodic, reflecting, or fixed."""
    with pytest.raises(ValidationError, match="'open'"):
        shift_member(GroupElement('cars', element=CarElement, size=2).size, 1, boundary='open')


class HopInteraction(Interaction):
    """Hop of one particle from every site to the next one along the group order, at a rate equal to the occupation."""

    def __init__(
        self,
        label: str,
        *,
        sites: Annotated[str, RoleSpec()],
        boundary: str = 'periodic',
    ) -> None:
        super().__init__(label)
        self.sites = sites
        self.boundary = boundary

    def _get_laws(self, s):
        """Return the hop to the next site."""
        n = s.sites.n
        return [JumpLaw(n, {n: -1, shift_member(n, 1, boundary=self.boundary): 1})]


class SiteElement(Element):
    """Site element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        n: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.n = n


def _hopping(occupations, boundary='periodic'):
    """Return a group of sites whose particles hop to the next site."""
    sites = GroupElement(
        'sites',
        element=SiteElement,
        size=len(occupations),
        values={'n': np.asarray(occupations, dtype=float)},
    )
    return CompositeSystem(elements=[sites], interactions=[HopInteraction('hop', sites='sites', boundary=boundary)])


@pytest.mark.parametrize('backend', [NumpyBackend, NumbaBackend])
def test_jump_at_member_offset_conserves_particles_on_a_ring(backend):
    """Test that a hop to the next member of a ring keeps every particle and moves one per event."""
    system = _hopping(np.full(50, 3.0))
    problem = PassCompiler(backend=backend()).prepare(system).run().problem_
    assert problem.jumps[0].size == 50
    np.testing.assert_array_equal(problem.stoichiometry, np.roll(np.eye(50), 1, axis=1) - np.eye(50))
    for seed in range(3):
        trajectory = simulate(system, until=20, dt=1.0, backend=backend(), random_state=seed)
        totals = np.asarray(trajectory.values('sites.n')).sum(axis=1)
        assert (totals == 150).all()


@pytest.mark.parametrize('backend', [NumpyBackend, NumbaBackend])
def test_jump_at_member_offset_leaves_past_a_fixed_end(backend):
    """Test that the hop of the last member past a fixed end removes the particle and changes no other member."""
    system = _hopping([0.0, 0.0, 0.0, 0.0, 1.0], boundary='fixed')
    problem = PassCompiler(backend=backend()).prepare(system).run().problem_
    assert problem.stoichiometry[4].tolist() == [0.0, 0.0, 0.0, 0.0, -1.0]
    trajectory = simulate(system, until=100, dt=10.0, backend=backend(), random_state=0)
    assert np.asarray(trajectory.values('sites.n'))[-1].tolist() == [0.0] * 5


@pytest.mark.parametrize('backend', [NumpyBackend, NumbaBackend])
def test_jump_at_member_offset_matches_per_link_interactions(backend):
    """Test that a zero-range process of offset jumps and one of per-link interactions have one mean profile."""
    occupations = np.zeros(20)
    occupations[0] = 40.0
    sites = GroupElement(
        'sites',
        element=ParticleSiteElement,
        space=LatticeSpace(20),
        values={'n': occupations},
        on='sites',
    )
    links = [
        ZeroRangeHopInteraction(f'hop_{index}', link=[f'sites[{index}]', f'sites[{(index + 1) % 20}]'], q=0.5)
        for index in range(20)
    ]
    systems = [
        CompositeSystem(elements=[sites], interactions=[ZeroRangeHopsInteraction('hops', sites='sites', q=0.5)]),
        CompositeSystem(elements=[sites], interactions=links),
    ]
    profiles = []
    for system in systems:
        finals = [
            np.asarray(simulate(system, until=2, dt=2.0, backend=backend(), random_state=seed).values('sites.n'))[-1]
            for seed in range(200)
        ]
        profiles.append((np.mean(finals, axis=0), np.std(finals, axis=0) / np.sqrt(200)))
    (first, first_error), (second, second_error) = profiles
    assert (np.abs(first - second) <= 4 * np.hypot(first_error, second_error) + 1e-12).all()


@pytest.mark.benchmark
def test_jump_at_member_offset_compiles_many_sites_fast():
    """Test that a zero-range process of 1000 sites compiles one group law and runs until 5 within 10 seconds."""
    start = time.perf_counter()
    trajectory = simulate(
        ZeroRangeProcessSystem(1000).prepare().system_,
        until=5,
        backend=NumbaBackend(),
        random_state=0,
    )
    assert time.perf_counter() - start < 10
    assert len(trajectory.problem.interactions) == 1
    assert [channels.size for channels in trajectory.problem.jumps] == [1000, 1000]
