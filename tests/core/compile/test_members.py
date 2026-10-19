"""Test the lowering of member offset reads along the order of a group."""

import numpy as np
import pytest
from sage.all__sagemath_symbolics import sin

from skcomplex.core.base import ValidationError, role
from skcomplex.core.compile import compile_system
from skcomplex.core.model import Element, Interaction, Rate, System, Update, shift_member
from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import RK4

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
