"""Test the `_basin_stability` module of `skcomplex.core.experiments`."""

from typing import Annotated

import pytest
from sage.all__sagemath_symbolics import NN

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import JumpLaw
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.errors import ValidationError
from skcomplex.core.experiments import basin_stability
from skcomplex.core.specs import StateSpec
from skcomplex.systems.physics.nonlinear_oscillators import DuffingSystem

BOUNDS = {'duffing.x': (-2.0, 2.0), 'duffing.x_d1': (-2.0, 2.0)}


class ArrivalsElement(Element):
    """Count of arrivals that grows by one at a constant rate."""

    def __init__(self, label, visual=None, *, count: Annotated[float, StateSpec(domain=NN)] = 0) -> None:
        super().__init__(label, visual=visual)
        self.count = count

    def _get_laws(self, s):
        """Return the arrivals."""
        return [JumpLaw(1.0, {s.count: 1})]


def side(trajectory):
    """Return the well an unforced double-well oscillator settles in."""
    return 'right' if trajectory.get_values('duffing.x')[-1] > 0 else 'left'


def test_double_well_basins_are_symmetric():
    """Test that the damped double-well oscillator's two basins hold about half of the initial states each."""
    result = basin_stability(DuffingSystem(gamma=0.0), BOUNDS, side, n=80, until=40, dt=0.5, random_state=0)
    assert set(result.fractions) == {'left', 'right'}
    assert result.fractions['left'] == pytest.approx(0.5, abs=3 * result.errors['left'] + 0.05)
    assert len(result.labels) == 80
    assert len(result.rows) == 80


@pytest.mark.slow
def test_vectorised_basins_match_the_serial_runs():
    """Test that basin stability over 200 initial states run as one ensemble equals the serial runs."""
    settings = {'n': 200, 'until': 40, 'dt': 0.5, 'random_state': 0}
    serial = basin_stability(DuffingSystem(gamma=0.0), BOUNDS, side, **settings)
    vectorised = basin_stability(DuffingSystem(gamma=0.0), BOUNDS, side, vectorized=True, **settings)
    assert vectorised.fractions == serial.fractions


def test_invalid_number_of_initial_states_and_bounds():
    """Test that zero initial states or no bounds raise errors naming them."""
    with pytest.raises(ValidationError, match='n = 0'):
        basin_stability(DuffingSystem(gamma=0.0), BOUNDS, side, n=0)
    with pytest.raises(ValidationError, match='bounds of at least one initial state'):
        basin_stability(DuffingSystem(gamma=0.0), {}, side)


def test_counting_states_start_from_integers():
    """Test that the starts of a state whose domain holds integers are integers within the bounds."""
    system = CompositeSystem(elements=[ArrivalsElement('queue')])
    result = basin_stability(
        system,
        {'queue.count': (0, 5)},
        lambda trajectory: 'long' if trajectory.get_values('queue.count')[-1] > 5 else 'short',
        n=40,
        until=2,
        dt=1.0,
        random_state=0,
    )
    starts = [row['queue.count'] for row in result.rows]
    assert all(float(start).is_integer() and 0 <= start <= 5 for start in starts)
    assert {0, 5} <= set(starts)
    assert len(result.labels) == 40
