"""Test the `_map` module of `skcomplex.core.interactions`."""

from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.interactions import MapInteraction
from skcomplex.core.laws.utils import normal
from skcomplex.core.spaces import DiscreteTime
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.errors import ValidationError
from skcomplex.specs import StateSpec, VisualSpec

RATE = 3.2


class ValueElement(Element):
    """Element with a value `x`."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 0.1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x


def _values(elements, interactions, until, path='value.x'):
    """Return the values of a state at every step of a discrete-time system."""
    system = System(elements=elements, interactions=interactions, time=DiscreteTime())
    simulation = Simulation(system, random_state=0).check().prepare()
    return simulation.run(until=until).trajectory_.values(path)


def test_logistic_map_settles_on_the_closed_form_two_cycle():
    """Test that the logistic map at `r = 3.2` settles on its closed-form 2-cycle."""
    logistic = MapInteraction('map', target='value', state='x', function=lambda s: RATE * s.target.x * (1 - s.target.x))
    values = _values([ValueElement('value')], [logistic], 400)
    root = np.sqrt((RATE + 1) * (RATE - 3))
    cycle = sorted([(RATE + 1 - root) / (2 * RATE), (RATE + 1 + root) / (2 * RATE)])
    np.testing.assert_allclose(sorted(values[-2:]), cycle, rtol=1e-9)
    assert values[-1] != pytest.approx(values[-2])


def test_increment_form_adds_the_function_to_every_member():
    """Test that the increment form adds the function to the state of every member of a group."""
    group = GroupElement('value', element=ValueElement, size=3, values={'x': [0.0, 1.0, 2.0]})
    increment = MapInteraction('step', target='value', state='x', function=lambda s: 0.5, form='increment')
    values = _values([group], [increment], 4)
    np.testing.assert_allclose(values[-1], [2.0, 3.0, 4.0])


def test_map_reads_its_sources():
    """Test that a map reads the states of the elements its sources bind."""
    copy = MapInteraction('copy', target='value', state='x', function=lambda s: 2 * s.sources[0].x, sources=['input'])
    values = _values([ValueElement('value', x=0.0), ValueElement('input', x=0.25)], [copy], 2)
    assert values.tolist() == [0.0, 0.5, 0.5]


def test_map_draws_one_value_per_member():
    """Test that a map with a draw gives every member of a group its own value."""
    group = GroupElement('value', element=ValueElement, size=200, values={'x': 0.0})
    noise = MapInteraction('noise', target='value', state='x', function=lambda s: normal('kick'), form='increment')
    values = _values([group], [noise], 100)
    assert values[-1].var() == pytest.approx(100.0, rel=0.25)


@pytest.mark.parametrize(
    ('arguments', 'match'),
    [({'form': 'assign', 'function': lambda s: 0.0}, 'form of map'), ({'function': 1.0}, 'callable')],
)
def test_check_rejects_unknown_forms_and_functions(arguments, match):
    """Test that an unknown form or a function that is not callable raise errors."""
    with pytest.raises(ValidationError, match=match):
        MapInteraction('map', target='value', state='x', **arguments).check()
