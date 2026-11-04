"""Test the `_metabolic_network` module of `skcomplex.models.biology.common.metabolism`."""

import pytest

from skcomplex.core.spaces import StaticTime
from skcomplex.core.system import System
from skcomplex.engine.compile import compile_system
from skcomplex.engine.runtime import Simulation
from skcomplex.errors import ValidationError
from skcomplex.models.biology.common.metabolism import (
    ReactionSpec,
    load_network,
    metabolic_network_type,
)


def _core_system(**values):
    """Return the system of the E. coli core model."""
    network = load_network('textbook', 'cell')
    return System(elements=[type(network)('cell', **values)], time=StaticTime())


def _objective(system):
    """Return the objective of the only program of a static system."""
    trajectory = Simulation(system).check().prepare().run(0.0).trajectory_
    return float(trajectory.values('cell.objective_0')[0])


def test_network_rule_gates_the_bounds_of_a_reaction():
    """Test that a rule closes its reaction only when it evaluates to false."""
    reactions = {
        'in': ReactionSpec({'A': 1.0}, upper=5.0),
        'use': ReactionSpec({'A': -1.0}, rule='g1 and (g2 or g3)'),
    }
    element_type = metabolic_network_type('GatedElement', reactions, {'use': 1.0})
    values = [(1.0, 0.0, 1.0, 5.0), (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 1.0, 0.0)]
    for first, second, third, expected in values:
        element = element_type('cell', g_g1=first, g_g2=second, g_g3=third)
        assert _objective(System(elements=[element], time=StaticTime())) == pytest.approx(expected)


def test_network_rejects_a_rule_with_another_operator():
    """Test that a rule with an operator other than `and` and `or` raises an error that names the element."""
    element_type = metabolic_network_type('BadRuleElement', {'r': ReactionSpec({'A': 1.0}, rule='a and not b')})
    with pytest.raises(ValidationError, match='gene rule of cell'):
        compile_system(System(elements=[element_type('cell')], time=StaticTime()))
