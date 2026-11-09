"""Test the `_metabolic_network` module of `skcomplex.systems.biology.common.metabolism`."""

import pytest

from skcomplex.core.components.spaces import StaticTime
from skcomplex.core.components.systems import System
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.systems.biology.common.metabolism import (
    ReactionSpec,
    load_network,
    metabolic_network_type,
)
from skcomplex.systems.biology.systems_biology import FluxBalanceSystem, MetabolicPerturbationsSystem


def _core_system(**values):
    """Return the system of the E. coli core system."""
    return System(elements=[load_network('textbook', 'cell').set_params(**values)], time=StaticTime())


def _objective(system):
    """Return the objective of the only program of a static system."""
    trajectory = simulate(system, until=0.0)
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
        PassCompiler().prepare(System(elements=[element_type('cell')], time=StaticTime())).run()


def test_network_keeps_its_reference_values():
    """Test the toy and core networks against the values recorded before they moved to declared states."""
    trajectory = simulate(FluxBalanceSystem(), until=0.0)
    assert float(trajectory.values('cell.v_vbio')[0]) == pytest.approx(50.0, rel=1e-9)
    assert _objective(MetabolicPerturbationsSystem(genes=('b4025',)).prepare().system_) == pytest.approx(
        0.8631595522084183,
        rel=1e-9,
    )
