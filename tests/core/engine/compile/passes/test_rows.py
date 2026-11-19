"""Test the `_rows` pass of `skcomplex.core.engine.compile.passes`."""

import numpy as np
import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import ConstraintLaw, ObjectiveLaw
from skcomplex.core.systems import System


def _get_program(s, scale):
    """Return the largest `v` of every cell below its capacity times a scale."""
    v = s.cell.v
    return [ObjectiveLaw(-v, [v]), ConstraintLaw(v <= s.cell.cap * scale(s))]


@pytest.mark.parametrize(
    ('scale', 'expected'),
    [
        (lambda s: 1, [1.0, 2.0, 3.0]),
        (lambda s: s.mass.x, [2.0, 1.0, 6.0]),
        (lambda s: s.mass.x * s.weight.w, [6.0, 3.0, 18.0]),
    ],
)
def test_linear_program_of_an_interaction_bound_to_two_groups_runs_per_member(scale, expected):
    """Test that a linear program bound to two groups sets every member, reading the other group row by row."""
    cells = Element('cells', values={'v': np.zeros(3), 'cap': np.array([1.0, 2.0, 3.0])}, size=3)
    mass = Element('mass', values={'x': np.array([2.0, 0.5, 2.0])}, size=3)
    weight = Element('weight', values={'w': 3.0})
    program = Interaction(
        'program',
        roles={'cell': 'cells', 'mass': 'mass', 'weight': 'weight'},
        laws=lambda s: _get_program(s, scale),
    )
    system = System(elements=[cells, mass, weight], interactions=[program])
    trajectory = system.prepare().simulate(until=0.0)
    np.testing.assert_allclose(trajectory.get_values('cells.v')[0], expected)
    assert np.shape(trajectory.get_values('cells.objective_0')[0]) == (3,)
