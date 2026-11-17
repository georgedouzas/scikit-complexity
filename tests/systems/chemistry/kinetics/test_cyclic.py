"""Test the `_cyclic` module of `skcomplex.systems.chemistry.kinetics`."""

import numpy as np
from sklearn.base import clone

from skcomplex.core.components.interactions import LocalRuleInteraction
from skcomplex.core.components.spaces import RadiusNeighborhood
from skcomplex.core.systems import System
from skcomplex.systems.chemistry.kinetics import CyclicAdvanceInteraction, CyclicCellularAutomatonSystem

from .conftest import _run


def test_cyclic_cellular_automaton_advances_around_the_cycle():
    """Test that every cell of the cyclic automaton stays or moves to the following level."""
    trajectory = _run(CyclicCellularAutomatonSystem((20, 20), n=8), until=30, dt=1)
    levels = trajectory.get_lattice('cells.level')
    steps = np.mod(levels[1:] - levels[:-1], 8)
    assert set(np.unique(steps)) <= {0.0, 1.0}
    assert steps[-1].sum() > 0


def test_cyclic_advance_gives_the_levels_of_the_local_rule_law():
    """Test that the advance as an update with a neighbourhood gives the levels of the same rule as a `LocalRuleLaw`."""
    system = CyclicCellularAutomatonSystem((30, 30), n=8)
    cells = system.prepare().system_.elements[0]
    rule = LocalRuleInteraction(
        'advance',
        target='cells',
        state='level',
        rule=CyclicAdvanceInteraction._advance_cyclically,
        neighborhood=RadiusNeighborhood(1, metric='manhattan', center=True),
    )
    reference = System(elements=[clone(cells)], interactions=[rule]).prepare()
    np.testing.assert_array_equal(
        system.simulate(until=60, dt=1).get_values('cells.level'),
        reference.simulate(until=60, dt=1).get_values('cells.level'),
    )
