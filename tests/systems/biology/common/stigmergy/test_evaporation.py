"""Test the `_evaporation` module of `skcomplex.systems.biology.common.stigmergy`."""

import numpy as np
import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System
from skcomplex.systems.biology.common.stigmergy import EvaporationInteraction


def test_evaporation_decays_every_cell_at_its_own_rate():
    """Test that the mark of every cell of a field decays exponentially at the rate the cell holds."""
    rates = np.array([0.1, 0.5, 1.0, 2.0])
    marks = Element('marks', values={'level': np.ones(4), 'rate': rates}, space=LatticeSpace((4,)))
    evaporation = EvaporationInteraction('evaporation', target='marks', state='level', evaporation='rate')
    system = System(elements=[marks], interactions=[evaporation])
    trajectory = system.prepare().simulate(dt=0.001, until=1.0)
    np.testing.assert_allclose(np.asarray(trajectory.get_values('marks.level'))[-1], np.exp(-rates), rtol=1e-3)


def test_evaporation_rejects_a_rate_that_is_no_name():
    """Test that the name of the rate of evaporation must be a name."""
    with pytest.raises(ValidationError, match='evaporation'):
        EvaporationInteraction('evaporation', target='marks', evaporation='1 rate').prepare()
