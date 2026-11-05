"""Test the `_evaporation` module of `skcomplex.systems.biology.common.stigmergy`."""

from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.components.elements import Element, FieldElement
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import ParameterSpec, StateSpec
from skcomplex.systems.biology.common.stigmergy import EvaporationInteraction


class MarkElement(Element):
    """Mark `level` of a cell, which evaporates at the rate `rate`."""

    def __init__(
        self,
        label: str,
        level: Annotated[float, StateSpec()] = 1.0,
        rate: Annotated[float, ParameterSpec()] = 0.5,
    ) -> None:
        super().__init__(label)
        self.level = level
        self.rate = rate


def test_evaporation_decays_every_cell_at_its_own_rate():
    """Test that the mark of every cell of a field decays exponentially at the rate the cell holds."""
    rates = np.array([0.1, 0.5, 1.0, 2.0])
    marks = FieldElement('marks', element=MarkElement, space=LatticeSpace((4,)), values={'rate': rates})
    evaporation = EvaporationInteraction('evaporation', target='marks', state='level', evaporation='rate')
    system = System(elements=[marks], interactions=[evaporation])
    trajectory = Simulation(system, dt=0.001).prepare().run(until=1.0).trajectory_
    np.testing.assert_allclose(np.asarray(trajectory.values('marks.level'))[-1], np.exp(-rates), rtol=1e-3)


def test_evaporation_rejects_a_rate_that_is_no_name():
    """Test that the name of the rate of evaporation must be a name."""
    with pytest.raises(ValidationError, match='evaporation'):
        EvaporationInteraction('evaporation', target='marks', evaporation='1 rate').prepare()
