"""Test the `_lookups` module of `skcomplex.core.components.laws.utils`."""

from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import lookup
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import ObservableSpec, StateSpec, VisualSpec


def _fertility(s):
    """Return the fertility, a table of food per person."""
    return lookup(s.food, [(0, 0.0), (1, 2.0), (2, 3.0)])


class FertilityElement(Element):
    """Fertility element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        food: Annotated[object, StateSpec()] = -1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.food = food

    def _get_laws(self, s):
        """Return the laws of the fertility element."""
        return [RateLaw(s.food, 1.0)]


def test_table_function_interpolates_and_stays_flat():
    """Test that the fertility, observed as an expression of the element, interpolates the table and holds its ends."""
    fertility = ObservableSpec('fertility', _fertility, element='f')
    trajectory = Simulation(System(elements=[FertilityElement('f')]), dt=0.5, observables=[fertility]).prepare().run(4)
    trajectory = trajectory.trajectory_
    food = trajectory.values('f.food')
    expected = np.interp(food, [0, 1, 2], [0.0, 2.0, 3.0])
    assert trajectory.values('f.fertility') == pytest.approx(expected)
    assert trajectory.values('f.fertility')[0] == 0.0
    assert trajectory.values('f.fertility')[-1] == 3.0
