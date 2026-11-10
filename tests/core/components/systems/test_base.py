"""Test the `_base` module of `skcomplex.core.components.systems`."""

from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet
from sklearn.base import clone

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.systems import BaseSystem, CompositeSystem, RunSettings
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import StateSpec, VisualSpec


class StockElement(Element):
    """Element with one nonnegative state."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        n: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.n = n


class GrowthSystem(BaseSystem[CompositeSystem]):
    """CompositeSystem of exponential growth whose model parameter is the rate."""

    until = 5.0

    def __init__(self, label: str = 'growth', r: float = 1.0) -> None:
        super().__init__(label)
        self.r = r

    def _build_system(self) -> CompositeSystem:
        growth = RateInteraction('growth', target='stock', state='n', rate=self.r)
        return CompositeSystem(self.label, elements=[StockElement('stock')], interactions=[growth])


def test_base_system_returns_its_tuned_run_settings():
    """Test that the run settings hold the class variables of the system, and `None` for those it leaves out."""
    settings = GrowthSystem().get_run_settings()
    assert settings.keys() == RunSettings.__annotations__.keys()
    assert (settings['until'], settings['dt'], settings['observables']) == (5.0, None, ())


def test_base_system_passes_model_parameters_down_to_its_parts():
    """Test that setting a model parameter of a system subclass changes the parts of the next `prepare`."""
    system = GrowthSystem(r=0.3).prepare()
    before = system.system_
    assert system.get_params() == {'label': 'growth', 'r': 0.3}
    assert before.interactions[0].rate == 0.3
    system.set_params(r=0.5).prepare()
    assert system.system_ is not before
    assert system.system_.interactions[0].rate == 0.5
    assert clone(system).get_params()['r'] == 0.5
    assert system.get_tags().laws == {'rate'}


class CountingSystem(GrowthSystem):
    """CompositeSystem of exponential growth that counts how often it builds its parts."""

    def __init__(self, label: str = 'growth', r: float = 1.0, weights: object = None) -> None:
        super().__init__(label, r)
        self.weights = weights
        self.n_builds = 0

    def _build_system(self) -> CompositeSystem:
        self.n_builds += 1
        return super()._build_system()


def test_base_system_builds_its_parts_once_per_set_of_parameters():
    """Test that tags, checks, and prepares reuse the built parts until a parameter changes by value."""
    system = CountingSystem(r=0.3, weights=np.ones(3))
    system.get_tags()
    system.prepare().prepare()
    system.get_tags()
    assert system.n_builds == 1
    system.set_params(r=0.3, weights=np.ones(3)).prepare()
    assert system.n_builds == 1
    system.set_params(weights=np.zeros(3)).prepare()
    assert system.n_builds == 2
    system.set_params(r=0.5).prepare()
    assert (system.n_builds, system.system_.interactions[0].rate) == (3, 0.5)
    system.get_tags()
    assert system.n_builds == 3


def test_base_system_rejects_parts_that_are_no_system():
    """Test that a system subclass whose hook returns no `CompositeSystem` fails to prepare."""

    class BrokenSystem(BaseSystem):
        """CompositeSystem whose hook returns an element."""

        def _build_system(self):
            return StockElement('stock')

    with pytest.raises(ValidationError, match='builds its parts as a CompositeSystem, not as StockElement'):
        BrokenSystem().prepare()


def test_base_system_rejects_parts_that_are_no_composite():
    """Test that a system subclass whose hook returns a system class, not its own composite, fails to prepare."""

    class NestingSystem(BaseSystem):
        """System whose hook returns another system class."""

        def _build_system(self):
            return GrowthSystem()

    with pytest.raises(ValidationError, match='builds its parts as a CompositeSystem, not as GrowthSystem'):
        NestingSystem().prepare()
