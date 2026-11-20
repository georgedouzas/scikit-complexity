"""Test the `base` module of `skcomplex`."""

import dataclasses
import inspect
from dataclasses import dataclass

import pytest
from sage.all__sagemath_symbolics import function, var
from sklearn.base import clone

from skcomplex.core import errors
from skcomplex.core.base import BaseConfig, BaseSpec
from skcomplex.core.components import Observable, RoleSpec
from skcomplex.core.components.base import BaseComponent
from skcomplex.core.components.elements import Element, VisualSpec
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import (
    MoveSpec,
    ObjectiveSpec,
    RateLaw,
    RequirementSpec,
    SamplerSpec,
    TopologyRequestSpec,
)
from skcomplex.core.components.laws.utils import series
from skcomplex.core.components.spaces import LatticeSpace, RadiusNeighborhood, ScheduleSpec
from skcomplex.core.engine.compile import PhasesScheduleSpec
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System
from skcomplex.systems.physics.common.quantum import HilbertSpec


class ScalingConfig(BaseConfig):
    """Scale its input by a factor fixed when it is prepared."""

    def __init__(self, factor=2):
        self.factor = factor

    def _prepare(self, context):
        if self.factor == 0:
            msg = 'factor'
            raise ValueError(msg)
        self.offset_ = 0 if context is None else context

    def _run(self, inputs):
        self.scaled_ = self.factor * inputs + self.offset_

    def _release(self):
        self.released_ = True


def test_verbs_prepare_and_run():
    """Test that preparing against a context and running act through the hooks."""
    config = ScalingConfig(3).prepare(1)
    assert config.offset_ == 1
    assert config.run(2) is config
    assert config.scaled_ == 7


def test_prepare_checks_before_it_derives():
    """Test that preparing checks the settings at the start of `_prepare`, before it derives anything."""
    config = ScalingConfig(0)
    with pytest.raises(ValueError, match='factor'):
        config.prepare()
    assert not hasattr(config, 'offset_')


def test_prepare_validates_a_setting_changed_after_preparing():
    """Test that preparing again validates a parameter changed with `set_params`."""
    config = ScalingConfig(3).prepare()
    with pytest.raises(ValueError, match='factor'):
        config.set_params(factor=0).prepare()


def test_no_public_check_remains():
    """Test that the bases have no public `check`, no `_validated`, and the error for an unchecked object is gone."""
    assert not any(hasattr(base, 'check') for base in (BaseConfig, BaseSpec, BaseComponent))
    assert not any(hasattr(base, '_validated') for base in (BaseConfig, BaseSpec, BaseComponent))
    assert not hasattr(BaseConfig, '_check')
    assert not hasattr(BaseComponent, '_check')
    assert not hasattr(BaseSpec, 'prepare')
    assert not hasattr(BaseSpec, 'run')
    assert not hasattr(errors, 'NotCheckedError')


def test_bases_list_their_methods():
    """Test that an engine class prepares and runs, a component only prepares, and a spec has no public method."""

    def get_public_methods(klass):
        return {name for name, value in inspect.getmembers(klass) if not name.startswith('_') and callable(value)}

    params = {'get_params', 'set_params', 'get_metadata_routing'}
    assert get_public_methods(BaseConfig) == params | {'prepare', 'run'}
    assert get_public_methods(BaseComponent) == params | {'get_tags', 'prepare'}
    assert get_public_methods(BaseSpec) == set()
    assert not issubclass(BaseComponent, BaseConfig)


def test_bad_component_fails_before_it_is_prepared():
    """Test that a component with a bad label raises from `prepare` of its system, before it sets any attribute."""
    interaction = Interaction(1, roles={'target': 'stock'}, laws=lambda s: [RateLaw(s.target.n.diff(s.t), 1.0)])
    system = System(elements=[Element('stock', values={'n': 0.0})], interactions=[interaction])
    with pytest.raises(ValidationError, match='label'):
        system.prepare()
    assert not [name for name in vars(interaction) if name.endswith('_')]


def test_final_run_releases():
    """Test that only the final run releases what preparing acquired."""
    config = ScalingConfig().prepare()
    config.run(1)
    assert not hasattr(config, 'released_')
    config.run(1, final=True)
    assert config.released_


def test_verbs_return_the_object():
    """Test that checking, preparing, and running return the object, as in scikit-learn."""
    config = ScalingConfig()
    assert config.prepare() is config
    assert config.prepare() is config
    assert config.run(1) is config


def test_settings_follow_scikit_learn():
    """Test that the stored settings work with `get_params`, `set_params`, and `clone`."""
    config = ScalingConfig(4)
    assert config.get_params() == {'factor': 4}
    assert clone(config.set_params(factor=5)).factor == 5


def test_a_class_without_a_hook_cannot_be_created():
    """Test that a configurable class that implements `_prepare` but not `_run` cannot be created."""

    class IdleConfig(BaseConfig):
        def _prepare(self, context):
            pass

    with pytest.raises(TypeError, match='_run'):
        IdleConfig()


def test_a_spec_without_hooks_is_a_value():
    """Test that a spec without hooks has neither `prepare` nor `run`, and its owner checks it through `_check`."""

    @dataclass(frozen=True)
    class SpanSpec(BaseSpec):
        start: float
        end: float

    span = SpanSpec(0.0, 1.0)
    assert not isinstance(span, BaseConfig)
    assert not hasattr(span, 'prepare')
    assert not hasattr(span, 'run')
    assert span._check() is None


def test_clone_shares_the_symbolic_values_of_sage():
    """Test that a clone keeps the conditions of a lookup table and the symbolic functions of a law as they are."""
    x = var('x')
    deaths = series(x, (0.0, 0.25, 0.5, 1.0), (30.0, 3.0, 2.0, 1.0))
    speed = function('speed')
    law = RateLaw(x, deaths + speed(x), label='decline')
    cloned = clone(law)
    assert cloned is not law
    assert cloned.value is law.value
    assert str(cloned.value) == str(deaths + speed(x))
    assert '< 0.25' in str(cloned.value)
    assert any(item.operator() is speed for item in cloned.value.operands())


def test_clone_copies_mutable_settings():
    """Test that a clone copies a mutable setting, such as a list, and shares a frozen spec."""

    @dataclass(frozen=True)
    class WindowSpec(BaseSpec):
        width: float = 1.0

    class WindowConfig(BaseConfig):
        def __init__(self, sizes=None, window=None):
            self.sizes = sizes
            self.window = window

        def _prepare(self, context):
            pass

        def _run(self, inputs):
            pass

    config = WindowConfig(sizes=[1, 2], window=WindowSpec())
    cloned = clone(config)
    assert cloned.sizes == [1, 2]
    assert cloned.sizes is not config.sizes
    assert cloned.window is config.window


def test_specs_are_frozen_records():
    """Test that the specs are frozen dataclasses that derive from the spec base, which is not an estimator."""
    assert not issubclass(BaseSpec, BaseConfig)
    assert not hasattr(BaseSpec, 'prepare')
    assert not hasattr(BaseSpec, 'run')
    for spec in (RoleSpec, VisualSpec, Observable, HilbertSpec, TopologyRequestSpec):
        assert spec.__bases__ == (BaseSpec,)
        assert spec.__dataclass_params__.frozen
    assert PhasesScheduleSpec.__bases__ == (ScheduleSpec,)
    with pytest.raises(dataclasses.FrozenInstanceError):
        RoleSpec().variadic = True


def test_requirement_and_visual_specs_hold_their_fields():
    """Test that an empty requirement accepts any time and space and a visual keeps its options."""
    assert RequirementSpec() == RequirementSpec(time=(), space=())
    assert RequirementSpec(space=(LatticeSpace,)).space == (LatticeSpace,)
    assert VisualSpec('points', {'position': 'x'}).options == {'position': 'x'}
    assert VisualSpec().kind == 'scalars'


def test_monte_carlo_specs_check_their_kinds():
    """Test that an energy has nothing to check, and that moves, samplers, and schedules must be known."""
    spin = var('spin')
    assert ObjectiveSpec(-spin, RadiusNeighborhood(1, metric='manhattan')).expression == -spin
    assert MoveSpec(spin, 'rotate', width=0.5).width == 0.5
    assert MoveSpec(spin, 'gaussian', width=0.5).kind == 'gaussian'
    assert MoveSpec(spin, 'function', function=lambda values, random: values).kind == 'function'
    with pytest.raises(ValidationError, match='Unknown move'):
        MoveSpec(spin, 'swap')._check()
    with pytest.raises(ValidationError, match='callable function'):
        MoveSpec(spin, 'function')._check()
    with pytest.raises(ValidationError, match='Unknown sampler'):
        SamplerSpec('wolff')._check()
    with pytest.raises(ValidationError, match='Unknown Monte Carlo schedule'):
        SamplerSpec(schedule='parallel')._check()
