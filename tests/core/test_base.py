"""Test the `base` module of `skcomplex`."""

from dataclasses import dataclass

import pytest
from sage.all__sagemath_symbolics import function, var
from sklearn.base import clone

from skcomplex.core import errors
from skcomplex.core.base import BaseConfig
from skcomplex.core.components.base import BaseComponent
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import series
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import BaseSpec
from skcomplex.core.systems import System


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
