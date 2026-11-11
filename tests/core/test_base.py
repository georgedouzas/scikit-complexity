"""Test the `base` module of `skcomplex`."""

from dataclasses import dataclass

import pytest
from sklearn.base import clone

from skcomplex.core import errors
from skcomplex.core.base import BaseConfig
from skcomplex.core.components.base import BaseComponent
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import BaseSpec


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


def test_bad_part_fails_before_it_is_prepared():
    """Test that a part with a bad label raises from `prepare` of its system, before it sets any attribute."""
    interaction = RateInteraction(1, target='stock', state='n', rate=1.0)
    system = CompositeSystem(elements=[Element('stock')], interactions=[interaction])
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
