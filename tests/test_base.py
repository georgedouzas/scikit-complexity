"""Test the `base` module of `skcomplex`."""

import importlib
import inspect
import pkgutil

import pytest
from sklearn.base import clone

import skcomplex
from skcomplex.base import BaseConfig
from skcomplex.errors import NotCheckedError


class ScalingConfig(BaseConfig):
    """Scale its input by a factor fixed when it is prepared."""

    def __init__(self, factor=2):
        self.factor = factor

    def _check(self):
        if self.factor == 0:
            msg = 'factor'
            raise ValueError(msg)
        return self

    def _prepare(self, context):
        self.offset_ = 0 if context is None else context
        return self

    def _run(self, inputs):
        return self.factor * inputs + self.offset_

    def _release(self):
        self.released_ = True


def test_verbs_check_prepare_and_run():
    """Test that checking, preparing against a context, and running act through the hooks."""
    config = ScalingConfig(3).check().prepare(1)
    assert config.offset_ == 1
    assert config.run(2) == 7


def test_check_runs_the_check_hook():
    """Test that checking calls the hook that validates the settings."""
    with pytest.raises(ValueError, match='factor'):
        ScalingConfig(0).check()


def test_config_must_be_checked_before_it_is_prepared_or_run():
    """Test that preparing or running an object that was not checked raises an error."""
    with pytest.raises(NotCheckedError):
        ScalingConfig().prepare()
    with pytest.raises(NotCheckedError):
        ScalingConfig().run(1)


def test_final_run_releases():
    """Test that only the final run releases what preparing acquired."""
    config = ScalingConfig().check().prepare()
    config.run(1)
    assert not hasattr(config, 'released_')
    config.run(1, final=True)
    assert config.released_


def test_hooks_of_the_root_do_nothing():
    """Test that a child overriding no hook checks, prepares, and runs to nothing."""

    class EmptyConfig(BaseConfig):
        pass

    assert EmptyConfig().check().prepare().run() is None


def test_settings_follow_scikit_learn():
    """Test that the stored settings work with `get_params`, `set_params`, and `clone`."""
    config = ScalingConfig(4)
    assert config.get_params() == {'factor': 4}
    assert clone(config.set_params(factor=5)).factor == 5


def test_every_base_derives_from_the_root():
    """Test that every class named `Base*` outside the models derives from `BaseConfig`."""
    outside = []
    for module in pkgutil.walk_packages(skcomplex.__path__, 'skcomplex.'):
        if module.name.startswith('skcomplex.models'):
            continue
        try:
            loaded = importlib.import_module(module.name)
        except ImportError:
            continue
        for name, value in vars(loaded).items():
            if (
                inspect.isclass(value)
                and value.__module__ == loaded.__name__
                and name.lstrip('_').startswith('Base')
                and not issubclass(value, BaseConfig | BaseException)
            ):
                outside.append(f'{loaded.__name__}.{name}')
    assert outside == []
