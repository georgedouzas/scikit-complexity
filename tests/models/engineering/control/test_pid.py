"""Test the `_pid` module of `skcomplex.models.engineering.control`."""

import importlib
import pkgutil

import skcomplex.core
from skcomplex.models.engineering.control import PIDInteraction


def test_pid_lives_only_in_the_control_models():
    """Test that no package of the core exports `PIDInteraction`, which the control models provide."""
    for module in pkgutil.walk_packages(skcomplex.core.__path__, 'skcomplex.core.'):
        if module.ispkg:
            exported = getattr(importlib.import_module(module.name), '__all__', [])
            assert 'PIDInteraction' not in exported, module.name
    assert PIDInteraction('pid', plant='cart', integrator='memory', control='force', measure='theta').kp == 1.0
