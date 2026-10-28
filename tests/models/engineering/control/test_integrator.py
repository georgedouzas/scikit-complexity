"""Test the `_integrator` module of `skcomplex.models.engineering.control`."""

import importlib
import pkgutil

import skcomplex.core
from skcomplex.models.engineering.control import IntegratorElement


def test_integrator_lives_only_in_the_control_models():
    """Test that no package of the core exports `IntegratorElement`, which the control models provide."""
    for module in pkgutil.walk_packages(skcomplex.core.__path__, 'skcomplex.core.'):
        if module.ispkg:
            exported = getattr(importlib.import_module(module.name), '__all__', [])
            assert 'IntegratorElement' not in exported, module.name
    assert IntegratorElement('memory').value == 0.0
