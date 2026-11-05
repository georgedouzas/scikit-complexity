"""Test the `_integrator` module of `skcomplex.systems.engineering.control`."""

import importlib
import pkgutil

import skcomplex.core.components
from skcomplex.systems.engineering.control import IntegratorElement


def test_integrator_lives_only_in_the_control_systems():
    """Test that no package of the core exports `IntegratorElement`, which the control systems provide."""
    for module in pkgutil.walk_packages(skcomplex.core.components.__path__, 'skcomplex.core.components.'):
        if module.ispkg:
            exported = getattr(importlib.import_module(module.name), '__all__', [])
            assert 'IntegratorElement' not in exported, module.name
    assert IntegratorElement('memory').value == 0.0
