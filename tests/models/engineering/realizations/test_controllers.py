"""Test the `_controllers` module of `skcomplex.models.engineering.realizations`."""

import importlib
import pkgutil

import skcomplex.core
from skcomplex.models.engineering.realizations import IntegratorElement, PIDInteraction, StateFeedbackInteraction


def test_controllers_live_only_in_the_engineering_realizations():
    """Test that no package of the core exports a controller, which the engineering realizations provide."""
    names = {'PIDInteraction', 'StateFeedbackInteraction', 'IntegratorElement'}
    for module in pkgutil.walk_packages(skcomplex.core.__path__, 'skcomplex.core.'):
        if module.ispkg:
            assert not names & set(getattr(importlib.import_module(module.name), '__all__', [])), module.name
    assert issubclass(PIDInteraction, object)
    assert StateFeedbackInteraction('feedback', plant='plant').states == ()
    assert IntegratorElement('memory').value == 0.0
