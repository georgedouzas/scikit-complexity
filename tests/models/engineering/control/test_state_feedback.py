"""Test the `_state_feedback` module of `skcomplex.models.engineering.control`."""

import importlib
import pkgutil

import skcomplex.core
from skcomplex.models.engineering.control import StateFeedbackInteraction


def test_state_feedback_lives_only_in_the_control_models():
    """Test that no package of the core exports `StateFeedbackInteraction`, which the control models provide."""
    for module in pkgutil.walk_packages(skcomplex.core.__path__, 'skcomplex.core.'):
        if module.ispkg:
            exported = getattr(importlib.import_module(module.name), '__all__', [])
            assert 'StateFeedbackInteraction' not in exported, module.name
    assert StateFeedbackInteraction('feedback', plant='plant').states == ()
