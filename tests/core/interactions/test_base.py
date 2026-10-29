"""Test the `_base` module of `skcomplex.core.interactions`."""

from types import SimpleNamespace
from typing import Annotated

import pytest

from skcomplex.core.interactions import BaseInteraction
from skcomplex.errors import ValidationError
from skcomplex.specs import ParameterSpec, RoleSpec, StateSpec


class GrowthInteraction(BaseInteraction):
    """Interaction with a role and a parameter."""

    def __init__(self, label, target: Annotated[str, RoleSpec()], r: Annotated[float, ParameterSpec()] = 1.0):
        super().__init__(label)
        self.target = target
        self.r = r

    def _get_laws(self, namespace):
        """Return no laws."""
        return []


class WeightedInteraction(BaseInteraction):
    """Interaction that declares a state, which interactions do not hold."""

    def __init__(self, label, weight: Annotated[float, StateSpec()] = 1.0):
        super().__init__(label)
        self.weight = weight

    def _get_laws(self, namespace):
        """Return no laws."""
        return []


class PolicyInteraction(BaseInteraction):
    """Interaction that declares an adaptive parameter, which interactions do not hold."""

    def __init__(self, label, rate: Annotated[float, ParameterSpec(adaptive=True)] = 1.0):
        super().__init__(label)
        self.rate = rate

    def _get_laws(self, namespace):
        """Return no laws."""
        return []


def test_interaction_stores_its_roles_and_parameters():
    """Test that an interaction stores its arguments, which scikit-learn lists."""
    assert GrowthInteraction('growth', target='rabbits', r=0.5).get_params() == {
        'label': 'growth',
        'r': 0.5,
        'target': 'rabbits',
    }


def test_interaction_prepares_into_itself():
    """Test that a checked interaction has nothing to prepare and runs into its laws."""
    growth = GrowthInteraction('growth', target='rabbits').check()
    assert growth.prepare() is growth
    assert growth.run(SimpleNamespace()).laws_ == []


def test_state_on_an_interaction_is_rejected():
    """Test that checking an interaction that declares a state raises an error that suggests a group on edges."""
    with pytest.raises(ValidationError, match='on="edges"'):
        WeightedInteraction('w').check()


def test_adaptive_parameter_on_an_interaction_is_rejected():
    """Test that checking an interaction that declares an adaptive parameter raises an error that names it."""
    with pytest.raises(ValidationError, match='adaptive parameters rate'):
        PolicyInteraction('policy').check()
