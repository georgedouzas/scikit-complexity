"""Test the `_base_spec` module of `skcomplex.core.specs`."""

import dataclasses

import pytest
from sage.all__sagemath_symbolics import var

from skcomplex.core.base import BaseConfig
from skcomplex.core.components.spaces import BallNeighborhood, ContinuousTime
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import (
    BaseSpec,
    HilbertSpec,
    MoveSpec,
    ObjectiveSpec,
    ObservableSpec,
    ParameterSpec,
    PhasesScheduleSpec,
    RequirementSpec,
    RoleSpec,
    SamplerSpec,
    ScheduleSpec,
    StateSpec,
    TopologyRequestSpec,
    VisualSpec,
)


def test_specs_are_frozen_records():
    """Test that the specs are frozen dataclasses that derive from the spec base, which is not an estimator."""
    assert not issubclass(BaseSpec, BaseConfig)
    assert not hasattr(BaseSpec, 'prepare')
    assert not hasattr(BaseSpec, 'run')
    for spec in (StateSpec, ParameterSpec, RoleSpec, ObservableSpec, HilbertSpec, TopologyRequestSpec):
        assert spec.__bases__ == (BaseSpec,)
        assert spec.__dataclass_params__.frozen
    assert PhasesScheduleSpec.__bases__ == (ScheduleSpec,)
    with pytest.raises(dataclasses.FrozenInstanceError):
        StateSpec().order = 2


def test_requirement_and_visual_specs_hold_their_fields():
    """Test that an empty requirement accepts any time and space and a visual keeps its options."""
    assert RequirementSpec() == RequirementSpec(time=(), space=())
    assert RequirementSpec(time=(ContinuousTime,)).time == (ContinuousTime,)
    assert VisualSpec('points', {'position': 'x'}).options == {'position': 'x'}
    assert VisualSpec().kind == 'scalars'


def test_monte_carlo_specs_check_their_kinds():
    """Test that an energy has nothing to check, and that moves, samplers, and schedules must be known."""
    spin = var('spin')
    assert ObjectiveSpec(-spin, BallNeighborhood(metric='manhattan')).expression == -spin
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
