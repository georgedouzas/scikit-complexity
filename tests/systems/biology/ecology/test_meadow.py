"""Test the `_meadow` module of `skcomplex.systems.biology.ecology`."""

import pytest

from skcomplex.core.components.systems import CompositeSystem
from skcomplex.systems.biology.ecology import MeadowCapacitySystem

from ..conftest import _run_until


@pytest.mark.parametrize('growth', [0.0, 0.2])
def test_meadow_runs_nested_in_a_farm(growth):
    """Test that the meadow placed as the subsystem `farm` runs as the meadow at the root."""
    root = _run_until(MeadowCapacitySystem(growth=growth, random_state=0), 25, random_state=0)
    nested = MeadowCapacitySystem(growth=growth, random_state=0).prepare().system_.set_params(label='farm')
    farm = _run_until(CompositeSystem(systems=[nested]), 25, random_state=0)
    assert farm.values('farm/cows.count').tolist() == root.values('cows.count').tolist()
    assert {interaction.pasture for interaction in nested.interactions} == {'pasture'}


def test_meadow_regrowth_raises_carrying_capacity():
    """Test that a meadow without regrowth starves its cows within the summer, and regrowth feeds them."""
    barren = _run_until(MeadowCapacitySystem(cows=80, growth=0.0, random_state=0), 25, random_state=0)
    growing = _run_until(MeadowCapacitySystem(cows=80, growth=0.2, random_state=0), 25, random_state=0)
    assert barren.values('cows.count')[-1] == 0
    assert growing.values('cows.count')[-1] == 80


def test_meadow_capacity_carries_or_loses_the_whole_herd():
    """Test that a meadow of 1600 patches carries 61 cows without regrowth and 80 with it, and loses larger herds."""

    def survivors(cows, growth):
        trajectory = _run_until(
            MeadowCapacitySystem(cows=cows, growth=growth, random_state=0),
            25,
            random_state=0,
        )
        return len(trajectory.agents('cows', len(trajectory.times) - 1).ids)

    assert survivors(61, 0.0) == 61
    assert survivors(80, 0.0) == 0
    assert survivors(80, 0.2) == 80
    assert survivors(160, 0.2) == 0
