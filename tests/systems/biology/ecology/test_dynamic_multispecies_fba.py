"""Test the `_dynamic_multispecies_fba` module of `skcomplex.systems.biology.ecology`."""

import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.ecology import DynamicMultispeciesFBASystem


def test_competition_raises_the_share_of_the_faster_strain():
    """Test the rising share of the faster strain, and that the glucose consumed follows the uptake of both strains."""
    trajectory = simulate(DynamicMultispeciesFBASystem())
    times = trajectory.times
    fast, slow = trajectory.values('culture/biomass_fast.x'), trajectory.values('culture/biomass_slow.x')
    glucose = trajectory.values('culture/glucose.c')
    share = fast / (fast + slow)
    assert np.all(np.diff(share) >= -1e-12)
    assert share[-1] > share[0]
    consumed = 0.0
    for name, amount in (('fast', fast), ('slow', slow)):
        exchange = -trajectory.values(f'fba/cell_{name}.v_EX_glc__D_e')
        consumed += np.sum(exchange[:-1] * 0.5 * (amount[:-1] + amount[1:]) * np.diff(times))
    assert glucose[0] - glucose[-1] == pytest.approx(consumed, rel=0.01)
