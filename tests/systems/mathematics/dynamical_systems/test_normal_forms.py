"""Test the `_normal_forms` module of `skcomplex.systems.mathematics.dynamical_systems`."""

import math

import pytest

from skcomplex.systems.mathematics.dynamical_systems import (
    CuspFormSystem,
    HopfFormSystem,
    SaddleNodeFormSystem,
    TakensBogdanovFormSystem,
)

from ..conftest import _run


def test_normal_forms_settle_on_their_attractors():
    """Test the stable node of the saddle-node, the Hopf cycle, the Takens-Bogdanov focus, and the cusp bistability."""
    assert _run(SaddleNodeFormSystem(0.0, mu=1.0), 20, dt=0.1).values('form.x')[-1] == pytest.approx(1.0, abs=1e-4)
    hopf = _run(HopfFormSystem(mu=0.25), 60, dt=0.1)
    assert math.hypot(hopf.values('form.x')[-1], hopf.values('form.y')[-1]) == pytest.approx(0.5, abs=1e-3)
    focus = (1 - math.sqrt(0.6)) / 2
    assert _run(TakensBogdanovFormSystem(), 200, dt=0.5).values('form.x')[-1] == pytest.approx(focus, abs=1e-3)
    assert _run(CuspFormSystem(0.5), 30, dt=0.5).values('form.x')[-1] == pytest.approx(1.0, abs=1e-4)
    assert _run(CuspFormSystem(-0.5), 30, dt=0.5).values('form.x')[-1] == pytest.approx(-1.0, abs=1e-4)
