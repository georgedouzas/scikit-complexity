"""Test the `_marangoni` module of `skcomplex.models.physics.fluids`."""

import numpy as np
import pytest
from scipy.stats import skew

from skcomplex.analysis import fields_of, pattern_wavelength
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import ETDRK4Solver
from skcomplex.models.physics.fluids import benard_marangoni


def test_benard_marangoni_forms_hexagons_with_a_free_surface():
    """Test that the thermocapillary term turns isotropic, skewed hexagons out of what are rolls without it."""
    patterns = {}
    for g in (1.0, 0.0):
        system = benard_marangoni((32, 32), length=8 * np.pi, g=g)
        field = fields_of(
            Simulation(system, dt=10.0, solver=ETDRK4Solver(dt=0.5)).run(until=200).trajectory_,
            'layer.u',
        )
        patterns[g] = (pattern_wavelength(field[-1], spacing=np.pi / 4), skew(field[-1].ravel()))
    (hexagons, hexagons_skew), (rolls, rolls_skew) = patterns[1.0], patterns[0.0]
    assert hexagons.wavenumber == pytest.approx(1.0)
    assert rolls.wavenumber == pytest.approx(1.0)
    assert hexagons.orientation < 0.2
    assert rolls.orientation > 0.9
    assert hexagons_skew > 0.5
    assert abs(rolls_skew) < 0.05
