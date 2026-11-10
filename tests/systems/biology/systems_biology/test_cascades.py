"""Test the `_cascades` module of `skcomplex.systems.biology.systems_biology`."""

import numpy as np
import pytest

from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.systems.biology.systems_biology import HormoneCascadeAmplifierSystem

from ..conftest import _run_with_solver


def test_hormone_cascade_amplifier_multiplies_the_signal_by_the_gains_until_the_timer_ends_it():
    """Test that the cascade amplifies the G-protein signal by the product of the gains and the GTPase ends it."""
    runs = {}
    for k_gtp in (0.5, 0.05):
        trajectory = _run_with_solver(
            HormoneCascadeAmplifierSystem(k_gtp=k_gtp),
            60.0,
            0.05,
            ScipyODESolver(rtol=1e-9, atol=1e-12),
        )
        runs[k_gtp] = (trajectory.times, trajectory.get_values('g_protein.g'), trajectory.get_values('phosphorylase.x'))
    times, signal, output = runs[0.5]
    gain = np.trapezoid(output, times) / np.trapezoid(signal, times)
    assert gain == pytest.approx(10 * 50 * 2 * 10 * 10, rel=1e-3)
    late = times >= 20.0
    assert signal[late].max() < 1e-3 < runs[0.05][1][late].max()
    assert runs[0.05][2].max() > 2 * output.max()
