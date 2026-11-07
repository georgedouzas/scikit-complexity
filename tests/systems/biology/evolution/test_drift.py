"""Test the `_drift` module of `skcomplex.systems.biology.evolution`."""

import numpy as np
import pytest

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import EulerMaruyamaSolver
from skcomplex.systems.biology.evolution import kimura_diffusion


@pytest.mark.parametrize('Ns', [0.0, 3.0])
def test_alleles_fix_with_kimuras_probability(Ns):
    """Test that the share of fixed replicates is `(1 - exp(-N s x)) / (1 - exp(-N s))`, and `x` when neutral."""
    system = kimura_diffusion(0.2, Ns=Ns, replicates=2000)
    solver = EulerMaruyamaSolver(dt=0.002)
    final = Simulation(system, dt=1.0, solver=solver, random_state=0).prepare().run(until=12).trajectory_
    frequencies = final.values('alleles.x')[-1]
    expected = 0.2 if Ns == 0 else (1 - np.exp(-Ns * 0.2)) / (1 - np.exp(-Ns))
    assert np.all((frequencies == 0) | (frequencies == 1) | (np.abs(frequencies - 0.5) < 0.5))
    assert (frequencies > 0.999).mean() == pytest.approx(expected, abs=0.03)
    assert ((frequencies > 1e-3) & (frequencies < 1 - 1e-3)).mean() < 0.01
