"""Test the two-compartment pharmacokinetics against its bi-exponential solution and its mass balance."""

import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import ScipyODE
from skcomplex.models.biology.physiology import two_compartment_pharmacokinetics

PRECISE = ScipyODE(rtol=1e-10, atol=1e-14)


def _rates(k10, k12, k21):
    """Return the fast and slow rates `alpha` and `beta` of the bi-exponential."""
    total = k10 + k12 + k21
    root = np.sqrt(total**2 - 4 * k10 * k21)
    return (total + root) / 2, (total - root) / 2


def _central_concentration(times, c0, k10, k12, k21):
    """Return the bi-exponential central concentration after a bolus."""
    alpha, beta = _rates(k10, k12, k21)
    return c0 * ((alpha - k21) * np.exp(-alpha * times) + (k21 - beta) * np.exp(-beta * times)) / (alpha - beta)


def _total_drug(times, c0, k10, k12, k21):
    """Return the bolus minus the drug eliminated, `c0 - k10` times the integral of the central concentration."""
    alpha, beta = _rates(k10, k12, k21)
    fast = (alpha - k21) * (1 - np.exp(-alpha * times)) / alpha
    slow = (k21 - beta) * (1 - np.exp(-beta * times)) / beta
    return c0 - k10 * c0 * (fast + slow) / (alpha - beta)


def test_two_compartment_pharmacokinetics_matches_bi_exponential():
    """Test that one bolus gives the bi-exponential central concentration and that only elimination lowers the drug."""
    c0, k10, k12, k21 = 2.0, 0.082, 0.0475, 0.00125
    system = two_compartment_pharmacokinetics(c0=c0, k10=k10, k12=k12, k21=k21)
    trajectory = Simulation(system, dt=10.0, solver=PRECISE).run(until=3000.0).trajectory_
    times, central, peripheral = trajectory.times, trajectory['patient.Uc'], trajectory['patient.Up']
    assert np.max(np.abs(central / _central_concentration(times, c0, k10, k12, k21) - 1)) < 1e-6
    assert np.max(np.abs((central + peripheral) / _total_drug(times, c0, k10, k12, k21) - 1)) < 1e-6
    assert np.all(np.diff(central + peripheral) < 0)


def test_two_compartment_pharmacokinetics_conserves_drug_without_elimination():
    """Test that without elimination the drug only moves between compartments, to the split `k21 : k12`."""
    trajectory = Simulation(two_compartment_pharmacokinetics(c0=1.0, k10=0.0), dt=10.0, solver=PRECISE)
    trajectory = trajectory.run(until=3000.0).trajectory_
    total = trajectory['patient.Uc'] + trajectory['patient.Up']
    assert np.allclose(total, 1.0, rtol=1e-9)
    assert abs(trajectory['patient.Uc'][-1] - 0.00125 / (0.00125 + 0.0475)) < 1e-6
