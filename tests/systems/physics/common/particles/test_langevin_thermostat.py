"""Test the `_langevin_thermostat` module of `skcomplex.systems.physics.common.particles`."""

import pytest

from skcomplex.core.engine.solvers import EulerMaruyamaSolver, LammpsSolver
from skcomplex.core.systems import System
from skcomplex.systems.physics.common.particles import LAMMPS_FIXES, LAMMPS_PAIRS, LangevinThermostatInteraction
from skcomplex.systems.physics.statistical_mechanics import IdealGasSystem


@pytest.mark.parametrize(
    'solver',
    [EulerMaruyamaSolver(dt=0.005), LammpsSolver(dt=0.01, pairs=LAMMPS_PAIRS, fixes=LAMMPS_FIXES)],
    ids=['native', 'lammps'],
)
def test_langevin_thermostat_reaches_its_temperature(solver):
    """Test that a gas at temperature 0.5 under a thermostat at 2 approaches 2 within 5 percent."""
    gas = IdealGasSystem(2000, kT=0.5, size=40.0, random_state=0).prepare().system_
    system = System(
        elements=gas.elements,
        interactions=[*gas.interactions, LangevinThermostatInteraction('bath', target='gas', temperature=2.0)],
        space=gas.space,
        solver=solver,
    )
    trajectory = system.prepare().simulate(until=20, dt=1.0, random_state=0)
    temperature = (trajectory.get_values("gas.x'")[-5:] ** 2).sum(axis=2).mean() / 2
    assert temperature == pytest.approx(2.0, rel=0.05)
