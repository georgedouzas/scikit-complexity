"""Test the `_langevin_thermostat` module of `skcomplex.systems.physics.common.particles`."""

import pytest

from skcomplex.core.components.systems import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import EulerMaruyamaSolver, LammpsSolver
from skcomplex.systems.physics.common.particles import LAMMPS_FIXES, LAMMPS_PAIRS, LangevinThermostatInteraction
from skcomplex.systems.physics.statistical_mechanics import IdealGasSystem


@pytest.mark.parametrize(
    ('solver', 'dt'),
    [(EulerMaruyamaSolver(dt=0.005), 1.0), (LammpsSolver(dt=0.01, pairs=LAMMPS_PAIRS, fixes=LAMMPS_FIXES), 1.0)],
    ids=['native', 'lammps'],
)
def test_langevin_thermostat_reaches_its_temperature(solver, dt):
    """Test that a gas at temperature 0.5 under a thermostat at 2 approaches 2 within 5 percent."""
    gas = IdealGasSystem(2000, kT=0.5, size=40.0, random_state=0).prepare().system_
    system = System(
        elements=gas.elements,
        interactions=[
            *gas.interactions,
            LangevinThermostatInteraction('bath', target='gas', temperature=2.0, gamma=1.0),
        ],
        space=gas.space,
    )
    trajectory = simulate(system, until=20, solver=solver, dt=dt, random_state=0)
    temperature = (trajectory.values('gas.x_d1')[-5:] ** 2).sum(axis=2).mean() / 2
    assert temperature == pytest.approx(2.0, rel=0.05)
