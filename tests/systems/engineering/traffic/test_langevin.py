"""Test the `_langevin` module of `skcomplex.systems.engineering.traffic`."""

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import EulerMaruyamaSolver, RK4Solver
from skcomplex.systems.engineering.traffic import LangevinTrafficSystem, TrafficDensityEquationSystem


def test_langevin_traffic_follows_desired_speed_with_density_dependent_noise() -> None:
    """Mean speed equals the desired speed of the density and the speed variance equals `D1 rho / gamma`."""
    gamma, D1, density = 3.0, 0.05, 0.5
    system = LangevinTrafficSystem(50, 100.0, gamma=gamma, D1=D1).prepare().system_
    simulation = (
        Simulation(system, dt=1.0, solver=EulerMaruyamaSolver(dt=0.01), random_state=0).prepare().run(until=200)
    )
    speeds = simulation.trajectory_.values('cars.v')[50:]
    assert abs(speeds.mean() - (1 - density)) < 0.02
    assert abs(speeds.var() / (D1 * density / gamma) - 1) < 0.2


def test_traffic_density_equation_jams_above_threshold_density() -> None:
    """Density waves decay below the density `D1 gamma` and grow into jams above it."""
    spreads = []
    for density in (0.1, 0.5):
        system = TrafficDensityEquationSystem(50, length=25.0, density=density, D1=0.2).prepare().system_
        simulation = Simulation(system, dt=10.0, solver=RK4Solver(dt=0.1)).prepare().run(until=300)
        rho = simulation.trajectory_.values('road.rho')
        spreads.append(rho[-1].std() / rho[0].std())
        assert abs(rho[-1].mean() - rho[0].mean()) < 1e-9
    stable, unstable = spreads
    assert stable < 0.5
    assert unstable > 10
