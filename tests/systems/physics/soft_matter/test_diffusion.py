"""Test the `_diffusion` module of `skcomplex.systems.physics.soft_matter`."""

import numpy as np
import pytest

from skcomplex.core.engine.runtime import Simulation, simulate
from skcomplex.core.engine.solvers import EulerMaruyamaSolver
from skcomplex.systems.physics.soft_matter import (
    BrownianMotionSystem,
    DiffusionInASpeckleFieldSystem,
    KramersEscapeSystem,
    MultiplicativeNoiseSystem,
    OpticalTrapSystem,
    ScaledBrownianMotionSystem,
)


def test_brownian_motion_is_ballistic_then_diffusive():
    """Test that inertial Brownian motion is ballistic below `m / gamma` and diffusive above, and overdamped is not."""
    system = BrownianMotionSystem(4000)
    trajectory = simulate(system, until=1, dt=0.05, solver=EulerMaruyamaSolver(dt=0.001), random_state=0)
    positions = trajectory.values('beads.x')[..., 0]
    assert (positions[2] ** 2).mean() == pytest.approx(2 * (0.1 - 1 + np.exp(-0.1)), rel=0.1)
    overdamped = BrownianMotionSystem(4000, inertial=False)
    trajectory = simulate(overdamped, dt=0.5, solver=EulerMaruyamaSolver(dt=0.01), random_state=0)
    assert (trajectory.values('beads.x')[-1] ** 2).mean() == pytest.approx(2 * 10, rel=0.1)


def test_optical_trap_holds_particles_with_the_thermal_variance():
    """Test that trapped particles have the variance `kT / k` along each axis, and that moving traps run."""
    trajectory = (
        Simulation(OpticalTrapSystem(1000).prepare().system_, dt=1, solver=EulerMaruyamaSolver(dt=0.01), random_state=0)
        .prepare()
        .run(until=30)
    )
    positions = trajectory.trajectory_.values('beads.x')[10:]
    assert positions[..., 0].var() == pytest.approx(1.0, rel=0.1)
    assert positions[..., 1].var() == pytest.approx(4.0, rel=0.1)
    blinking = OpticalTrapSystem(100, blink_period=4.0, amplitude=(1.0, 0.0), frequency=0.1)
    moving = simulate(blinking, until=10, dt=1, random_state=0)
    assert np.isfinite(moving.values('beads.x')).all()


def test_kramers_escape_rate():
    """Test that particles hop between the wells at the Kramers rate within 25%."""
    D, size, until = 0.1, 200, 400
    solver = EulerMaruyamaSolver(dt=0.01)
    trajectory = simulate(KramersEscapeSystem(size, D=D), until=until, dt=0.1, solver=solver, random_state=0)
    positions = trajectory.values('particles.x')
    wells = np.sign(positions[0])
    hops = 0
    for row in positions[1:]:
        following = np.where(row > 0.5, 1, np.where(row < -0.5, -1, wells))
        hops += (following != wells).sum()
        wells = following
    expected = np.sqrt(2) / (2 * np.pi) * np.exp(-1 / (4 * D))
    assert hops / (size * until) == pytest.approx(expected, rel=0.25)


@pytest.mark.parametrize(('noise', 'left'), [('ito', 0.75), ('stratonovich', np.log(2) / np.log(3)), ('anti-ito', 0.5)])
def test_multiplicative_noise_conventions_give_their_stationary_densities(noise, left):
    """Test that the fraction in the low-noise half follows the stationary density of each convention."""
    system = MultiplicativeNoiseSystem(500, s0=1.0, ds=1.0, L=10.0, convention=noise)
    trajectory = simulate(system, until=400, dt=0.05, random_state=0)
    positions = trajectory.values('particles.x')[2000:]
    assert (positions < 5).mean() == pytest.approx(left, abs=0.03)


def test_scaled_brownian_motion_has_the_anomalous_exponent():
    """Test that the mean squared displacement of scaled Brownian motion grows as `t^alpha`."""
    system = ScaledBrownianMotionSystem(4000, alpha=0.5)
    trajectory = simulate(system, until=10, dt=0.1, solver=EulerMaruyamaSolver(dt=0.001), random_state=0)
    squared = (trajectory.values('walkers.x') ** 2).mean(axis=1)
    exponent = np.polyfit(np.log(trajectory.times[10:]), np.log(squared[10:]), 1)[0]
    assert exponent == pytest.approx(0.5, abs=0.05)


def test_speckle_field_traps_particles():
    """Test that particles diffuse freely without the speckle field and are trapped by a strong one."""
    coefficients = {}
    for strength in (0.0, 1000.0):
        system = DiffusionInASpeckleFieldSystem(200, strength=strength)
        solver = EulerMaruyamaSolver(dt=0.01)
        trajectory = simulate(system, until=50, dt=1, solver=solver, random_state=0)
        positions = trajectory.values('beads.x')
        coefficients[strength] = ((positions[-1] - positions[20]) ** 2).sum(axis=-1).mean() / (4 * 30)
    assert coefficients[0.0] == pytest.approx(1.0, rel=0.15)
    assert coefficients[1000.0] < 0.1


def test_speckle_field_defaults_give_anomalous_diffusion():
    """Test that the default field draws the particles in over several frames and lets them escape slowly."""
    solver = EulerMaruyamaSolver(dt=0.01)
    trajectory = simulate(DiffusionInASpeckleFieldSystem(), solver=solver, random_state=0)
    positions = np.asarray(trajectory.values('beads.x'))
    times = trajectory.times[21:] - trajectory.times[20]
    squared = ((positions[21:] - positions[20]) ** 2).sum(axis=-1).mean(axis=-1)
    exponent = np.polyfit(np.log(times), np.log(squared), 1)[0]
    assert np.linalg.norm(positions[1] - positions[0], axis=-1).mean() < 5
    assert 0.0 < exponent < 0.8
