"""Test the `_criminality` module of `skcomplex.systems.social.crowds`."""

from skcomplex.core.engine.compile import PassCompiler
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import KineticSolver
from skcomplex.systems import create_system
from skcomplex.systems.social.crowds import CriminalitySystem


def test_security_forces_respond_to_criminality():
    """Test that recruitment reads the mean criminality, so security grows while crime is high and crime recedes."""
    trajectory = simulate(CriminalitySystem(), until=60, dt=2.0, random_state=0)
    security = trajectory.values('security_forces')
    assert security.max() > 5 * security[0]
    assert trajectory.values('mean_criminality')[-1] < trajectory.values('mean_criminality')[0]
    weak = simulate(CriminalitySystem(deterrence=0.0, recruitment=0.0), until=60, dt=60.0, random_state=0)
    assert weak.values('mean_criminality')[-1] > 0.5


def test_criminality_needs_an_ode_solver():
    """Test that the macro-scale recruitment, a rate law, rules out the particle solver."""
    issues = PassCompiler(solver=KineticSolver()).prepare(create_system('criminality')).report_.errors
    assert [issue.code for issue in issues] == ['solver-unsupported']
