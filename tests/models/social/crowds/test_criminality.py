"""Test the `_criminality` module of `skcomplex.models.social.crowds`."""

from skcomplex.engine.compile import validate
from skcomplex.engine.solvers import KineticSolver
from skcomplex.models import create

from ...conftest import _trajectory


def test_security_forces_respond_to_criminality():
    """Test that recruitment reads the mean criminality, so security grows while crime is high and crime recedes."""
    trajectory = _trajectory('criminality', 60, 2.0)
    security = trajectory.values('security_forces')
    assert security.max() > 5 * security[0]
    assert trajectory.values('mean_criminality')[-1] < trajectory.values('mean_criminality')[0]
    weak = _trajectory('criminality', 60, 60.0, deterrence=0.0, recruitment=0.0)
    assert weak.values('mean_criminality')[-1] > 0.5


def test_criminality_needs_an_ode_solver():
    """Test that the macro-scale recruitment, a rate law, rules out the particle solver."""
    issues = validate(create('criminality'), solver=KineticSolver()).errors
    assert [issue.code for issue in issues] == ['solver-unsupported']
