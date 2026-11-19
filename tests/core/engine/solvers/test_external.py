"""Test the `_external` module of `skcomplex.core.engine.solvers`."""

import pytest

from skcomplex.core.engine.solvers import LammpsSolver
from skcomplex.core.errors import BackendError

from ..compile.conftest import phase_system


def test_lammps_backend_rejects_complex_state():
    """Test that the LAMMPS backend names the complex state it cannot run."""
    with pytest.raises(BackendError, match=r'LAMMPS backend.*complex state osc\.z'):
        phase_system(solver=LammpsSolver()).prepare().simulate(until=0.1, dt=0.1)
