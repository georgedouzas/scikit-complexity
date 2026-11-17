"""Test the `_external` module of `skcomplex.core.engine.solvers`."""

import pytest

from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import LammpsSolver
from skcomplex.core.errors import BackendError

from ..compile.conftest import PhaseElement


def test_lammps_backend_rejects_complex_state():
    """Test that the LAMMPS backend names the complex state it cannot run."""
    with pytest.raises(BackendError, match=r'LAMMPS backend.*complex state osc\.z'):
        simulate(CompositeSystem(elements=[PhaseElement('osc')]), until=0.1, solver=LammpsSolver(), dt=0.1)
