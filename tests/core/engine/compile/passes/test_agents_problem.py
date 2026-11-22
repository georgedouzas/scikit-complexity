"""Test the `_agents_problem` module of `skcomplex.core.engine.compile.passes`."""

import numpy as np
import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.errors import BackendError
from skcomplex.core.systems import System


def _turn(agent, system):
    """Turn the complex value of an agent by a quarter."""
    agent.z = agent.z * 1j


def test_mesa_backend_rejects_complex_agent_state():
    """Test that an agent type with a complex state names the state and the Mesa backend."""
    walkers = Element('walkers', values={'z': np.zeros(3, dtype=complex)}, size=3)
    rule = Interaction('turn', roles={'target': 'walkers'}, laws=lambda s: [UpdateLaw(s.target, _turn)])
    with pytest.raises(BackendError, match=r'walkers.*complex state z.*Mesa'):
        System(elements=[walkers], interactions=[rule]).prepare()
