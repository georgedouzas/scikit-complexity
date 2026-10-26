"""Test the `_wiener` module of `skcomplex.core.components`."""

import numpy as np
import pytest

from skcomplex.core.base import BaseElement, BaseInteraction, LabelError, RoleError, role
from skcomplex.core.components import WienerProcess
from skcomplex.core.laws import NoiseLaw
from skcomplex.core.system import System
from skcomplex.engine.compile import compile_system
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import EulerMaruyamaSolver

from ...conftest import HoldingElement, _correlated


class ShockInteraction(BaseInteraction):
    """Unit noise on one element from the first process of a set of Wiener processes."""

    target: str = role()
    noise: WienerProcess = role(noise=True)

    def laws(self, s):
        """Return the noise of the target."""
        return [NoiseLaw(s.target.n, 1.0, process=s.noise.dW)]


def _shocked(first, second):
    """Return two holdings, each shocked by an interaction that binds its own Wiener processes."""
    return System(
        components=[HoldingElement('a'), HoldingElement('b')],
        interactions=[
            ShockInteraction('first', target='a', noise=first),
            ShockInteraction('second', target='b', noise=second),
        ],
    )


def test_correlated_noise():
    """Test that two noise laws drawing the processes of a set bound to a role have the increments `h C`."""
    covariance = [[1.0, 0.6], [0.6, 2.0]]
    simulation = Simulation(_correlated(covariance), solver=EulerMaruyamaSolver(dt=0.5), random_state=0, dt=0.5)
    trajectory = simulation.run(until=5_000).trajectory_
    increments = np.diff(np.stack([trajectory.values('a.n'), trajectory.values('b.n')]), axis=1)
    np.testing.assert_allclose(np.cov(increments), 0.5 * np.asarray(covariance), atol=0.05)
    assert not issubclass(WienerProcess, BaseElement)


def test_interactions_share_processes_of_one_label():
    """Test that two interactions that bind processes of one label draw the same increments."""
    market = WienerProcess('market')
    trajectory = Simulation(_shocked(market, WienerProcess('market')), random_state=0).run(until=1).trajectory_
    np.testing.assert_array_equal(trajectory.values('a.n'), trajectory.values('b.n'))
    assert compile_system(_shocked(market, market)).n_processes == 1


def test_processes_of_one_label_must_agree():
    """Test that two different sets of processes with one label, and a path bound to a noise role, are rejected."""
    with pytest.raises(LabelError, match='two different Wiener processes'):
        compile_system(_shocked(WienerProcess('market'), WienerProcess('market', dimension=2)))
    with pytest.raises(LabelError, match='label of another child'):
        compile_system(_shocked(WienerProcess('a'), WienerProcess('a')))
    with pytest.raises(RoleError, match='takes a WienerProcess'):
        compile_system(_shocked('market', 'market'))
