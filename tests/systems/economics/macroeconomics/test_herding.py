"""Test the `_herding` module of `skcomplex.systems.economics.macroeconomics`."""

import numpy as np
import pytest

from skcomplex.systems.economics.macroeconomics import NetworkHerdingSystem

from ..conftest import _run


def test_network_herding_aggregates_through_the_out_degrees():
    """Test Eq.

    1.12, the mean choice from the out-degrees, and the fixed point `(I - Theta)^-1 gamma x` without shocks.
    """
    system = NetworkHerdingSystem(sigma=0.0).prepare().system_
    theta = np.asarray(system.interactions[0].weights)
    x = system.elements[0].values['x']
    trajectory = _run(system, 40, observables=NetworkHerdingSystem().get_run_settings()['observables'])
    choices = trajectory.get_values('agents.c')
    degrees = theta.sum(axis=0)
    assert trajectory.get_values('mean_choice')[1:] == pytest.approx(choices[:-1] @ degrees / len(x) + np.mean(x))
    assert choices[-1] == pytest.approx(np.linalg.solve(np.eye(len(x)) - theta, x), rel=1e-6)


def test_network_herding_persists_more_with_stronger_links():
    """Test that the mean choice of a shocked economy is more autocorrelated with stronger herding."""

    def persistence(herding):
        system = NetworkHerdingSystem(sigma=1.0, herding=herding, density=0.3).prepare().system_
        mean = _run(system, 2000, observables=NetworkHerdingSystem().get_run_settings()['observables']).get_values(
            'mean_choice',
        )
        return np.corrcoef(mean[100:-1], mean[101:])[0, 1]

    assert persistence(0.8) > persistence(0.0) + 0.5
