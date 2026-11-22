"""Test the `_metropolis` module of `skcomplex.core.engine.solvers`."""

import numpy as np
import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import ObjectiveLaw
from skcomplex.core.components.laws.utils import neighbors, where
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.engine.solvers import MetropolisSolver
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System

from .test_objective import Well


def test_metropolis_samples_the_boltzmann_weight_of_a_well():
    """Test that the states of a well `(x - 2)^2` at temperature 0.5 have mean 2 and variance 0.25."""
    solver = MetropolisSolver(temperature=0.5, step=0.5)
    system = System(elements=[Element('ball', values={'x': 2.0})], interactions=[Well('well')], solver=solver)
    positions = system.prepare().simulate(until=20000, dt=1.0, random_state=0).get_values('ball.x')
    assert np.mean(positions) == pytest.approx(2.0, abs=0.05)
    assert np.var(positions) == pytest.approx(0.25, abs=0.03)


@pytest.mark.parametrize('setting', ['temperature', 'step'])
def test_metropolis_rejects_a_setting_that_is_not_positive(setting):
    """Test that a temperature or a step of zero raises an error naming it."""
    solver = MetropolisSolver(**{setting: 0.0})
    system = System(elements=[Element('ball', values={'x': 0.0})], interactions=[Well('well')], solver=solver)
    with pytest.raises(ValidationError, match=f'{setting} of MetropolisSolver must be positive'):
        system.prepare().simulate(until=1.0)


def _potts(s):
    """Return the objective of a Potts color, minus its like neighbours over the temperature."""
    color = s.target.color
    return [ObjectiveLaw(-neighbors(where(s.target.other.color == color, 1, 0)) / s.target.T, color)]


def test_metropolis_moves_the_members_of_a_group_by_their_moves():
    """Test that random values drawn from three colors by the heat bath order a cold lattice and keep the colors."""
    colors = np.random.default_rng(1).integers(0, 3, 144).astype(float)
    sites = Element('sites', values={'color': colors, 'T': 0.1}, space=LatticeSpace((12, 12)))
    solver = MetropolisSolver(
        moves={'sites.color': 'random-value'},
        rule='heat-bath',
        values={'sites.color': (0, 1, 2)},
    )
    potts = Interaction('potts', roles={'target': 'sites'}, laws=_potts)
    system = System(elements=[sites], interactions=[potts], solver=solver).prepare()
    values = system.simulate(until=50, random_state=0).get_values('sites.color')
    assert type(system.solver_).__name__ == '_MemberMetropolisSolver'
    assert set(np.unique(values)) <= {0.0, 1.0, 2.0}
    assert np.bincount(values[-1].astype(int)).max() > 0.6 * values[-1].size
