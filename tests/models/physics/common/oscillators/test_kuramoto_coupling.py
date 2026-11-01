"""Test the `_kuramoto_coupling` module of `skcomplex.models.physics.common.oscillators`."""

import numpy as np
import pytest

from skcomplex.core.elements import GroupElement
from skcomplex.core.neighborhoods import RadiusNeighborhood
from skcomplex.core.spaces import ContinuousSpace
from skcomplex.core.system import System
from skcomplex.engine.compile import compile_system
from skcomplex.models.physics.common.oscillators import (
    KuramotoCouplingInteraction,
    PhaseOscillatorElement,
    SitedOscillatorElement,
)

THETA = np.array([0.1, 1.3, 2.9, 4.2, 5.5])


def _drift(system):
    """Return the drift of a system at its initial state."""
    problem = compile_system(system)
    return problem.drift(0.0, problem.initial_state, problem.initial_parameters)


@pytest.mark.parametrize('include_self', [False, True])
@pytest.mark.parametrize('reduce', ['sum', 'mean'])
def test_group_coupling_is_the_lagged_sine_sum(include_self, reduce):
    """Test that the group coupling, read from the group sums, is the sum of `K sin(theta_j - theta_i - alpha)`."""
    phases = GroupElement('phases', element=PhaseOscillatorElement, size=5, values={'theta': THETA, 'omega': 0.0})
    coupling = KuramotoCouplingInteraction(
        'coupling',
        target='phases',
        K=0.7,
        alpha=0.4,
        mode='group',
        reduce=reduce,
        include_self=include_self,
    )
    terms = 0.7 * np.sin(THETA[np.newaxis, :] - THETA[:, np.newaxis] - 0.4)
    if not include_self:
        np.fill_diagonal(terms, 0.0)
    expected = terms.sum(axis=1) / (1 if reduce == 'sum' else 5 - (0 if include_self else 1))
    np.testing.assert_allclose(_drift(System(elements=[phases], interactions=[coupling])), expected, atol=1e-12)


def test_coupling_within_a_radius_on_a_periodic_ring():
    """Test that oscillators on a periodic ring couple to their neighbours within the radius and to themselves."""
    ring = GroupElement(
        'ring',
        element=SitedOscillatorElement,
        size=5,
        values={'theta': THETA, 'omega': 0.0, 'x': np.arange(5, dtype=float).reshape(-1, 1)},
    )
    coupling = KuramotoCouplingInteraction(
        'coupling',
        target='ring',
        K=0.7,
        alpha=0.4,
        include_self=True,
        neighborhood=RadiusNeighborhood(1.5),
    )
    system = System(elements=[ring], interactions=[coupling], space=ContinuousSpace((5.0,), boundary='periodic'))
    expected = 0.7 * sum(np.sin(np.roll(THETA, shift) - THETA - 0.4) for shift in (-1, 0, 1))
    np.testing.assert_allclose(_drift(system)[:5], expected, atol=1e-12)
