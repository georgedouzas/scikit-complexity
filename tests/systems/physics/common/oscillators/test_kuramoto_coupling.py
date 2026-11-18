"""Test the `_kuramoto_coupling` module of `skcomplex.systems.physics.common.oscillators`."""

import numpy as np
import pytest

from skcomplex.core.components.spaces import ContinuousSpace, RadiusNeighborhood
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System
from skcomplex.systems.physics.common.oscillators import (
    KuramotoCouplingInteraction,
    PhaseOscillatorElement,
    SitedOscillatorElement,
)

THETA = np.array([0.1, 1.3, 2.9, -2.1, -0.8])


def _rate(element, coupling):
    """Return the rate of the phases of an element at its initial state."""
    problem = System(elements=[element], interactions=[coupling]).prepare()._problem
    drift = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    return problem.states.read(drift, f'{element.label}.theta')


@pytest.mark.parametrize('include_self', [False, True])
@pytest.mark.parametrize('reduce', ['sum', 'mean'])
def test_group_coupling_is_the_lagged_sine_sum(include_self, reduce):
    """Test that the group coupling, read from the group sums, is the sum of `K sin(theta_j - theta_i - alpha)`."""
    phases = PhaseOscillatorElement('phases', theta=THETA, omega=0.0, size=5)
    coupling = KuramotoCouplingInteraction(
        'coupling',
        target='phases',
        K=0.7,
        alpha=0.4,
        reduce=reduce,
        include_self=include_self,
    )
    terms = 0.7 * np.sin(THETA[np.newaxis, :] - THETA[:, np.newaxis] - 0.4)
    if not include_self:
        np.fill_diagonal(terms, 0.0)
    expected = terms.sum(axis=1) / (1 if reduce == 'sum' else 5 - (0 if include_self else 1))
    np.testing.assert_allclose(_rate(phases, coupling), expected, atol=1e-12)


def test_coupling_within_a_radius_on_a_periodic_ring():
    """Test that oscillators on a periodic ring couple to their neighbours within the radius and to themselves."""
    ring = SitedOscillatorElement(
        'ring',
        theta=THETA,
        omega=0.0,
        x=np.arange(5, dtype=float).reshape(-1, 1),
        size=5,
        space=ContinuousSpace((5.0,), boundary='periodic'),
    )
    coupling = KuramotoCouplingInteraction(
        'coupling',
        target='ring',
        K=0.7,
        alpha=0.4,
        include_self=True,
        neighborhood=RadiusNeighborhood(1.5),
    )
    expected = 0.7 * sum(np.sin(np.roll(THETA, shift) - THETA - 0.4) for shift in (-1, 0, 1))
    np.testing.assert_allclose(_rate(ring, coupling), expected, atol=1e-12)


def test_coupling_rejects_an_unknown_reduction():
    """Test that a reduction other than a sum or a mean is named in the error."""
    phases = PhaseOscillatorElement('phases', theta=THETA, size=5)
    coupling = KuramotoCouplingInteraction('coupling', target='phases', reduce='max')
    with pytest.raises(ValidationError, match="'max'"):
        System(elements=[phases], interactions=[coupling]).prepare()
