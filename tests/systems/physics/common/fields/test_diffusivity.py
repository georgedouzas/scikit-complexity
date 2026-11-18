"""Test the `_diffusivity` module of `skcomplex.systems.physics.common.fields`."""

import numpy as np
import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System
from skcomplex.systems.physics.common.fields import DiffusivityInteraction

CELLS, SPACING, COEFFICIENT, WIDTH, UNTIL = 200, 0.1, 0.5, 1.0, 2.0


def _gaussian(positions, variance):
    """Return the normal density of a variance around the middle of the rod."""
    return np.exp(-((positions - CELLS * SPACING / 2) ** 2) / (2 * variance)) / np.sqrt(2 * np.pi * variance)


def test_run_conserves_the_total_and_spreads_as_the_heat_kernel():
    """Test that diffusion on a periodic field keeps the total and widens a Gaussian to the variance `s**2 + 2 D t`."""
    positions = np.arange(CELLS) * SPACING
    space = LatticeSpace((CELLS,), spacing=SPACING, boundary='periodic')
    field = Element('rod', values={'u': _gaussian(positions, WIDTH**2)}, space=space)
    heat = DiffusivityInteraction('heat', target='rod', state='u', coefficient=lambda s: COEFFICIENT)
    system = System(elements=[field], interactions=[heat]).prepare()
    values = system.simulate(until=UNTIL, dt=0.5).get_values('rod.u')
    np.testing.assert_allclose(values.sum(axis=1), values[0].sum(), rtol=1e-10)
    expected = _gaussian(positions, WIDTH**2 + 2 * COEFFICIENT * UNTIL)
    np.testing.assert_allclose(values[-1], expected, atol=1e-4)


def test_check_rejects_a_coefficient_that_is_not_a_function():
    """Test that `check` rejects a coefficient that is neither callable nor a number."""
    with pytest.raises(ValidationError, match='coefficient'):
        DiffusivityInteraction('heat', target='rod', coefficient='fast').prepare()


def test_run_reads_the_field_its_target_binds():
    """Test that a diffusion of a field with another label gives the trajectory of the plain diffusion."""
    positions = np.arange(CELLS) * SPACING
    space = LatticeSpace((CELLS,), spacing=SPACING, boundary='periodic')
    values = []
    for label in ('rod', 'bar'):
        field = Element(label, values={'u': _gaussian(positions, WIDTH**2)}, space=space)
        heat = DiffusivityInteraction('heat', target=label, state='u', coefficient=lambda s: COEFFICIENT)
        system = System(elements=[field], interactions=[heat]).prepare()
        values.append(system.simulate(until=1.0, dt=0.5).get_values(f'{label}.u'))
    np.testing.assert_allclose(values[1], values[0])
