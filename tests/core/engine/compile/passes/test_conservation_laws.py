"""Test the `_conservation_laws` module of `skcomplex.core.engine.compile.passes`."""

import numpy as np
import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import divergence
from skcomplex.core.components.spaces import FiniteVolumeSpec, LatticeSpace
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System

CELLS = 40


def _burgers(s):
    """Return the divergence of the Burgers flux plus a source in one rate."""
    return [RateLaw(s.target.u, -divergence([s.target.u**2 / 2]) + 0.2)]


def _squared(s):
    """Return the square of the divergence of the Burgers flux, which is no conservation law."""
    return [RateLaw(s.target.u, divergence([s.target.u**2 / 2]) ** 2)]


def _field(laws, scheme=None):
    """Return a periodic finite-volume field that starts from a sine wave."""
    centers = (np.arange(CELLS) + 0.5) * 2 * np.pi / CELLS
    space = LatticeSpace(
        (CELLS,),
        spacing=2 * np.pi / CELLS,
        boundary='periodic',
        discretization='finite-volume',
        scheme=scheme,
    )
    line = Element('line', values={'u': 1.0 + 0.5 * np.sin(centers)}, space=space)
    return System(elements=[line], interactions=[Interaction('transport', roles={'target': 'line'}, laws=laws)])


def test_rate_with_a_divergence_compiles_to_a_finite_volume_kernel_and_a_source():
    """Test that the divergence becomes the flux of the field and the other term a source that adds to the total."""
    system = _field(_burgers).prepare()
    assert len(system._problem.volumes) == 1
    values = system.simulate(until=1.0, dt=0.1).get_values('line.u')
    assert values[-1].sum() - values[0].sum() == pytest.approx(0.2 * CELLS, rel=1e-9)


def test_scheme_of_the_field_chooses_the_numerical_flux():
    """Test that the Godunov flux of the scheme gives another run than the Rusanov flux, with the same total."""
    godunov = _field(_burgers, FiniteVolumeSpec(riemann='godunov'))
    first = _field(_burgers).prepare().simulate(until=1.0, dt=0.1).get_values('line.u')
    second = godunov.prepare().simulate(until=1.0, dt=0.1).get_values('line.u')
    assert not np.allclose(first[-1], second[-1])
    assert second[-1].sum() == pytest.approx(first[-1].sum(), rel=1e-9)


def test_rate_that_is_no_conservation_law_is_rejected():
    """Test that a divergence inside a product says how to write the transport."""
    with pytest.raises(ValidationError, match='-divergence'):
        _field(_squared).prepare()
