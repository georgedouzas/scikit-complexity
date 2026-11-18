"""Test the `_advection` module of `skcomplex.systems.physics.common.fields`."""

import numpy as np
import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import divergence
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System
from skcomplex.systems.physics.common.fields import AdvectionInteraction

CELLS = 64


def _run(*interactions, discretization='finite-volume', label='tube'):
    """Return the tracer of a periodic field that starts from a bump."""
    spacing = 1 / CELLS
    centers = (np.arange(CELLS) + 0.5) * spacing
    values = {'u': np.exp(-(((centers - 0.5) / 0.1) ** 2)), 'v': 1.0 + 0.5 * np.sin(2 * np.pi * centers)}
    space = LatticeSpace((CELLS,), spacing=spacing, boundary='periodic', discretization=discretization)
    field = Element(label, values=values, space=space)
    system = System(elements=[field], interactions=list(interactions)).prepare()
    return system.simulate(until=0.2, dt=0.05).get_values(f'{label}.u')


def test_run_matches_the_conservation_law_of_the_flux_v_u():
    """Test that advection gives the trajectory of the conservation law of the flux `a u` and keeps the total."""
    advection = AdvectionInteraction('drift', target='tube', state='u', drift=lambda s: s.target.v)
    flux = Interaction(
        'drift',
        roles={'target': 'tube'},
        laws=lambda s: [RateLaw(s.target.u, -divergence([s.target.v * s.target.u]))],
    )
    values = _run(advection)
    np.testing.assert_allclose(values, _run(flux), atol=1e-12)
    np.testing.assert_allclose(values.sum(axis=1), values[0].sum(), rtol=1e-12)


def test_run_keeps_the_total_of_a_spectral_field():
    """Test that advection keeps the total of a spectral field and moves the tracer."""
    advection = AdvectionInteraction('drift', target='tube', state='u', drift=lambda s: [1.0])
    values = _run(advection, discretization='spectral')
    assert values[-1].sum() == pytest.approx(values[0].sum(), rel=1e-9)
    assert not np.allclose(values[-1], values[0])


def test_check_rejects_a_drift_that_is_not_callable():
    """Test that `check` rejects a drift that is not a function."""
    with pytest.raises(ValidationError, match='drift'):
        AdvectionInteraction('drift', target='tube', drift=1.0).prepare()


def test_run_reads_the_field_its_target_binds():
    """Test that an advection of a field with another label gives the trajectory of the plain advection."""
    plain = AdvectionInteraction('drift', target='tube', state='u', drift=lambda s: s.target.v)
    renamed = AdvectionInteraction('drift', target='pipe', state='u', drift=lambda s: s.target.v)
    np.testing.assert_allclose(_run(renamed, label='pipe'), _run(plain))
