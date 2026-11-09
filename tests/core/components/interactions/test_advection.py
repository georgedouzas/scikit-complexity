"""Test the `_advection` module of `skcomplex.core.components.interactions`."""

from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.components.elements import Element, FieldElement
from skcomplex.core.components.interactions import AdvectionInteraction, ConservationInteraction
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.systems import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import RoleSpec, StateSpec

CELLS = 64


class TracerElement(Element):
    """Cell of a field with a tracer and a drift."""

    def __init__(
        self,
        label: str,
        *,
        u: Annotated[object, StateSpec()] = 0.0,
        v: Annotated[object, StateSpec()] = 1.0,
    ) -> None:
        super().__init__(label)
        self.u = u
        self.v = v


class PipeAdvectionInteraction(AdvectionInteraction):
    """Advection that binds its field as `pipe`."""

    def __init__(self, label: str, *, pipe: Annotated[str, RoleSpec()], **arguments: object) -> None:
        super().__init__(label, target=pipe, **arguments)
        self.pipe = pipe

    def _get_role(self, role: str) -> str:
        """Return `pipe` for the target."""
        return 'pipe' if role == 'target' else role


def _run(*interactions, discretization='finite-volume'):
    """Return the tracer of a periodic field that starts from a bump."""
    spacing = 1 / CELLS
    centers = (np.arange(CELLS) + 0.5) * spacing
    values = {'u': np.exp(-(((centers - 0.5) / 0.1) ** 2)), 'v': 1.0 + 0.5 * np.sin(2 * np.pi * centers)}
    space = LatticeSpace((CELLS,), spacing=spacing, boundary='periodic')
    field = FieldElement('tube', element=TracerElement, space=space, values=values, discretization=discretization)
    return simulate(System(elements=[field], interactions=list(interactions)), until=0.2, dt=0.05).values('tube.u')


def test_run_matches_the_conservation_law_of_the_flux_v_u():
    """Test that advection gives the trajectory of the conservation law of the flux `a u` and keeps the total."""
    advection = AdvectionInteraction('drift', target='tube', state='u', drift=lambda s: s.target.v)
    flux = ConservationInteraction('drift', target='tube', state='u', flux=lambda s: s.target.v * s.target.u)
    values = _run(advection)
    np.testing.assert_allclose(values, _run(flux), atol=1e-12)
    np.testing.assert_allclose(values.sum(axis=1), values[0].sum(), rtol=1e-12)


def test_run_without_numerical_flux_keeps_the_total_of_a_spectral_field():
    """Test that advection without a numerical flux keeps the total of a spectral field and moves the tracer."""
    advection = AdvectionInteraction('drift', target='tube', state='u', drift=lambda s: [1.0], riemann=None)
    values = _run(advection, discretization='spectral')
    assert values[-1].sum() == pytest.approx(values[0].sum(), rel=1e-9)
    assert not np.allclose(values[-1], values[0])


def test_check_rejects_a_drift_that_is_not_callable():
    """Test that `check` rejects a drift that is not a function."""
    with pytest.raises(ValidationError, match='drift'):
        AdvectionInteraction('drift', target='tube', drift=1.0).prepare()


def test_get_laws_reads_a_renamed_target():
    """Test that an advection that binds its field as `pipe` gives the trajectory of the plain advection."""
    plain = AdvectionInteraction('drift', target='tube', state='u', drift=lambda s: s.target.v)
    renamed = PipeAdvectionInteraction('drift', pipe='tube', state='u', drift=lambda s: s.pipe.v)
    np.testing.assert_allclose(_run(renamed), _run(plain))
