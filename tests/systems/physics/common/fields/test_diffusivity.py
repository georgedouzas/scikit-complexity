"""Test the `_diffusivity` module of `skcomplex.systems.physics.common.fields`."""

from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.components.elements import Element, FieldElement
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import RoleSpec, StateSpec
from skcomplex.systems.physics.common.fields import DiffusivityInteraction

CELLS, SPACING, COEFFICIENT, WIDTH, UNTIL = 200, 0.1, 0.5, 1.0, 2.0


class CellElement(Element):
    """Cell of a field with one state."""

    def __init__(self, label: str, *, u: Annotated[object, StateSpec()] = 0.0) -> None:
        super().__init__(label)
        self.u = u


class RodDiffusivityInteraction(DiffusivityInteraction):
    """Diffusion that binds its field as `rod`."""

    def __init__(self, label: str, *, rod: Annotated[str, RoleSpec()], **arguments: object) -> None:
        super().__init__(label, target=rod, **arguments)
        self.rod = rod

    def _get_role(self, role: str) -> str:
        """Return `rod` for the target."""
        return 'rod' if role == 'target' else role


def _gaussian(positions, variance):
    """Return the normal density of a variance around the middle of the rod."""
    return np.exp(-((positions - CELLS * SPACING / 2) ** 2) / (2 * variance)) / np.sqrt(2 * np.pi * variance)


def test_run_conserves_the_total_and_spreads_as_the_heat_kernel():
    """Test that diffusion on a periodic field keeps the total and widens a Gaussian to the variance `s**2 + 2 D t`."""
    positions = np.arange(CELLS) * SPACING
    space = LatticeSpace((CELLS,), spacing=SPACING, boundary='periodic')
    field = FieldElement('rod', element=CellElement, space=space, values={'u': _gaussian(positions, WIDTH**2)})
    heat = DiffusivityInteraction('heat', target='rod', state='u', coefficient=lambda s: COEFFICIENT)
    values = simulate(CompositeSystem(elements=[field], interactions=[heat]), until=UNTIL, dt=0.5).values('rod.u')
    np.testing.assert_allclose(values.sum(axis=1), values[0].sum(), rtol=1e-10)
    expected = _gaussian(positions, WIDTH**2 + 2 * COEFFICIENT * UNTIL)
    np.testing.assert_allclose(values[-1], expected, atol=1e-4)


def test_check_rejects_a_coefficient_that_is_not_a_function():
    """Test that `check` rejects a coefficient that is neither callable nor a number."""
    with pytest.raises(ValidationError, match='coefficient'):
        DiffusivityInteraction('heat', target='rod', coefficient='fast').prepare()


def test_get_laws_reads_a_renamed_target():
    """Test that a diffusion that binds its field as `rod` gives the trajectory of the plain diffusion."""
    positions = np.arange(CELLS) * SPACING
    space = LatticeSpace((CELLS,), spacing=SPACING, boundary='periodic')
    field = FieldElement('bar', element=CellElement, space=space, values={'u': _gaussian(positions, WIDTH**2)})
    values = []
    for heat in (
        DiffusivityInteraction('heat', target='bar', state='u', coefficient=lambda s: COEFFICIENT),
        RodDiffusivityInteraction('heat', rod='bar', state='u', coefficient=lambda s: COEFFICIENT),
    ):
        values.append(
            simulate(CompositeSystem(elements=[field], interactions=[heat]), until=1.0, dt=0.5).values('bar.u'),
        )
    np.testing.assert_allclose(values[1], values[0])
