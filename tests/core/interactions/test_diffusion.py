"""Test the `_diffusion` module of `skcomplex.core.interactions`."""

from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.elements import Element, FieldElement
from skcomplex.core.interactions import DiffusionInteraction
from skcomplex.core.spaces import LatticeSpace
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.errors import ValidationError
from skcomplex.specs import RoleSpec, StateSpec

CELLS, SPACING, COEFFICIENT, WIDTH, UNTIL = 200, 0.1, 0.5, 1.0, 2.0


class CellElement(Element):
    """Cell of a field with one state."""

    def __init__(self, label: str, *, u: Annotated[object, StateSpec()] = 0.0) -> None:
        super().__init__(label)
        self.u = u


class RodDiffusionInteraction(DiffusionInteraction):
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
    heat = DiffusionInteraction('heat', target='rod', state='u', coefficient=lambda s: COEFFICIENT)
    simulation = Simulation(System(elements=[field], interactions=[heat]), dt=0.5)
    values = simulation.prepare().run(until=UNTIL).trajectory_.values('rod.u')
    np.testing.assert_allclose(values.sum(axis=1), values[0].sum(), rtol=1e-10)
    expected = _gaussian(positions, WIDTH**2 + 2 * COEFFICIENT * UNTIL)
    np.testing.assert_allclose(values[-1], expected, atol=1e-4)


def test_check_rejects_a_coefficient_that_is_not_a_function():
    """Test that `check` rejects a coefficient that is neither callable nor a number."""
    with pytest.raises(ValidationError, match='coefficient'):
        DiffusionInteraction('heat', target='rod', coefficient='fast').prepare()


def test_get_laws_reads_a_renamed_target():
    """Test that a diffusion that binds its field as `rod` gives the trajectory of the plain diffusion."""
    positions = np.arange(CELLS) * SPACING
    space = LatticeSpace((CELLS,), spacing=SPACING, boundary='periodic')
    field = FieldElement('bar', element=CellElement, space=space, values={'u': _gaussian(positions, WIDTH**2)})
    values = []
    for heat in (
        DiffusionInteraction('heat', target='bar', state='u', coefficient=lambda s: COEFFICIENT),
        RodDiffusionInteraction('heat', rod='bar', state='u', coefficient=lambda s: COEFFICIENT),
    ):
        simulation = Simulation(System(elements=[field], interactions=[heat]), dt=0.5)
        values.append(simulation.prepare().run(until=1.0).trajectory_.values('bar.u'))
    np.testing.assert_allclose(values[1], values[0])
