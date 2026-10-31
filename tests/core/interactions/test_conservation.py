"""Test the `_conservation` module of `skcomplex.core.interactions`."""

from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.elements import Element, FieldElement
from skcomplex.core.interactions import ConservationInteraction
from skcomplex.core.laws import FluxLaw
from skcomplex.core.spaces import LatticeSpace
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.errors import ValidationError
from skcomplex.specs import StateSpec

CELLS = 64


class CellElement(Element):
    """Cell of a field with one state."""

    def __init__(self, label: str, *, u: Annotated[object, StateSpec()] = 0.0) -> None:
        super().__init__(label)
        self.u = u


class BurgersElement(CellElement):
    """Cell of a field that follows the Burgers equation written by hand."""

    def _get_laws(self, s):
        """Return the Burgers flux."""
        return [FluxLaw(s.u, s.u**2 / 2)]


def _run(element, *interactions):
    """Return the state of a periodic finite-volume field that starts from a sine wave."""
    spacing = 2 * np.pi / CELLS
    values = {'u': 1.0 + 0.5 * np.sin((np.arange(CELLS) + 0.5) * spacing)}
    space = LatticeSpace((CELLS,), spacing=spacing, boundary='periodic')
    field = FieldElement('line', element=element, space=space, values=values, discretization='finite-volume')
    simulation = Simulation(System(elements=[field], interactions=list(interactions)), dt=0.1)
    return simulation.check().prepare().run(until=1.0).trajectory_.values('line.u')


def test_run_matches_the_flux_law_and_conserves_the_total():
    """Test that the Burgers flux gives the trajectory of its flux law and keeps the total of the field."""
    burgers = ConservationInteraction('burgers', target='line', state='u', flux=lambda s: s.target.u**2 / 2)
    values = _run(CellElement, burgers)
    np.testing.assert_allclose(values, _run(BurgersElement), atol=1e-12)
    np.testing.assert_allclose(values.sum(axis=1), values[0].sum(), rtol=1e-12)


def test_run_adds_the_source():
    """Test that a constant source raises the total of the field at the rate of the source times the cells."""
    burgers = ConservationInteraction(
        'burgers',
        target='line',
        state='u',
        flux=lambda s: s.target.u**2 / 2,
        source=lambda s: 0.2,
    )
    values = _run(CellElement, burgers)
    assert values[-1].sum() - values[0].sum() == pytest.approx(0.2 * CELLS * 1.0, rel=1e-9)


@pytest.mark.parametrize(
    ('arguments', 'message'),
    [
        ({'state': '1u'}, 'no name'),
        ({'flux': 'quadratic'}, 'function of the namespace'),
        ({'limiter': 'superbee'}, 'Unknown limiter'),
        ({'riemann': 'roe'}, 'Unknown numerical flux'),
    ],
)
def test_check_rejects_invalid_arguments(arguments, message):
    """Test that `check` rejects a bad state, function, limiter, or numerical flux."""
    with pytest.raises(ValidationError, match=message):
        ConservationInteraction('conservation', target='line', **arguments).check()
