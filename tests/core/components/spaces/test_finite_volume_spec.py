"""Test the `_finite_volume_spec` module of `skcomplex.core.components.spaces`."""

from typing import Annotated

import pytest

from skcomplex.core.components.elements import Element, FieldElement
from skcomplex.core.components.spaces import FiniteVolumeSpec, LatticeSpace
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import StateSpec


class CellElement(Element):
    """Cell of a field with one state."""

    def __init__(self, label: str, *, u: Annotated[object, StateSpec()] = 0.0) -> None:
        super().__init__(label)
        self.u = u


@pytest.mark.parametrize(
    ('spec', 'message'),
    [
        (FiniteVolumeSpec(limiter='superbee'), 'Unknown limiter'),
        (FiniteVolumeSpec(riemann='roe'), 'Unknown numerical flux'),
        (FiniteVolumeSpec(riemann='godunov', speed=lambda t, values: values['u']), 'takes no `speed`'),
    ],
)
def test_field_rejects_an_invalid_scheme(spec, message):
    """Test that a finite-volume field rejects an unknown limiter or flux, or a speed of the Godunov flux."""
    field = FieldElement(
        'line',
        element=CellElement,
        space=LatticeSpace(8),
        discretization='finite-volume',
        scheme=spec,
    )
    with pytest.raises(ValidationError, match=message):
        field.prepare()


def test_field_of_another_discretization_takes_no_scheme():
    """Test that a finite-difference field rejects a finite-volume scheme."""
    field = FieldElement('line', element=CellElement, space=LatticeSpace(8), scheme=FiniteVolumeSpec())
    with pytest.raises(ValidationError, match="discretization='finite-volume'"):
        field.prepare()


def test_lattice_checks_the_scheme_of_its_finite_volume_fields():
    """Test that a lattice takes a scheme only with the finite-volume discretization, and checks it."""
    with pytest.raises(ValidationError, match="discretization='finite-volume'"):
        LatticeSpace(8, scheme=FiniteVolumeSpec()).prepare()
    with pytest.raises(ValidationError, match='Unknown numerical flux'):
        LatticeSpace(8, discretization='finite-volume', scheme=FiniteVolumeSpec(riemann='roe')).prepare()
