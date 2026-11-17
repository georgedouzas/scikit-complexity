"""Test the `_conservation_laws` module of `skcomplex.core.engine.compile.passes`."""

from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.components.elements import Element, FieldElement
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import divergence
from skcomplex.core.components.spaces import FiniteVolumeSpec, LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import StateSpec

CELLS = 40


class BurgersElement(Element):
    """Cell of a Burgers field with a constant source."""

    def __init__(self, label: str, *, u: Annotated[object, StateSpec()] = 0.0) -> None:
        super().__init__(label)
        self.u = u

    def _get_laws(self, s):
        """Return the divergence of the Burgers flux plus a source in one rate."""
        return [RateLaw(s.u, -divergence([s.u**2 / 2]) + 0.2)]


class SquaredElement(BurgersElement):
    """Cell whose rate squares a divergence, which is no conservation law."""

    def _get_laws(self, s):
        """Return the square of the divergence of the Burgers flux."""
        return [RateLaw(s.u, divergence([s.u**2 / 2]) ** 2)]


def _field(element, scheme=None):
    """Return a periodic finite-volume field that starts from a sine wave."""
    centers = (np.arange(CELLS) + 0.5) * 2 * np.pi / CELLS
    return FieldElement(
        'line',
        element=element,
        space=LatticeSpace((CELLS,), spacing=2 * np.pi / CELLS, boundary='periodic'),
        values={'u': 1.0 + 0.5 * np.sin(centers)},
        discretization='finite-volume',
        scheme=scheme,
    )


def test_rate_with_a_divergence_compiles_to_a_finite_volume_kernel_and_a_source():
    """Test that the divergence becomes the flux of the field and the other term a source that adds to the total."""
    problem = PassCompiler().prepare(CompositeSystem(elements=[_field(BurgersElement)])).run().problem_
    assert len(problem.volumes) == 1
    values = simulate(CompositeSystem(elements=[_field(BurgersElement)]), until=1.0, dt=0.1).get_values('line.u')
    assert values[-1].sum() - values[0].sum() == pytest.approx(0.2 * CELLS, rel=1e-9)


def test_scheme_of_the_field_chooses_the_numerical_flux():
    """Test that the Godunov flux of the scheme gives another run than the Rusanov flux, with the same total."""
    godunov = _field(BurgersElement, FiniteVolumeSpec(riemann='godunov'))
    first = simulate(CompositeSystem(elements=[_field(BurgersElement)]), until=1.0, dt=0.1).get_values('line.u')
    second = simulate(CompositeSystem(elements=[godunov]), until=1.0, dt=0.1).get_values('line.u')
    assert not np.allclose(first[-1], second[-1])
    assert second[-1].sum() == pytest.approx(first[-1].sum(), rel=1e-9)


def test_rate_that_is_no_conservation_law_is_rejected():
    """Test that a divergence inside a product says how to write the transport."""
    with pytest.raises(ValidationError, match='-divergence'):
        PassCompiler().prepare(CompositeSystem(elements=[_field(SquaredElement)])).run()
