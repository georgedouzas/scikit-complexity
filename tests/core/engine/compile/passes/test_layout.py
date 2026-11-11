"""Test the `_layout` module of `skcomplex.core.engine.compile.passes`."""

from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from skcomplex.core.components.elements import Element, FieldElement
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import StateSpec, VisualSpec

from ..conftest import PhaseElement, QuadraticElement


class CountElement(Element):
    """Amount that cannot be negative beside a count of whole events."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        amount: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        events: Annotated[float, StateSpec(domain=NN)] = 0,
        level: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.amount = amount
        self.events = events
        self.level = level

    def _get_laws(self, s):
        """Return constant states."""
        return [UpdateLaw(s.amount, s.amount), UpdateLaw(s.events, s.events), UpdateLaw(s.level, s.level)]


def test_complex_state_reads_back_its_initial_value():
    """Test that a complex default is stored as a pair of floats and read back as a complex number."""
    problem = PassCompiler().prepare(CompositeSystem(elements=[PhaseElement('osc', z=1 + 2j)])).run().problem_
    assert problem.initial_state.tolist() == [1.0, 2.0]
    assert problem.states.read(problem.initial_state, 'osc.z') == 1 + 2j


def test_real_state_rejects_complex_value():
    """Test that a complex value of a real state names the state, its domain, and the complex domain."""
    with pytest.raises(ValidationError, match=r'osc\.omega = 1j.*its domain Real Field.*domain=CC'):
        PassCompiler().prepare(CompositeSystem(elements=[PhaseElement('osc', omega=1j)])).run()


@pytest.mark.parametrize(
    ('values', 'message'),
    [
        ({'amount': -1.0}, r'a\.amount = -1\.0 is not in its domain \[0, \+oo\)\.'),
        ({'events': 1.5}, r'a\.events = 1\.5 is not in its domain Non negative integer semiring\.'),
    ],
)
def test_real_value_outside_real_domain_is_reported_outside_it(values, message):
    """Test that a real value outside a real domain is reported as outside it, with no word of an imaginary part."""
    with pytest.raises(ValidationError, match=message) as error:
        PassCompiler().prepare(CompositeSystem(elements=[CountElement('a', **values)])).run()
    assert 'imaginary' not in str(error.value)


def test_complex_value_of_real_state_names_its_imaginary_part():
    """Test that a complex value of a real state is reported as having a nonzero imaginary part."""
    with pytest.raises(ValidationError, match=r'a\.level = \(1\+2j\) has a nonzero imaginary part'):
        PassCompiler().prepare(CompositeSystem(elements=[CountElement('a', level=1 + 2j)])).run()


def test_real_state_accepts_complex_value_with_zero_imaginary_part():
    """Test that a complex value with a zero imaginary part is a real value."""
    problem = PassCompiler().prepare(CompositeSystem(elements=[PhaseElement('osc', omega=3 + 0j)])).run().problem_
    assert problem.parameters.read(problem.initial_parameters, 'osc.omega') == 3.0


def test_complex_parameter_per_site_is_read_and_updated():
    """Test that every site reads its own complex parameter, and that an update changes the next step."""
    c = np.arange(16).reshape(4, 4) * (1 + 1j)
    grid = FieldElement('grid', element=QuadraticElement, space=LatticeSpace((4, 4)), values={'c': c})
    trajectory = simulate(CompositeSystem(elements=[grid]), until=1, dt=1)
    assert np.allclose(trajectory.get_lattice('grid.z')[1], c)
    grid.set_params(values={'c': np.full((4, 4), 2j)})
    trajectory = simulate(CompositeSystem(elements=[grid]), until=1, dt=1)
    assert np.allclose(trajectory.get_lattice('grid.z')[1], 2j)
