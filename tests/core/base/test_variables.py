"""Test the declarations of states, parameters, and roles."""

import pytest

from skcomplex.core.base import VariableKind, declarations_of, parameter
from skcomplex.core.components import Element


class Valve(Element):
    """Element with a bounded control input."""

    opening: float = parameter(0.5, adaptive=True, bounds=(0, 1))


def test_parameter_bounds_declare_a_bounded_adaptive_parameter():
    """Test that bounds of an adaptive parameter are stored as floats on its state declaration."""
    declaration = declarations_of(Valve, VariableKind.STATE)['opening']
    assert declaration.adaptive
    assert declaration.bounds == (0.0, 1.0)
    assert isinstance(declaration.bounds[0], float)


@pytest.mark.parametrize(
    ('arguments', 'message'),
    [
        ({'bounds': (0, 1)}, 'Only an adaptive parameter takes bounds'),
        ({'adaptive': True, 'bounds': (1, 0)}, 'lower at most upper'),
    ],
)
def test_parameter_bounds_reject_invalid_declarations(arguments, message):
    """Test that bounds without adaptivity or with a lower bound above the upper one are rejected."""
    with pytest.raises(ValueError, match=message):
        parameter(0.5, **arguments)
