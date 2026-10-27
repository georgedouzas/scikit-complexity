"""Test the `errors` module of `skcomplex.core`."""

import pytest

from skcomplex.core import errors
from skcomplex.core.errors import (
    ArgumentError,
    CapacityError,
    ConservationError,
    LabelError,
    NotCheckedError,
    SimulationError,
    SkcomplexError,
    SpaceError,
    SpaceTooLargeError,
    ValidationError,
)


def test_every_error_derives_from_the_base_of_the_library():
    """Test that every exception of the library derives from the base exception, and the warnings from warnings."""
    for name in dir(errors):
        value = getattr(errors, name)
        if not isinstance(value, type) or value.__module__ != errors.__name__:
            continue
        expected = UserWarning if name.endswith('Warning') else SkcomplexError
        assert issubclass(value, expected), name


@pytest.mark.parametrize(
    ('error', 'base'),
    [
        (ConservationError, ValidationError),
        (LabelError, ValidationError),
        (SpaceError, ValidationError),
        (SpaceTooLargeError, SpaceError),
        (CapacityError, SimulationError),
        (ArgumentError, TypeError),
        (NotCheckedError, SkcomplexError),
    ],
)
def test_errors_refine_their_bases(error, base):
    """Test that the errors refine the errors they are a kind of, so callers catch either."""
    assert issubclass(error, base)
