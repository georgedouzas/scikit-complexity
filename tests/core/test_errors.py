"""Test the `errors` module of `skcomplex`."""

import re

import pytest

from skcomplex.core import errors
from skcomplex.core.errors import (
    AnalysisError,
    CapacityError,
    ConservationError,
    EnumerationLimitError,
    InfeasibleError,
    LabelError,
    MissingExtraError,
    SimulationError,
    SkcomplexError,
    SolverError,
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
    ],
)
def test_errors_refine_their_bases(error, base):
    """Test that the errors refine the errors they are a kind of, so callers catch either."""
    assert issubclass(error, base)


def test_named_errors_of_linear_programs_derive_from_their_families():
    """Test the place of the errors of linear programs, enumerations, and optional extras in the hierarchy."""
    assert issubclass(InfeasibleError, SolverError)
    assert issubclass(EnumerationLimitError, AnalysisError)
    assert issubclass(MissingExtraError, SkcomplexError)


def test_every_error_is_raised_in_the_library(request):
    """Test that the library builds every error class, or a subclass of it, where it reports a failure."""
    paths = (request.config.rootpath / 'src' / 'skcomplex').rglob('*.py')
    source = '\n'.join(path.read_text(encoding='utf-8') for path in paths if path.name != 'errors.py')
    unraised = [
        name
        for name, item in vars(errors).items()
        if isinstance(item, type)
        and issubclass(item, SkcomplexError)
        and not item.__subclasses__()
        and not re.search(rf'(?<!class )\b{name}\(|\berror={name}\b', source)
    ]
    assert unraised == []
