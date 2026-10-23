"""Test the `_symbols` module of `skcomplex.core.laws`."""

import pytest

from skcomplex.core.laws import Symbols


def test_symbols_unknown_name_lists_available_names():
    """Test that a missing symbol names the available ones."""
    with pytest.raises(AttributeError, match='Available names are: a'):
        _ = Symbols(a=1).b
