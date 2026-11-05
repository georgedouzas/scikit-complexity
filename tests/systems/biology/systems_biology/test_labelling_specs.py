"""Test the `_labelling_specs` module of `skcomplex.systems.biology.systems_biology`."""

import pytest

from skcomplex.core.errors import ValidationError
from skcomplex.systems.biology.systems_biology import AtomMapSpec


def test_origins_of_a_valid_map():
    """Test that the map records where atoms of citrate come from."""
    atoms = AtomMapSpec('v1', [('AcCoA', 'ab'), ('OAC', 'cdef')], [('Cit', 'fedbac')])
    assert atoms.origins[('Cit', 1)] == ('OAC', 4)
    assert atoms.origins[('Cit', 5)] == ('AcCoA', 1)


def test_check_names_an_unbalanced_letter():
    """Test that a product letter no reactant carries raises an error that names the reaction and the letter."""
    with pytest.raises(ValidationError, match=r"reaction v1.*'z'"):
        AtomMapSpec('v1', [('A', 'ab')], [('B', 'az')]).prepare()
