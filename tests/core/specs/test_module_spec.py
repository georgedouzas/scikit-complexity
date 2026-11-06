"""Test the `_module_spec` module of `skcomplex.core.specs`."""

import pytest

from skcomplex.core.errors import RewritingError
from skcomplex.core.specs import ModuleSpec


@pytest.mark.parametrize('spec', [ModuleSpec('')])
def test_invalid_fields_raise_rewriting_errors(spec):
    """Test that a module rejects invalid fields with a rewriting error."""
    with pytest.raises(RewritingError):
        spec.prepare()


def test_parse_word_reads_strings_modules_and_declared_names():
    """Test that a word is read as modules, keeping declared names of several characters whole."""
    module = ModuleSpec('A', (1.0,))
    assert [item.name for item in ModuleSpec.parse_word('F+')] == ['F', '+']
    assert ModuleSpec.parse_word(module) == [module]
    assert [item.name for item in ModuleSpec.parse_word(['stem', module, '+-'], ['stem'])] == ['stem', 'A', '+', '-']
