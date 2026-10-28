"""Test the `types` module of `skcomplex`."""

import typing

from sage.all__sagemath_symbolics import Expression
from sage.modules.free_module_element import FreeModuleElement

from skcomplex.specs import ModuleSpec
from skcomplex.types import Letters, Process, Target


def test_targets_are_symbolic_expressions_or_vectors():
    """Test that a law targets a symbolic expression or a vector of them."""
    assert typing.get_args(Target.__value__) == (Expression, FreeModuleElement)


def test_processes_and_letters_name_their_alternatives():
    """Test that a process may be nothing, and letters may be modules or sequences of them."""
    assert type(None) in typing.get_args(Process.__value__)
    assert ModuleSpec in typing.get_args(Letters.__value__)
