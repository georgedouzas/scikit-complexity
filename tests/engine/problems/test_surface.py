"""Test that the public names of the problems package are classes under its bases."""

import enum
import inspect

from skcomplex.engine import problems as package
from skcomplex.engine.problems import (
    BaseKernel,
    BasePrinter,
    BaseProblem,
    BaseRecord,
    BaseSearch,
    BaseStructure,
    BaseValue,
)

BASES = (BaseValue, BaseKernel, BaseSearch, BasePrinter)


def test_public_names_are_classes_under_the_bases():
    """Every public name is a class derived from a base of the package, or an enumeration of values."""
    for name in package.__all__:
        item = getattr(package, name)
        assert inspect.isclass(item), name
        assert issubclass(item, BASES) or issubclass(item, enum.Enum), name


def test_value_bases_derive_from_the_value_base():
    """The problem, record, and structure bases are values."""
    assert issubclass(BaseProblem, BaseValue)
    assert issubclass(BaseRecord, BaseValue)
    assert issubclass(BaseStructure, BaseValue)


def test_kernel_search_and_printer_bases_are_configurable():
    """The kernel, search, and printer bases speak the three verbs of every configurable class."""
    for base in (BaseKernel, BaseSearch, BasePrinter):
        assert all(callable(getattr(base, verb)) for verb in ('check', 'prepare', 'run'))
