"""Test the `_base` module of `skcomplex.core.engine.problems`."""

import enum
import inspect

from skcomplex.core.engine import problems as package
from skcomplex.core.engine.problems import (
    BaseKernel,
    BasePrinter,
    BaseProblemSpec,
    BaseRecordSpec,
    BaseSearch,
)
from skcomplex.core.engine.problems._source_loader import SourceLoader
from skcomplex.core.specs import BaseSpec

BASES = (BaseSpec, BaseKernel, BaseSearch, BasePrinter, SourceLoader)


def test_public_names_are_classes_under_the_bases():
    """Every public name is a class derived from a base of the package, the source loader, or an enumeration."""
    for name in package.__all__:
        item = getattr(package, name)
        assert inspect.isclass(item), name
        assert issubclass(item, BASES) or issubclass(item, enum.Enum), name


def test_value_bases_derive_from_the_value_base():
    """The problem and record bases are values."""
    assert issubclass(BaseProblemSpec, BaseSpec)
    assert issubclass(BaseRecordSpec, BaseSpec)


def test_kernel_search_and_printer_bases_are_configurable():
    """The kernel, search, and printer bases have the two main public methods of every configurable class."""
    for base in (BaseKernel, BaseSearch, BasePrinter):
        assert all(callable(getattr(base, method)) for method in ('prepare', 'run'))


def test_public_problems_are_the_bases_frames_and_two_forms():
    """The bases, the problems of the shared pass core and the operator form, the frames, and values are public.

    The printer of expressions is public too, for the kernels of system packages.
    """
    expected = {
        'BaseKernel',
        'ExpressionPrinter',
        'BasePrinter',
        'BaseProblemSpec',
        'BaseSearch',
        'FrameSpec',
        'OperatorProblemSpec',
        'ProblemSpec',
    }
    assert expected <= set(package.__all__)
    assert all(name.endswith('Spec') for name in set(package.__all__) - expected)
