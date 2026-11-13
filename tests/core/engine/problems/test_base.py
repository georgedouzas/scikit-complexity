"""Test the `_base` module of `skcomplex.core.engine.problems`."""

import enum
import inspect

from skcomplex.core.engine import problems as package
from skcomplex.core.engine.problems import BaseKernel, BasePrinter, BaseProblemSpec, BaseSearch
from skcomplex.core.engine.problems._source_loader import SourceLoader
from skcomplex.core.specs import BaseRecordSpec, BaseSpec, BaseStructureSpec

BASES = (BaseSpec, BaseKernel, BaseSearch, BasePrinter, SourceLoader)


def test_public_names_are_classes_under_the_bases():
    """Every public name is a class derived from a base of the package, the source loader, or an enumeration."""
    for name in package.__all__:
        item = getattr(package, name)
        assert inspect.isclass(item), name
        assert issubclass(item, BASES) or issubclass(item, enum.Enum), name


def test_value_bases_derive_from_the_value_base():
    """The problem, record, and structure bases are values."""
    assert issubclass(BaseProblemSpec, BaseSpec)
    assert issubclass(BaseRecordSpec, BaseSpec)
    assert issubclass(BaseStructureSpec, BaseSpec)


def test_kernel_search_and_printer_bases_are_configurable():
    """The kernel, search, and printer bases have the two main public methods of every configurable class."""
    for base in (BaseKernel, BaseSearch, BasePrinter):
        assert all(callable(getattr(base, method)) for method in ('prepare', 'run'))


def test_public_problems_are_the_bases_frames_and_two_forms():
    """The bases, the problems of the shared pass core and the operator form, and the frames are public."""
    assert set(package.__all__) == {
        'BaseKernel',
        'BasePrinter',
        'BaseProblemSpec',
        'BaseSearch',
        'FrameSpec',
        'OperatorProblemSpec',
        'ProblemSpec',
    }
