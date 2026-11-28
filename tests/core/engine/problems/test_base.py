"""Test the `_base` module of `skcomplex.core.engine.problems`."""

import dataclasses
import enum
import inspect

import numpy as np

from skcomplex.core.base import BaseSpec
from skcomplex.core.engine import problems as package
from skcomplex.core.engine.problems import (
    BaseKernel,
    BasePrinter,
    BaseProblemSpec,
    BaseRecordSpec,
    BaseSearch,
    RunStateSpec,
)
from skcomplex.core.engine.problems._source_loader import SourceLoader

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
    """The bases, the shared problems, the operator form and kernel, the frames, and values are public.

    The printer of expressions is public too, for the kernels of system packages.
    """
    expected = {
        'BaseKernel',
        'ExpressionPrinter',
        'BasePrinter',
        'BaseProblemSpec',
        'BaseSearch',
        'FrameSpec',
        'OperatorKernel',
        'OperatorProblemSpec',
        'ProblemSpec',
    }
    assert expected <= set(package.__all__)
    assert all(name.endswith('Spec') for name in set(package.__all__) - expected)


def test_public_records_are_the_solver_contract():
    """The public records are the forms, what a backend is run with, their field records, and the run frames.

    The run frames are what runs return to users.
    """
    expected = {
        'AgentSnapshotSpec',
        'BaseProblemSpec',
        'BaseRecordSpec',
        'CompiledElementSpec',
        'DiffusiveLimitSpec',
        'FrameSpec',
        'KernelSpec',
        'KineticProblemSpec',
        'LayoutSpec',
        'LearningProblemSpec',
        'LinearTermSpec',
        'LoopTermSpec',
        'LoweredKernelsSpec',
        'NetworkSnapshotSpec',
        'OperatorProblemSpec',
        'PartitionSpec',
        'ProblemSpec',
        'ReferenceSpec',
        'RunStateSpec',
        'TransferNetworkSpec',
    }
    assert {name for name in package.__all__ if dataclasses.is_dataclass(getattr(package, name))} == expected


class DecayKernel(BaseKernel):
    """A kernel of the decay rate `-r x`."""

    def __init__(self, rate=1.0):
        self.rate = rate

    def _run(self, inputs):
        self.values_ = -self.rate * inputs.x


def test_a_kernel_keeps_what_it_computes_at_a_run_state():
    """A kernel keeps the values it computes at a run state in `values_`."""
    state = RunStateSpec(0.0, np.array([2.0]), np.zeros(0))
    assert DecayKernel(0.5).prepare().run(state).values_.tolist() == [-1.0]
