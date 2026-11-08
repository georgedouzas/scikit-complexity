"""Define what a system compiles to: the problem, its kernels, and the frames the runtime hands to observers.

The problems of the library's forms and their kernels are private. The bases, the problem of the shared pass core, and
the frames of a run stay public for the observers and the analyses that read them, and for a system package that brings
a form of its own.
"""

from ._base import BaseKernel, BasePrinter, BaseSearch
from ._base_problem_spec import BaseProblemSpec
from ._frame_spec import FrameSpec
from ._problem_spec import ProblemSpec

__all__: list[str] = [
    'BaseKernel',
    'BasePrinter',
    'BaseProblemSpec',
    'BaseSearch',
    'FrameSpec',
    'ProblemSpec',
]
