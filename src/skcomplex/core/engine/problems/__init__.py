"""Define what a system compiles to: a problem, its kernels, and run frames.

The bases, the problem of the shared compiler, and the frames stay public for observers, analyses, and system packages
that bring a form of their own. The problems of the forms of the library are private.
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
