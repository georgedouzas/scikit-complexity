"""Compile systems into problems that solvers run.

A prepared system hands its components to a compiler, which checks them and turns their laws into one problem with
numeric functions, such as the two equations of `LotkaVolterraSystem`. A system package that brings a kind of problem
of its own writes its compiler on `BaseCompiler`, with passes on `BasePass` and lowerings on `BaseLowering`.
"""

from ._base import BaseCompiler, BaseLowering, BasePass
from ._base import _Composite as _Composite
from ._compilation import _Issue as _Issue
from ._phase_spec import PhaseSpec
from ._phases_schedule_spec import PhasesScheduleSpec
from .passes._flatten import FlattenPass
from .passes._hilbert_space_error import HilbertSpaceError
from .passes._operator_compiler import OperatorCompiler

__all__: list[str] = [
    'BaseCompiler',
    'BaseLowering',
    'BasePass',
    'FlattenPass',
    'HilbertSpaceError',
    'OperatorCompiler',
    'PhaseSpec',
    'PhasesScheduleSpec',
]
