"""Compile systems into problems of one mathematical form, with numeric functions.

`simulate` chooses the compiler of the form a system holds. A system package that brings a form of its own writes its
compiler on `BaseCompiler`, with passes on `BasePass` and lowerings on `BaseLowering`.
"""

from ._base import BaseCompiler, BaseLowering, BasePass
from ._issue_spec import IssueSpec
from ._phase_spec import PhaseSpec
from ._phases_schedule_spec import PhasesScheduleSpec
from ._validation_report_spec import ValidationReportSpec
from .passes._hilbert_space_error import HilbertSpaceError
from .passes._operator_compiler import OperatorCompiler

__all__: list[str] = [
    'BaseCompiler',
    'BaseLowering',
    'BasePass',
    'HilbertSpaceError',
    'IssueSpec',
    'OperatorCompiler',
    'PhaseSpec',
    'PhasesScheduleSpec',
    'ValidationReportSpec',
]
